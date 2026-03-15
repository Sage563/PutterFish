from time import time
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import chess

from eval import predict_eval


DEFAULT_C_PUCT = 2.0
DEFAULT_VIRTUAL_LOSS = 3
MAX_WORKER_THREADS = 64
DEFAULT_EVAL_TO_WINPROB_SCALE = 0.5
DEFAULT_EVAL_TO_CP = 500.0

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2


def _zobrist_key(board: chess.Board) -> int:
    """64-bit position key from FEN (stable within process)."""
    h = hash(board.fen())
    return h & 0xFFFFFFFFFFFFFFFF


class TranspositionTable:
    """Locked transposition table for alpha-beta; size in MB from UCI Hash option."""

    def __init__(self, size_mb: int = 128):
        # ~24 bytes per entry: key(8) + depth(2) + score(4) + flag(1) + padding
        self.size = max(1024, min(2**24, (size_mb * 1024 * 1024) // 24))
        self.mask = self.size - 1
        self.entries = [None] * self.size  # (key, depth, score, flag)
        self.lock = threading.Lock()

    def clear(self):
        with self.lock:
            self.entries = [None] * self.size

    def probe(self, key: int, depth: int, alpha: float, beta: float):
        """Return (score, ok) if we can use the entry for a cutoff or exact score."""
        idx = key & self.mask
        with self.lock:
            e = self.entries[idx]
        if e is None or e[0] != key or e[1] < depth:
            return None, False
        _, _, score, flag = e[0], e[1], e[2], e[3]
        if flag == TT_EXACT:
            return score, True
        if flag == TT_LOWER and score >= beta:
            return score, True
        if flag == TT_UPPER and score <= alpha:
            return score, True
        return score, False

    def store(self, key: int, depth: int, score: float, flag: int):
        idx = key & self.mask
        with self.lock:
            self.entries[idx] = (key, depth, score, flag)

# ELO scaling: at 2500 use full strength; below that scale time/depth and add move noise
ELO_MIN, ELO_MAX = 400, 3000
ELO_REF = 2500


class MCTSNode:
    __slots__ = ("board", "parent", "move", "children", "untried_moves", "wins", "visits", "virtual_visits", "prior")

    def __init__(self, board: chess.Board, parent=None, move=None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.untried_moves = list(board.legal_moves)
        self.wins = 0.0
        self.visits = 0
        self.virtual_visits = 0
        self.prior = prior

    @property
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    @property
    def is_terminal(self):
        return self.board.is_game_over()

    def effective_visits(self):
        return self.visits + self.virtual_visits


class CPutterfish:
    """Chess engine: MCTS + neural eval, pondering, ELO strength."""

    def __init__(self, board: chess.Board, depth: int = 3, time_limit: float = 2.0):
        self.board = board
        self.time_limit = time_limit
        self.max_rollout = depth
        self.start_time = time()
        self.model_path = "models/model.pth"
        self._threads = 1
        self._info_callback = None
        self._stop_requested = False
        self._pondering = False
        self._elo = ELO_REF
        self._tt = TranspositionTable(128)
        self._c_puct = DEFAULT_C_PUCT
        self._virtual_loss = DEFAULT_VIRTUAL_LOSS
        self._eval_to_winprob_scale = DEFAULT_EVAL_TO_WINPROB_SCALE
        self._eval_to_cp = DEFAULT_EVAL_TO_CP
        self._verbose_info = False

    def set_info_callback(self, callback):
        self._info_callback = callback

    def request_stop(self):
        self._stop_requested = True

    def clear_stop(self):
        self._stop_requested = False

    def _time_elapsed(self) -> float:
        return time() - self.start_time

    def _time_remaining(self) -> float:
        return max(0.0, self.time_limit - self._time_elapsed())

    def _scale_for_elo(self, value: float, is_time: bool = False) -> float:
        """Scale value down for lower ELO (weaker = less time/depth)."""
        if self._elo >= ELO_REF:
            return value
        # Linear scale: at 400 ELO use ~15% of value, at 2500 use 100%
        t = (self._elo - ELO_MIN) / (ELO_REF - ELO_MIN)
        t = max(0.0, min(1.0, t))
        if is_time:
            return value * (0.15 + 0.85 * t)
        return value * (0.3 + 0.7 * t)

    def _apply_elo_noise_to_choice(self, root: MCTSNode) -> tuple:
        """Return (best_move, ponder_move) possibly downgraded for ELO (pick 2nd/3rd best)."""
        if not root.children:
            return None, None
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        best_child = sorted_children[0]
        best_move = best_child.move
        ponder_move = None
        if best_child.children:
            ponder_move = max(best_child.children, key=lambda c: c.visits).move
        if self._elo >= ELO_REF or len(sorted_children) < 2:
            return best_move, ponder_move
        t = (self._elo - ELO_MIN) / (ELO_REF - ELO_MIN)
        t = max(0.0, min(1.0, t))
        noise_prob = (1.0 - t) * 0.35
        if random.random() < noise_prob and len(sorted_children) >= 2:
            idx = 1 if len(sorted_children) < 3 or random.random() < 0.75 else 2
            weaker = sorted_children[idx]
            best_move = weaker.move
            ponder_move = None
            if weaker.children:
                ponder_move = max(weaker.children, key=lambda c: c.visits).move
        return best_move, ponder_move

    def _select(self, node: MCTSNode, root_player: chess.Color, virtual_loss: int = 0):
        best_score = float("-inf")
        best_child = None
        n_parent = node.effective_visits()
        if n_parent <= 0:
            n_parent = 1
        log_n = math.sqrt(n_parent)

        for child in node.children:
            n_child = child.visits + (virtual_loss if child.virtual_visits > 0 else 0)
            if n_child == 0:
                p = max(1e-6, child.prior if child.prior > 0 else 1.0)
                q = 0.5
                u = self._c_puct * p * log_n / 1.0
            else:
                q = child.wins / child.visits
                p = max(1e-6, child.prior if child.prior > 0 else 1.0)
                u = self._c_puct * p * log_n / (1.0 + n_child)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _expand(self, node: MCTSNode, sort_by_eval: bool = True, lock: threading.Lock = None):
        if lock is not None:
            lock.acquire()
        try:
            if not node.untried_moves:
                return None
            # At high depth, skip sort to save NN calls (we get more nodes instead)
            if sort_by_eval and self.max_rollout < 18 and len(node.untried_moves) > 1:
                def eval_after(m):
                    b = node.board.copy()
                    b.push(m)
                    return self.evaluate_board(b)

                reverse = node.board.turn == chess.WHITE
                node.untried_moves.sort(key=eval_after, reverse=reverse)
            move = node.untried_moves.pop()
            new_board = node.board.copy()
            new_board.push(move)
            child = MCTSNode(new_board, parent=node, move=move)
            node.children.append(child)
            return child
        finally:
            if lock is not None:
                lock.release()

    def evaluate_board(self, board: chess.Board) -> float:
        return predict_eval(self.model_path, board.fen())

    def _negamax_depth_for_simulation(self) -> int:
        """Use cheaper leaf eval at high Depth so we can run depth 25 fast (more nodes, less work per node)."""
        if self.max_rollout >= 20:
            return 0   # NN only: maximum nodes/sec
        if self.max_rollout >= 14:
            return 1   # 1-ply: fast, still some lookahead
        return 2       # 2-ply: best quality when depth is low

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, tt: TranspositionTable = None) -> float:
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        key = _zobrist_key(board)
        if tt is not None:
            score, ok = tt.probe(key, depth, alpha, beta)
            if ok:
                return score
        max_score = float("-inf")
        for move in board.legal_moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, tt=tt)
            board.pop()
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        if tt is not None:
            if max_score <= alpha - 1e-6:
                flag = TT_UPPER
            elif max_score >= beta:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            tt.store(key, depth, max_score, flag)
        return max_score

    def evaluate(self, board: chess.Board) -> float:
        depth = self._negamax_depth_for_simulation()
        score = self.negamax(
            board, depth=depth, alpha=float("-inf"), beta=float("inf"), tt=self._tt
        )
        return 1.0 / (1.0 + math.exp(-score / self._eval_to_winprob_scale))

    def _simulate(self, board: chess.Board) -> float:
        return self.evaluate(board)

    def _backpropagate(self, node: MCTSNode, result: float, root_player: chess.Color, virtual_delta: int = 0, lock: threading.Lock = None):
        if lock is not None:
            lock.acquire()
        try:
            while node is not None:
                node.visits += 1
                if virtual_delta != 0:
                    node.virtual_visits = max(0, node.virtual_visits + virtual_delta)
                perspective = result if (node.board.turn == root_player) else (1.0 - result)
                node.wins += perspective
                node = node.parent
        finally:
            if lock is not None:
                lock.release()

    def _run_simulation(self, root: MCTSNode, root_player: chess.Color, virtual_loss: int, lock: threading.Lock = None) -> bool:
        node = root
        path = []
        while node.is_fully_expanded and not node.is_terminal:
            if self._stop_requested:
                return False
            child = self._select(node, root_player, virtual_loss)
            if child is None:
                break
            path.append(child)
            node = child
        if not node.is_terminal and node.untried_moves:
            node = self._expand(node, sort_by_eval=True, lock=lock)
            if node is not None:
                path.append(node)
        if node is None:
            return False
        if lock is not None:
            lock.acquire()
        try:
            for n in path:
                n.virtual_visits += virtual_loss
        finally:
            if lock is not None:
                lock.release()
        result = self._simulate(node.board)
        self._backpropagate(node, result, root_player, virtual_delta=-virtual_loss, lock=lock)
        return True

    def _collect_pv(self, root: MCTSNode, max_plies: int = 10) -> list:
        """Principal variation: best move sequence from root."""
        pv = []
        node = root
        for _ in range(max_plies):
            if not node.children:
                break
            best = max(node.children, key=lambda c: c.visits)
            pv.append(best.move.uci())
            node = best
        return pv

    def select_move(self, options=None):
        if options is None:
            options = {}
        self.model_path = options.get("Model", options.get("model", self.model_path))
        self._elo = int(options.get("Elo", options.get("UCI_Elo", ELO_REF)))
        self._elo = max(ELO_MIN, min(ELO_MAX, self._elo))
        depth_opt = int(options.get("Depth", self.max_rollout))
        self.max_rollout = max(1, int(self._scale_for_elo(depth_opt)))
        self._threads = int(options.get("Threads", 1))
        self._c_puct = float(options.get("CPuct", self._c_puct))
        self._virtual_loss = int(options.get("VirtualLoss", self._virtual_loss))
        self._eval_to_winprob_scale = float(options.get("EvalToWinProbScale", self._eval_to_winprob_scale))
        self._eval_to_cp = float(options.get("EvalToCP", self._eval_to_cp))
        self._verbose_info = bool(options.get("VerboseInfo", self._verbose_info))
        self.clear_stop()
        self._pondering = bool(options.get("Ponder", False))
        hash_mb = max(1, min(65536, int(options.get("Hash", 128))))
        if getattr(self, "_hash_mb", 0) != hash_mb:
            self._hash_mb = hash_mb
            self._tt = TranspositionTable(hash_mb)
        self._tt.clear()

        move_time_ms = options.get("MoveTime")
        wtime = options.get("wtime")
        btime = options.get("btime")
        move_overhead_ms = int(options.get("Move Overhead", 0))
        if move_time_ms is not None:
            self.time_limit = max(0.1, (move_time_ms - move_overhead_ms) / 1000.0)
        elif wtime is not None and btime is not None:
            t = wtime if self.board.turn == chess.WHITE else btime
            t = max(0, t - move_overhead_ms)
            inc = options.get("winc" if self.board.turn == chess.WHITE else "binc", 0)
            self.time_limit = (t / 20000.0) + (inc / 1000.0)
            self.time_limit = max(0.5, min(self.time_limit, t / 1000.0 * 0.95))
        else:
            self.time_limit = float(options.get("TimeLimit", 10))

        self.time_limit = self._scale_for_elo(self.time_limit, is_time=True)
        self.time_limit *= 0.95
        self.start_time = time()
        root = MCTSNode(self.board.copy())
        root_player = self.board.turn
        num_workers = max(1, min(self._threads, MAX_WORKER_THREADS))
        virtual_loss = self._virtual_loss if num_workers > 1 else 0
        tree_lock = threading.Lock() if num_workers > 1 else None
        nodes_lock = threading.Lock()
        nodes = [0]  # mutable so workers can update

        if not root.is_terminal and root.untried_moves:
            self._expand(root, sort_by_eval=True, lock=tree_lock)

        last_info = [0.0]

        def emit_info(root_node, nodes_count, now_sec):
            best_move = None
            best_visits = -1
            for c in root_node.children:
                if c.visits > best_visits:
                    best_visits = c.visits
                    best_move = c.move
            score_cp = None
            score_eval = None
            if root_node.children:
                best_child = max(root_node.children, key=lambda c: c.visits)
                q = best_child.wins / best_child.visits
                q = min(1.0 - 1e-5, max(1e-5, q))
                score_eval = self._eval_to_winprob_scale * math.log(q / (1.0 - q))
                score_cp = int(score_eval * self._eval_to_cp)
            pv = self._collect_pv(root_node)
            nps = int(nodes_count / now_sec) if now_sec > 0 else 0
            currmove = None
            for c in root_node.children:
                if c.visits < best_visits and c.visits > 0:
                    currmove = c.move.uci()
                    break
            if currmove is None and root_node.untried_moves:
                currmove = root_node.untried_moves[-1].uci() if root_node.untried_moves else None
            if self._info_callback:
                info = {
                    "depth": self.max_rollout,
                    "seldepth": 1,
                    "nodes": nodes_count,
                    "time": int(now_sec * 1000),
                    "nps": nps,
                    "score": score_cp,
                    "pv": pv,
                    "currmove": currmove,
                    "currmovenumber": 1,
                }
                if self._verbose_info and score_eval is not None:
                    info["string"] = (
                        f"eval_raw {score_eval:.3f} winprob {q:.3f} "
                        f"time_left {self._time_remaining():.2f}s"
                    )
                self._info_callback(info)

        def run_one():
            if self._run_simulation(root, root_player, virtual_loss, lock=tree_lock):
                with nodes_lock:
                    nodes[0] += 1

        if num_workers <= 1:
            while self._time_remaining() > 0.02 and not self._stop_requested:
                run_one()
                now = self._time_elapsed()
                if now - last_info[0] >= 0.15:
                    last_info[0] = now
                    emit_info(root, nodes[0], now)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = set()
                while self._time_remaining() > 0.02 and not self._stop_requested:
                    while len(futures) < num_workers:
                        futures.add(executor.submit(run_one))
                    try:
                        f = next(as_completed(futures, timeout=0.15))
                        futures.discard(f)
                        try:
                            f.result()
                        except Exception:
                            pass
                    except (StopIteration, TimeoutError):
                        pass
                    now = self._time_elapsed()
                    if now - last_info[0] >= 0.15:
                        last_info[0] = now
                        emit_info(root, nodes[0], now)
                for f in list(futures):
                    try:
                        f.result(timeout=2.0)
                    except Exception:
                        pass

        nodes_final = nodes[0]

        best_move, ponder_move = self._apply_elo_noise_to_choice(root)
        best_visits = max((c.visits for c in root.children), default=0)
        elapsed = round(time() - self.start_time, 2)
        return {
            "time": elapsed,
            "move": best_move,
            "visits": best_visits,
            "nodes": nodes_final,
            "ponder_move": ponder_move,
            "pv": self._collect_pv(root),
        }
