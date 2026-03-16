import chess
import threading
from cputterfish import CPutterfish

# Standard and Chess 960 (Fischer Random) start positions
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# Chess 960 start position #0 (one of 960)
START_FEN_960 = "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w KQkq - 0 1"


def main():
    board = chess.Board()
    ai = CPutterfish(board, depth=3, time_limit=10)
    while not board.is_game_over():
        print(board)
        if board.turn == chess.WHITE:
            move = ai.select_move()["move"]
            if move is None:
                print("No legal moves available. Game over.")
                break
            print(f"AI selects move: {move}")
            board.push(move)
        else:
            user_move = input("Enter your move in UCI format (e.g., e2e4): ")
            try:
                move = chess.Move.from_uci(user_move)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move. Try again.")
            except ValueError:
                print("Invalid move format. Try again.")
    print("Game over.")


# Format: (name, type, default, min, max) for spin; (name, type, default, None, None) for check/string;
# (name, type, default, None, None, vars...) for combo
UCI_OPTIONS = [
    ("Hash", "spin", 128, 1, 65536),
    ("Threads", "spin", 1, 1, 128),
    ("Ponder", "check", False, None, None),
    ("Depth", "spin", 12, 1, 64),
    ("MoveTime", "spin", 5000, 100, 300000),
    ("TimeLimit", "spin", 10, 1, 3600),
    ("Move Overhead", "spin", 100, 0, 5000),
    ("Model", "string", "default", None, None),
    ("Elo", "spin", 2500, 400, 3000),
    ("UCI_CPutterfish_Strength", "spin", 100, 0, 100),
    ("UCI_Variant", "combo", "chess", None, None, "chess", "chess960"),
    ("NNCacheSize", "spin", 131072, 0, 999999999),
    ("Backend", "combo", "auto", None, None, "auto", "cpu", "cuda"),
    ("Device", "combo", "auto", None, None, "auto", "CPU only", "GPU only"),
    ("MultiPV", "spin", 1, 1, 500),
    ("CPuct", "spin", 2, 1, 100),
    ("VirtualLoss", "spin", 3, 0, 100),
    ("EvalToWinProbScale", "spin", 50, 1, 1000),
    ("EvalToCP", "spin", 500, 1, 5000),
    ("VerboseInfo", "check", False, None, None),
]


def parse_go_args(line: str) -> dict:
    args = {}
    parts = line.split()
    i = 1
    while i < len(parts):
        if parts[i] == "depth" and i + 1 < len(parts):
            args["Depth"] = int(parts[i + 1])
            i += 2
        elif parts[i] == "movetime" and i + 1 < len(parts):
            args["MoveTime"] = int(parts[i + 1])
            i += 2
        elif parts[i] == "wtime" and i + 1 < len(parts):
            args["wtime"] = int(parts[i + 1])
            i += 2
        elif parts[i] == "btime" and i + 1 < len(parts):
            args["btime"] = int(parts[i + 1])
            i += 2
        elif parts[i] == "winc" and i + 1 < len(parts):
            args["winc"] = int(parts[i + 1])
            i += 2
        elif parts[i] == "binc" and i + 1 < len(parts):
            args["binc"] = int(parts[i + 1])
            i += 2
        elif parts[i] == "ponder":
            args["Ponder"] = True
            i += 1
        else:
            i += 1
    return args


def parse_position(line: str, variant: str = "chess") -> tuple:
    """Parse position command; variant affects startpos for Chess 960."""
    parts = line.split()
    if len(parts) < 2:
        return START_FEN if variant != "chess960" else START_FEN_960, []
    i = 1
    if parts[i] == "startpos":
        fen = START_FEN_960 if variant == "chess960" else START_FEN
        i += 1
    elif parts[i] == "fen":
        i += 1
        fen_parts = []
        while i < len(parts) and parts[i] != "moves":
            fen_parts.append(parts[i])
            i += 1
        fen = " ".join(fen_parts)
    else:
        return START_FEN if variant != "chess960" else START_FEN_960, []
    moves = []
    if i < len(parts) and parts[i] == "moves":
        i += 1
        while i < len(parts):
            moves.append(parts[i])
            i += 1
    return fen, moves


def format_info(info: dict) -> str:
    """UCI 'info' line: depth, seldepth, nodes, time, nps, score, pv, currmove."""
    tokens = ["info"]
    if "depth" in info:
        tokens.append(f"depth {info['depth']}")
    if "seldepth" in info:
        tokens.append(f"seldepth {info['seldepth']}")
    if "nodes" in info:
        tokens.append(f"nodes {info['nodes']}")
    if "time" in info:
        tokens.append(f"time {info['time']}")
    if "nps" in info and info.get("nps"):
        tokens.append(f"nps {info['nps']}")
    if "score" in info and info["score"] is not None:
        tokens.append(f"score cp {info['score']}")
    if "currmove" in info and info.get("currmove"):
        tokens.append(f"currmove {info['currmove']}")
    if "currmovenumber" in info:
        tokens.append(f"currmovenumber {info['currmovenumber']}")
    if "pv" in info and info["pv"]:
        tokens.append("pv")
        tokens.extend(info["pv"])
    if "string" in info and info["string"]:
        tokens.append("string")
        tokens.append(info["string"])
    return " ".join(tokens)


class UCI:
    def __init__(self, board: chess.Board, ai: CPutterfish):
        self.board = board
        self.ai = ai
        self.options = {
            "Hash": 128,
            "Threads": 1,
            "Ponder": False,
            "Depth": 12,
            "MoveTime": 5000,
            "TimeLimit": 10,
            "Move Overhead": 100,
            "Model": "default",
            "Elo": 2500,
            "UCI_CPutterfish_Strength": 100,
            "UCI_Variant": "chess",
            "NNCacheSize": 131072,
            "Backend": "auto",
            "Device": "auto",
            "MultiPV": 1,
            "CPuct": 2,
            "VirtualLoss": 3,
            "EvalToWinProbScale": 50,
            "EvalToCP": 500,
            "VerboseInfo": False,
        }
        self._start_fen = START_FEN
        self._sync_model_path()
        self.ai.set_info_callback(lambda d: print(format_info(d)))
        self._search_result = None
        self._search_done = threading.Event()
        self._ponder_result = None
        self._ponder_done = threading.Event()
        self._ponder_thread = None
        self._search_thread = None
        self.commands()

    def _sync_model_path(self):
        if self.options.get("Model", "default") == "default":
            self.ai.model_path = "models/model.pth"
        else:
            self.ai.model_path = self.options["Model"]
        # LC0-style: sync NN cache size and backend to eval module
        try:
            from eval import set_nn_cache_size, set_backend
            set_nn_cache_size(int(self.options.get("NNCacheSize", 131072)))
            # Device "CPU only" / "GPU only" overrides Backend
            dev = str(self.options.get("Device", "auto"))
            if dev == "CPU only":
                set_backend("cpu")
            elif dev == "GPU only":
                set_backend("cuda")
            else:
                set_backend(str(self.options.get("Backend", "auto")))
        except Exception:
            pass

    def _print_options_uci(self):
        for opt in UCI_OPTIONS:
            name, typ, default, *rest = opt
            if typ == "spin":
                min_v, max_v = rest[:2]
                print(f"option name {name} type spin default {default} min {min_v} max {max_v}")
            elif typ == "check":
                val = "true" if default else "false"
                print(f"option name {name} type check default {val}")
            elif typ == "string":
                print(f"option name {name} type string default {default}")
            elif typ == "combo":
                vars_list = rest[2:] if len(rest) > 2 else []
                line = f"option name {name} type combo default {default}"
                for v in vars_list:
                    line += f" var {v}"
                print(line)

    def _normalized_options(self, opts: dict) -> dict:
        normalized = dict(opts)
        if "EvalToWinProbScale" in normalized:
            try:
                normalized["EvalToWinProbScale"] = float(normalized["EvalToWinProbScale"]) / 100.0
            except Exception:
                pass
        return normalized

    def _run_search(self, opts: dict):
        try:
            self._search_result = self.ai.select_move(opts)
        finally:
            self._search_done.set()

    def _run_ponder(self, board_after_ponder: chess.Board, opts: dict):
        try:
            self.ai.board = board_after_ponder
            self.ai.clear_stop()
            self._ponder_result = self.ai.select_move(opts)
        finally:
            self._ponder_done.set()

    def _start_ponder(self, our_move: chess.Move, ponder_move: chess.Move):
        """Start background ponder: position is after our_move and ponder_move (opponent's move)."""
        if self._ponder_thread and self._ponder_thread.is_alive():
            return
        self.ai.request_stop()
        self._ponder_done.clear()
        self._ponder_result = None
        b = self.board.copy()
        b.push(our_move)
        b.push(ponder_move)
        opts = self._normalized_options(self.options)
        opts["Ponder"] = True
        opts["TimeLimit"] = 3600
        opts["MoveTime"] = None
        opts["wtime"] = opts.get("wtime", 60000)
        opts["btime"] = opts.get("btime", 60000)
        self._ponder_thread = threading.Thread(
            target=self._run_ponder,
            args=(b, opts),
            daemon=True,
        )
        self._ponder_thread.start()

    def _stop_ponder(self):
        if self._ponder_thread and self._ponder_thread.is_alive():
            self.ai.request_stop()
            self._ponder_done.wait(timeout=5.0)
        self._ponder_thread = None

    def commands(self):
        while True:
            try:
                line = input("").strip()
            except EOFError:
                break
            if not line:
                continue
            if line == "uci":
                print("id name CPutterfish")
                print("id author Advik")
                self._print_options_uci()
                print("uciok")
            elif line == "isready":
                self.ai.request_stop()
                self._search_done.wait(timeout=5.0)
                self._stop_ponder()
                print("readyok")
            elif line.startswith("position "):
                self.ai.request_stop()
                self._search_done.wait(timeout=15.0)
                self._stop_ponder()
                variant = self.options.get("UCI_Variant", "chess")
                fen, move_strs = parse_position(line, variant=variant)
                self._start_fen = fen
                self.board.set_fen(fen)
                for m in move_strs:
                    try:
                        move = chess.Move.from_uci(m)
                        if move in self.board.legal_moves:
                            self.board.push(move)
                    except ValueError:
                        pass
            elif line.startswith("go "):
                self._stop_ponder()
                go_opts = parse_go_args(line)
                opts = self._normalized_options(self.options)
                opts.update(go_opts)
                self._search_done.clear()
                self._search_result = None
                self.ai.clear_stop()
                self.ai.board = self.board.copy()
                if opts.get("VerboseInfo"):
                    print(
                        "info string search start "
                        f"depth {opts.get('Depth', self.ai.max_rollout)} "
                        f"threads {opts.get('Threads', 1)} "
                        f"cpuct {opts.get('CPuct', 2)} "
                        f"vloss {opts.get('VirtualLoss', 3)}"
                    )

                def search_and_reply():
                    self._run_search(opts)
                    result = self._search_result
                    if result:
                        move = result.get("move")
                        ponder_move = result.get("ponder_move")
                        if move is not None:
                            if ponder_move and self.options.get("Ponder"):
                                print(f"bestmove {move.uci()} ponder {ponder_move.uci()}")
                                self._start_ponder(move, ponder_move)
                            else:
                                print(f"bestmove {move.uci()}")
                        else:
                            print("bestmove (none)")

                self._search_thread = threading.Thread(target=search_and_reply, daemon=True)
                self._search_thread.start()
            elif line == "stop":
                self.ai.request_stop()
                self._search_done.wait(timeout=30.0)
            elif line.startswith("setoption "):
                rest = line[9:].strip()
                if rest.startswith("name "):
                    rest = rest[5:]
                    idx = rest.find(" value ")
                    if idx >= 0:
                        name = rest[:idx].strip()
                        value = rest[idx + 7:].strip()
                        if name in self.options:
                            if isinstance(self.options[name], bool):
                                self.options[name] = value.lower() in ("true", "1", "yes")
                            elif isinstance(self.options[name], int):
                                try:
                                    self.options[name] = int(value)
                                except ValueError:
                                    pass
                            else:
                                self.options[name] = value
                            if name == "UCI_Variant":
                                self._start_fen = START_FEN_960 if value == "chess960" else START_FEN
                            self._sync_model_path()
            elif line == "ponderhit":
                self._stop_ponder()
                if self._ponder_result and self._ponder_result.get("move") is not None:
                    print(f"bestmove {self._ponder_result['move'].uci()}")
                else:
                    opts = self._normalized_options(self.options)
                    self._search_done.clear()
                    self._search_result = None
                    self.ai.clear_stop()
                    self.ai.board = self.board.copy()
                    self._run_search(opts)
                    result = self._search_result
                    if result and result.get("move") is not None:
                        print(f"bestmove {result['move'].uci()}")
                    else:
                        print("bestmove (none)")
            elif line == "ucinewgame":
                self._stop_ponder()
                self.board.set_fen(self._start_fen)
            elif line == "quit":
                self._stop_ponder()
                self.ai.request_stop()
                break


if __name__ == "__main__":
    board = chess.Board()
    ai = CPutterfish(board, depth=3, time_limit=10)
    UCI(board, ai)
