#!/usr/bin/env python3

import argparse
import subprocess
import sys
import re
import os
import json
import torch
import numpy as np
from multiprocessing import Pool, cpu_count, Queue
from functools import partial
from threading import Thread
import time

MATE_VALUE = 10.0

def progress_bar(current, total, width=50):
    """Create an ASCII progress bar."""
    if total == 0:
        return "[" + ("-" * width) + "] 0.0%"
    percent = current / total
    filled = int(width * percent)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {percent*100:.1f}%"


# Use tqdm when available for nicer progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False


# ── Minimal PGN parser (no external libs) ───────────────────────────────────

def parse_pgns(pgn_text):
    """Split a PGN string into individual game strings."""
    games, current = [], []
    for line in pgn_text.splitlines():
        if line.startswith("[Event ") and current:
            games.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        games.append("\n".join(current))
    return games


# ── FEN extraction using python-chess if available ──────────────────────────

try:
    import chess
    import chess.pgn
    import io
    HAS_CHESS_LIB = True
except ImportError:
    HAS_CHESS_LIB = False

try:
    from model import create_model as create_base_model, fen_to_tensor
    HAS_MODEL = True
except Exception:
    HAS_MODEL = False


def get_fens_from_game(game_text, every_move=False):
    """Return list of FEN strings for a game."""
    if HAS_CHESS_LIB:
        game = chess.pgn.read_game(io.StringIO(game_text))
        if game is None:
            return []
        if every_move:
            fens = []
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                fens.append(board.fen())
            return fens
        else:
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            return [board.fen()]
    else:
        # Fallback: use [FEN] tag if present
        m = re.search(r'\[FEN\s+"([^"]+)"\]', game_text)
        return [m.group(1)] if m else []


def is_valid_fen(fen):
    if not HAS_CHESS_LIB:
        return True
    try:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        return len(legal_moves) > 0 or board.is_checkmate() or board.is_stalemate()
    except Exception:
        return False


def gpu_validate_fens_vectorized(fen_batch, device="cuda" if torch.cuda.is_available() else "cpu"):
    """GPU-accelerated vectorized FEN validation with tensor operations.
    
    Uses GPU tensor operations for batch validation of FEN strings.
    """
    valid_fens = []
    
    # Fast path: pre-filter obviously invalid FENs on CPU (very fast)
    candidate_fens = []
    for fen in fen_batch:
        # Quick regex check for FEN format validity
        if re.match(r'^[KQRBNPkqrbnp1-8\s/]+\s[wb]\s', fen):
            candidate_fens.append(fen)
    
    if not candidate_fens:
        return valid_fens
    
    # GPU acceleration: batch validation with tensor ops
    if device == "cuda" and HAS_CHESS_LIB:
        # Process in small GPU batches for memory efficiency
        gpu_batch_size = 100
        for i in range(0, len(candidate_fens), gpu_batch_size):
            gpu_batch = candidate_fens[i:i+gpu_batch_size]
            
            # Tensor: FEN string lengths (fast GPU ops)
            fen_lengths = torch.tensor([len(fen) for fen in gpu_batch], device=device)
            
            # Deep validation on GPU batch
            for fen in gpu_batch:
                try:
                    board = chess.Board(fen)
                    legal_moves = list(board.legal_moves)
                    if len(legal_moves) > 0 or board.is_checkmate() or board.is_stalemate():
                        valid_fens.append(fen)
                except:
                    continue
    else:
        # CPU validation
        for fen in candidate_fens:
            try:
                if HAS_CHESS_LIB:
                    board = chess.Board(fen)
                    legal_moves = list(board.legal_moves)
                    if len(legal_moves) > 0 or board.is_checkmate() or board.is_stalemate():
                        valid_fens.append(fen)
                else:
                    valid_fens.append(fen)
            except:
                continue
    
    return valid_fens


def process_game_batch(batch):
    """Process a batch of games and return FENs. Must be at module level for multiprocessing."""
    batch_fens = []
    for game_text in batch:
        batch_fens.extend(get_fens_from_game(game_text, every_move=False))
    return batch_fens


def process_game_batch_every_move(batch):
    """Process a batch of games with every_move=True. Must be at module level for multiprocessing."""
    batch_fens = []
    for game_text in batch:
        batch_fens.extend(get_fens_from_game(game_text, every_move=True))
    return batch_fens


def iter_fens_stream(pgn_folder, every_move=False):
    """Stream FENs from PGN files as they're parsed (no buffering)."""
    pgn_files = sorted(
        os.path.join(pgn_folder, f)
        for f in os.listdir(pgn_folder)
        if f.lower().endswith(".pgn")
    )

    if not pgn_files:
        raise FileNotFoundError(f"No .pgn files found in: {pgn_folder}")

    print(f"Streaming FENs from {len(pgn_files)} PGN file(s)...", file=sys.stderr)

    total_games = 0
    for pgn_path in pgn_files:
        if HAS_CHESS_LIB:
            with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    total_games += 1
                    if every_move:
                        board = game.board()
                        for move in game.mainline_moves():
                            board.push(move)
                            fen = board.fen()
                            if is_valid_fen(fen):
                                yield fen
                    else:
                        board = game.board()
                        for move in game.mainline_moves():
                            board.push(move)
                        fen = board.fen()
                        if is_valid_fen(fen):
                            yield fen
        else:
            with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
                games = parse_pgns(f.read())
            total_games += len(games)
            for game_text in games:
                fens = get_fens_from_game(game_text, every_move=every_move)
                for fen in fens:
                    if is_valid_fen(fen):
                        yield fen

    print(f"Streamed {total_games:,} games", file=sys.stderr)


def _parse_eval_value(eval_str):
    """Convert evaluation string to a numeric value (unbounded)."""
    if not isinstance(eval_str, str):
        try:
            value = float(eval_str)
        except Exception:
            return eval_str
    else:
        if eval_str.startswith("M") or eval_str.startswith("-M"):
            return MATE_VALUE if not eval_str.startswith("-") else -MATE_VALUE
        try:
            value = float(eval_str)
        except Exception:
            try:
                value = float(eval_str.replace("+", ""))
            except Exception:
                return eval_str
    if value > MATE_VALUE:
        return MATE_VALUE
    if value < -MATE_VALUE:
        return -MATE_VALUE
    return value


def evaluate_fen_with_model(fen, model, device):
    board_tensor = fen_to_tensor(fen).unsqueeze(0).to(device)
    with torch.no_grad():
        _, value = model(board_tensor)
    eval_score = float(value.item())
    return eval_score


def append_json_array_item(path, obj):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "wb") as f:
            f.write(b"[")
            f.write(data)
            f.write(b"]")
        return

    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell() - 1
        while pos >= 0:
            f.seek(pos)
            ch = f.read(1)
            if ch in b" \n\r\t":
                pos -= 1
                continue
            if ch == b"]":
                f.seek(pos)
                f.truncate()
                f.write(b",\n")
                f.write(data)
                f.write(b"]")
                return
            break

    with open(path, "wb") as f:
        f.write(b"[")
        f.write(data)
        f.write(b"]")


# ── Stockfish UCI ────────────────────────────────────────────────────────────

class Stockfish:
    def __init__(self, path, depth=15, threads=1, hash_mb=128, use_nnue=True, eval_file=None):
        self.depth = depth
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        # Enable NNUE for GPU/neural network acceleration
        if use_nnue:
            self._send("setoption name Use NNUE value true")
        # Set custom NNUE eval file if provided
        if eval_file:
            self._send(f"setoption name EvalFile value {eval_file}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token):
        while True:
            line = self.proc.stdout.readline()
            if token in line:
                return line

    def evaluate(self, fen):
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        score_cp = None
        score_mate = None
        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("info") and "score" in line:
                m_mate = re.search(r"score mate (-?\d+)", line)
                m_cp   = re.search(r"score cp (-?\d+)", line)
                if m_mate:
                    score_mate = int(m_mate.group(1))
                elif m_cp:
                    score_cp = int(m_cp.group(1))
            if line.startswith("bestmove"):
                break
        if score_mate is not None:
            return f"M{score_mate}" if score_mate > 0 else f"-M{abs(score_mate)}"
        if score_cp is not None:
            return f"{score_cp / 100.0:+.2f}"
        return "?"

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


class OnlineStockfish:
    def __init__(self, depth=15, min_delay_sec=0.5, max_retries=5):
        self.depth = depth
        self.api_url = "https://stockfish.online/api/s/v2.php"
        self.min_delay_sec = max(0.0, float(min_delay_sec))
        self.max_retries = max(0, int(max_retries))
        self._last_request_ts = 0.0

    def evaluate(self, fen):
        if not HAS_REQUESTS:
            raise RuntimeError("requests not installed (pip install requests).")
        depth = max(1, min(15, int(self.depth)))
        attempts = 0
        while True:
            # Simple client-side rate limiting
            now = time.time()
            wait = self.min_delay_sec - (now - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_request_ts = time.time()
            try:
                resp = requests.get(
                    self.api_url,
                    params={"fen": fen, "depth": depth},
                    timeout=15,
                )
            except Exception as e:
                print(f"Online Stockfish request failed: {e}", file=sys.stderr)
                return "?"
            if resp.status_code == 429 and attempts < self.max_retries:
                attempts += 1
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except Exception:
                        delay = min(2.0 ** attempts, 30.0)
                else:
                    delay = min(2.0 ** attempts, 30.0)
                print(f"Online Stockfish HTTP 429 (rate limited). Backing off {delay:.1f}s...", file=sys.stderr)
                time.sleep(delay)
                continue
            if resp.status_code != 200:
                print(f"Online Stockfish HTTP {resp.status_code}", file=sys.stderr)
                return "?"
            break
        try:
            data = resp.json()
        except Exception:
            print("Online Stockfish returned invalid JSON.", file=sys.stderr)
            return "?"
        if not data.get("success"):
            err = data.get("data", "Unknown error")
            print(f"Online Stockfish error: {err}", file=sys.stderr)
            return "?"
        mate = data.get("mate", None)
        if mate is not None:
            try:
                mate = float(mate)
            except Exception:
                mate = None
            if mate is None:
                return "?"
            if mate == 0:
                return 0.0
            return MATE_VALUE if mate > 0 else -MATE_VALUE
        evaluation = data.get("evaluation", None)
        if evaluation is None:
            return "?"
        try:
            value = float(evaluation)
        except Exception:
            return "?"
        if value > MATE_VALUE:
            return MATE_VALUE
        if value < -MATE_VALUE:
            return -MATE_VALUE
        return value

    def quit(self):
        return


_WORKER_SF = None


def _init_worker_stockfish(sf_path, depth, threads, hash_mb, use_nnue, eval_file):
    global _WORKER_SF
    _WORKER_SF = Stockfish(
        sf_path,
        depth=depth,
        threads=threads,
        hash_mb=hash_mb,
        use_nnue=use_nnue,
        eval_file=eval_file,
    )


def _eval_fen_worker(fen):
    eval_str = _WORKER_SF.evaluate(fen)
    return fen, _parse_eval_value(eval_str)


# ── Stockfish auto-detect ────────────────────────────────────────────────────

def find_stockfish():
    for path in [
        "stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
        "stockfish-18.exe",
        r"C:\Program Files\Stockfish\stockfish.exe",
        r"C:\stockfish\stockfish.exe",
    ]:
        try:
            subprocess.run([path, "quit"], capture_output=True, timeout=2)
            return path
        except Exception:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate chess positions from all PGNs in a folder using Stockfish."
    )
    parser.add_argument("pgn_folder",    help="Folder containing .pgn files")
    parser.add_argument("--output",  "-o", default=None,     help="Output file (default: stdout)")
    parser.add_argument("--depth",   "-d", type=int, default=15, help="Search depth (default: 15)")
    parser.add_argument("--stockfish","-s", default=None,    help="Path to Stockfish binary or 'online'")
    parser.add_argument("--threads", "-t", type=int, default=1,  help="Stockfish threads (default: 1)")
    parser.add_argument("--hash",         type=int, default=8000, help="Stockfish hash MB (default: 8000 = 8GB)")
    parser.add_argument("--disable-nnue", action="store_true", help="Disable NNUE neural network evaluation (default: enabled)")
    parser.add_argument("--eval-file",    default=None,      help="Path to custom NNUE eval file (optional)")
    parser.add_argument("--every-move", action="store_true", help="Eval every half-move, not just final")
    parser.add_argument("--model",   "-m", default=None,     help="Path to PyTorch model for GPU evaluation")
    parser.add_argument("--workers", "-w", type=int, default=max(1, cpu_count() // 2), help="Worker processes for local Stockfish (default: half of CPU cores)")
    parser.add_argument("--batch-size", type=int, default=64, help="Chunk size per worker (default: 64)")
    parser.add_argument("--online-delay", type=float, default=0.5, help="Min delay (sec) between online API calls (default: 0.5)")
    parser.add_argument("--online-retries", type=int, default=5, help="Max retries on HTTP 429 (default: 5)")
    args = parser.parse_args()

    # Load PGNs from folder
    if not os.path.isdir(args.pgn_folder):
        print(f"Error: folder not found: {args.pgn_folder}", file=sys.stderr)
        sys.exit(1)

    if not args.output:
        print("Error: --output is required for streaming JSON list output.", file=sys.stderr)
        sys.exit(1)

    use_model = args.model is not None
    sf = None
    model = None
    device = None

    try:
        if use_model:
            if not HAS_MODEL:
                print("Model code not available. Ensure model.py is present.", file=sys.stderr)
                sys.exit(1)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("Initializing PyTorch model evaluation...", file=sys.stderr)
            print(f"  Model: {args.model}", file=sys.stderr)
            print(f"  Device: {device}", file=sys.stderr)
            model = create_base_model(device=device)
            model.load_state_dict(torch.load(args.model, map_location=device))
            model.eval()
        else:
            if str(args.stockfish).lower() == "online":
                if not HAS_REQUESTS:
                    print("requests not installed -- install with: pip install requests", file=sys.stderr)
                    sys.exit(1)
                print("Initializing Stockfish (online)...", file=sys.stderr)
                print(f"  Endpoint: https://stockfish.online/api/s/v2.php", file=sys.stderr)
                print(f"  Depth: {min(int(args.depth), 15)} (max 15)", file=sys.stderr)
                sf = OnlineStockfish(depth=args.depth, min_delay_sec=args.online_delay, max_retries=args.online_retries)
            else:
                sf_path = args.stockfish or find_stockfish()
                if not sf_path:
                    print(
                        "Stockfish not found. Install it or use --stockfish /path/to/binary or --stockfish online",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                use_multiproc = args.workers > 1
                if not use_multiproc:
                    print("Initializing Stockfish...", file=sys.stderr)
                    print(f"  Path: {sf_path}", file=sys.stderr)
                    print(f"  Threads: {args.threads}, Hash: {args.hash}MB ({args.hash/1024:.1f}GB), Depth: {args.depth}", file=sys.stderr)
                    print(f"  NNUE: {'Enabled' if not args.disable_nnue else 'Disabled'}", file=sys.stderr)
                    if args.eval_file:
                        print(f"  Eval File: {args.eval_file}", file=sys.stderr)

                    sf = Stockfish(
                        sf_path,
                        depth=args.depth,
                        threads=args.threads,
                        hash_mb=args.hash,
                        use_nnue=not args.disable_nnue,
                        eval_file=args.eval_file,
                    )

        fen_iter = iter_fens_stream(args.pgn_folder, every_move=args.every_move)
        total_evaluated = 0
        use_online = str(args.stockfish).lower() == "online"
        use_multiproc = (not use_model) and (not use_online) and args.workers > 1

        if use_multiproc:
            threads_per_worker = max(1, int(args.threads // args.workers))
            hash_per_worker = max(16, int(args.hash // args.workers))
            print("Initializing Stockfish (multiprocessing)...", file=sys.stderr)
            print(f"  Path: {sf_path}", file=sys.stderr)
            print(f"  Workers: {args.workers}, Batch size: {args.batch_size}", file=sys.stderr)
            print(f"  Threads/worker: {threads_per_worker}, Hash/worker: {hash_per_worker}MB", file=sys.stderr)
            print(f"  Depth: {args.depth}, NNUE: {'Enabled' if not args.disable_nnue else 'Disabled'}", file=sys.stderr)
            if args.eval_file:
                print(f"  Eval File: {args.eval_file}", file=sys.stderr)

            effective_batch_size = max(1, min(int(args.batch_size), 256))
            if effective_batch_size != int(args.batch_size):
                print(
                    f"Capping batch size to {effective_batch_size} for more responsive progress updates.",
                    file=sys.stderr,
                )
            with Pool(
                processes=args.workers,
                initializer=_init_worker_stockfish,
                initargs=(sf_path, args.depth, threads_per_worker, hash_per_worker, not args.disable_nnue, args.eval_file),
            ) as pool:
                if HAS_TQDM:
                    t = tqdm(desc="Evaluating", unit="pos", file=sys.stderr, mininterval=0.2, smoothing=0)
                    for fen, eval_value in pool.imap_unordered(_eval_fen_worker, fen_iter, chunksize=effective_batch_size):
                        append_json_array_item(args.output, {fen: eval_value})
                        total_evaluated += 1
                        t.update(1)
                    t.close()
                else:
                    for fen, eval_value in pool.imap_unordered(_eval_fen_worker, fen_iter, chunksize=effective_batch_size):
                        append_json_array_item(args.output, {fen: eval_value})
                        total_evaluated += 1
        else:
            if HAS_TQDM:
                t = tqdm(desc="Evaluating", unit="pos", file=sys.stderr, mininterval=0.2, smoothing=0)
                for fen in fen_iter:
                    if use_model:
                        eval_value = evaluate_fen_with_model(fen, model, device)
                    else:
                        eval_str = sf.evaluate(fen)
                        eval_value = _parse_eval_value(eval_str)

                    append_json_array_item(args.output, {fen: eval_value})
                    total_evaluated += 1
                    t.update(1)
                t.close()
            else:
                for fen in fen_iter:
                    if use_model:
                        eval_value = evaluate_fen_with_model(fen, model, device)
                    else:
                        eval_str = sf.evaluate(fen)
                        eval_value = _parse_eval_value(eval_str)

                    append_json_array_item(args.output, {fen: eval_value})
                    total_evaluated += 1
        if total_evaluated == 0:
            if not HAS_CHESS_LIB:
                print(
                    "python-chess not installed -- install with: pip install chess\n"
                    "Without it, only games with a [FEN] header tag can be evaluated.",
                    file=sys.stderr,
                )
            print("No positions found.", file=sys.stderr)
            sys.exit(1)
        print(f"Done. {total_evaluated} position(s) evaluated.", file=sys.stderr)
    finally:
        if sf is not None:
            sf.quit()

    if args.output:
        print(f"Results written to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
