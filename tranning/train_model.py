#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import chess
from model import create_model as create_base_model, fen_to_tensor, eval_string_to_target, MATE_VALUE
import numpy as np
import os
import sqlite3
import zlib
from typing import Optional


def analyze_position(fen: str) -> dict:
    """Analyze chess position for loss weighting and feature importance."""
    try:
        board = chess.Board(fen)
        piece_count = sum(1 for _ in board.piece_map().values())
        
        # Position classification
        is_endgame = piece_count <= 7
        is_opening = board.fullmove_number <= 10
        is_tactical = len(list(board.legal_moves)) > 30
        
        # Material balance
        white_material = sum([len(board.pieces(pt, chess.WHITE)) * [1, 3, 3, 5, 9, 0][pt] 
                             for pt in range(6)])
        black_material = sum([len(board.pieces(pt, chess.BLACK)) * [1, 3, 3, 5, 9, 0][pt] 
                             for pt in range(6)])
        material_diff = abs(white_material - black_material)
        
        return {
            'is_endgame': is_endgame,
            'is_opening': is_opening,
            'is_tactical': is_tactical,
            'piece_count': piece_count,
            'material_diff': material_diff,
            'legal_moves': len(list(board.legal_moves))
        }
    except:
        return {'is_endgame': False, 'is_opening': False, 'is_tactical': False, 
                'piece_count': 16, 'material_diff': 0, 'legal_moves': 20}


def _build_cache(data, cache_path: str, cache_tensors: bool) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    conn = sqlite3.connect(cache_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fen_cache (fen TEXT PRIMARY KEY, pos_info TEXT, tensor BLOB)"
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    to_insert = []
    for idx, (fen, eval_val) in enumerate(data, start=1):
        pos_info = analyze_position(fen)
        tensor_blob = None
        if cache_tensors:
            tensor = fen_to_tensor(fen)
            arr = tensor.to(dtype=torch.uint8).numpy()
            tensor_blob = zlib.compress(arr.tobytes())
        to_insert.append((fen, json.dumps(pos_info), tensor_blob))
        if len(to_insert) >= 1000:
            conn.executemany(
                "INSERT OR REPLACE INTO fen_cache (fen, pos_info, tensor) VALUES (?, ?, ?)",
                to_insert,
            )
            conn.commit()
            to_insert = []
            if idx % 100000 == 0:
                print(f"  Cached {idx:,} positions...")
    if to_insert:
        conn.executemany(
            "INSERT OR REPLACE INTO fen_cache (fen, pos_info, tensor) VALUES (?, ?, ?)",
            to_insert,
        )
        conn.commit()
    conn.close()

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, data, cache_path: Optional[str] = None, cache_tensors: bool = False):
        self.data = data
        self.position_cache = {}
        self.cache_path = cache_path
        self.cache_tensors = cache_tensors
        self._cache_conn = None

    def _get_cache_conn(self):
        if self.cache_path is None:
            return None
        if self._cache_conn is None:
            self._cache_conn = sqlite3.connect(self.cache_path, check_same_thread=False)
        return self._cache_conn

    def _load_cached(self, fen):
        conn = self._get_cache_conn()
        if conn is None:
            return None, None
        cur = conn.execute("SELECT pos_info, tensor FROM fen_cache WHERE fen = ?", (fen,))
        row = cur.fetchone()
        if row is None:
            return None, None
        pos_info = json.loads(row[0])
        tensor = None
        if self.cache_tensors and row[1] is not None:
            raw = zlib.decompress(row[1])
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(112, 8, 8)
            tensor = torch.from_numpy(arr).float()
        return pos_info, tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, eval_val = self.data[idx]
        cached_pos_info, cached_tensor = self._load_cached(fen)
        if cached_tensor is not None:
            board_tensor = cached_tensor
        else:
            board_tensor = fen_to_tensor(fen)
        target_value = torch.tensor(eval_string_to_target(eval_val), dtype=torch.float32)
        
        # Cache position analysis for loss weighting
        if cached_pos_info is not None:
            pos_info = cached_pos_info
        else:
            if fen not in self.position_cache:
                self.position_cache[fen] = analyze_position(fen)
            pos_info = self.position_cache[fen]

        return board_tensor, target_value, pos_info


def train(
    dataset_path: str,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    model_path: str = None,
    output: str = "model.pth",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 4,
    use_amp: bool = False,
    cache_path: Optional[str] = None,
    cache_tensors: bool = False
) -> None:
    """
    Chess-optimized model training for Stockfish-like evaluation.
    - Deep RL with chess-specific loss weighting
    - Endgame vs Opening position handling
    - Tactical position focus
    - Material-aware evaluation
    - Centipawn accuracy optimization
    """
    print(f"Loading chess dataset from {dataset_path}...")
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()

    # Support both JSON array files and JSONL (one JSON object per line)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    for fen, eval_val in item.items():
                        data.append((fen, eval_val))
        elif isinstance(parsed, dict):
            for fen, eval_val in parsed.items():
                data.append((fen, eval_val))
    except (json.JSONDecodeError, ValueError):
        # Fallback to JSONL parsing
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data_dict = json.loads(line)
                if isinstance(data_dict, dict):
                    for fen, eval_val in data_dict.items():
                        data.append((fen, eval_val))
            except (json.JSONDecodeError, ValueError):
                continue
    
    print(f"Loaded {len(data)} chess positions from database")
    if len(data) == 0:
        raise ValueError(
            "Dataset is empty. Ensure the dataset file contains JSON array or JSONL "
            "entries with {fen: eval} objects."
        )
    
    if cache_path:
        if not os.path.exists(cache_path):
            print(f"Building cache at {cache_path} (this can be very large and slow)...")
            _build_cache(data, cache_path, cache_tensors)
        else:
            print(f"Using cache at {cache_path}")

    dataset = ChessDataset(data, cache_path=cache_path, cache_tensors=cache_tensors)
    
    # Optimized DataLoader with chess learningparameters
    if num_workers < 0:
        num_workers = 0
    # Windows multiprocessing requires top-level Dataset and picklable objects
    if num_workers <= 0:
        num_workers = 0

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    # Model and optimizer
    if model_path:
        model = create_base_model(device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = create_base_model(device=device)
    
    model.train()

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.startswith("cuda"))
    
    # Chess-optimized optimizers and schedulers
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing for stable convergence on chess positions
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-7
    )
    
    mse_loss = nn.MSELoss(reduction='none')
    mae_loss = nn.L1Loss(reduction='none')
    huber_loss = nn.HuberLoss(delta=0.5 * MATE_VALUE, reduction='none')
    
    best_accuracy = 0.0
    patience = 0
    gae_lambda = 0.95
    entropy_coeff = 0.005  # Lower entropy for sharper evaluations
    # Accuracy tolerance is in pawns (Stockfish eval scale)
    accuracy_tol = 0.25
    endgame_tol = 0.15
    tactical_tol = 0.30
    policy_temp = 15.0 / MATE_VALUE
    
    print(f"Starting chess training for {epochs} epochs")
    print(f"Device: {device}")
    print(f"Model: Stockfish-like evaluation optimization")
    print(f"Accuracy tolerance: {accuracy_tol:.2f} pawns")
    print(f"AMP enabled: {use_amp and device.startswith('cuda')}")
    print()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_accuracies = []
        epoch_mae_pawns = []
        endgame_accuracies = []
        tactical_accuracies = []
        
        for batch_idx, (board_tensor, target_value, pos_info) in enumerate(train_loader):
            board_tensor = board_tensor.to(device)
            target_value = target_value.to(device).unsqueeze(1)
            batch_size_actual = board_tensor.size(0)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.startswith("cuda")):
                policy, predicted_value = model(board_tensor)
            
            # CHESS-SPECIFIC LOSS WEIGHTING
            # Weight endgames higher (critical accuracy needed)
            if isinstance(pos_info, dict):
                endgame_vals = pos_info.get('is_endgame', [])
                tactical_vals = pos_info.get('is_tactical', [])
                material_vals = pos_info.get('material_diff', [])
            else:
                endgame_vals = [p['is_endgame'] for p in pos_info]
                tactical_vals = [p['is_tactical'] for p in pos_info]
                material_vals = [p['material_diff'] for p in pos_info]

            is_endgame = torch.as_tensor(endgame_vals, dtype=torch.float32, device=device).unsqueeze(1)
            is_tactical = torch.as_tensor(tactical_vals, dtype=torch.float32, device=device).unsqueeze(1)
            material_weight = torch.as_tensor(
                [1.0 + (v / 100.0) for v in material_vals],
                dtype=torch.float32,
                device=device
            ).unsqueeze(1)
            
            # Endgames need 2x weight (higher eval precision needed)
            endgame_weight = 1.0 + (is_endgame * 1.5)
            # Tactical positions need sharper evaluation
            tactical_weight = 1.0 + (is_tactical * 0.8)
            # Material imbalance = more critical evaluation
            combined_weight = endgame_weight * tactical_weight * material_weight
            
            # VALUE LOSS - Multiple loss functions for robustness
            target_value_loss = target_value.to(dtype=predicted_value.dtype)
            combined_weight_loss = combined_weight.to(dtype=predicted_value.dtype)
            error = torch.abs(predicted_value - target_value_loss)
            
            # MSE for overall accuracy
            mse = (mse_loss(predicted_value, target_value_loss) * combined_weight_loss).mean()
            
            # Huber loss for outlier resistance (chess has some wild evaluations)
            huber = (huber_loss(predicted_value, target_value_loss) * combined_weight_loss).mean()
            
            # MAE for centipawn accuracy (like Stockfish)
            mae = (mae_loss(predicted_value, target_value_loss) * combined_weight_loss).mean()
            
            value_loss = 0.50 * mse + 0.30 * huber + 0.20 * mae
            
            # TD LEARNING - Deep temporal difference for chess reasoning
            td_error = error * combined_weight_loss.squeeze()
            td_loss = td_error.mean() + (td_error ** 2).mean() * 0.15
            
            # ADVANTAGE ESTIMATION (GAE) - Chess-specific
            advantage = (target_value_loss - predicted_value).detach() * combined_weight_loss
            if advantage.std() > 1e-6:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            gae_advantage = advantage * gae_lambda
            
            # POLICY LOSS - Learning chess move ordering
            policy_flat = policy.view(batch_size_actual, -1)
            policy_logprobs = torch.nn.functional.log_softmax(policy_flat, dim=1)
            policy_probs = torch.nn.functional.softmax(policy_flat, dim=1)
            
            policy_entropy = -(policy_probs * policy_logprobs).sum(dim=1).mean()
            entropy_bonus = -entropy_coeff * policy_entropy
            
            # Policy should align with strong evaluations (Stockfish-like)
            target_prob = torch.softmax(target_value_loss.abs() * policy_temp, dim=0)
            target_prob = target_prob.to(dtype=policy_logprobs.dtype)
            policy_divergence = torch.nn.functional.kl_div(
                policy_logprobs,
                target_prob.expand_as(policy_logprobs),
                reduction='batchmean'
            )
            
            # ADVANTAGE ACTOR-CRITIC - Chess position understanding
            advantages_expanded = gae_advantage.expand_as(policy_flat) * policy_probs
            advantage_loss = -advantages_expanded.mean()
            
            # COMBINED CHESS LOSS (Stockfish-like evaluation)
            total_loss = (
                0.48 * value_loss +           # Primary: accurate chess evaluation
                0.22 * td_loss +              # Temporal difference for depth
                0.12 * advantage_loss +       # Actor-critic for position understanding
                0.12 * policy_divergence +    # Move ordering/policy
                0.04 * entropy_bonus +        # Prevent over-sharpening
                0.02 * mae                    # Centipawn accuracy
            )
            
            if use_amp and device.startswith("cuda"):
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # CHESS EVALUATION METRICS
            error = torch.abs(predicted_value.float() - target_value.float())
            accuracy = (error < accuracy_tol).float().mean().item()
            mae_pawns = error.mean().item()
            mae_cp = mae_pawns * 100.0
            
            # Endgame accuracy (more strict)
            endgame_mask = is_endgame.squeeze() > 0.5
            if endgame_mask.sum() > 0:
                endgame_acc = (error[endgame_mask] < endgame_tol).float().mean().item()
                endgame_accuracies.append(endgame_acc)
            
            # Tactical accuracy (needs sharpness)
            tactical_mask = is_tactical.squeeze() > 0.5
            if tactical_mask.sum() > 0:
                tactical_acc = (error[tactical_mask] < tactical_tol).float().mean().item()
                tactical_accuracies.append(tactical_acc)
            
            epoch_losses.append(total_loss.item())
            epoch_accuracies.append(accuracy)
            epoch_mae_pawns.append(mae_pawns)
            
            # Log only every 50 batches (reduce overhead)
            if (batch_idx + 1) % 50 == 0:
                recent_loss = np.mean(epoch_losses[-50:])
                recent_acc = np.mean(epoch_accuracies[-50:])
                print(
                    f"Epoch {epoch+1:3d} | Batch {batch_idx+1:4d}/{len(train_loader):4d} | "
                    f"Loss: {recent_loss:.6f} | Acc: {recent_acc*100:5.2f}% | "
                    f"MAE: {mae_pawns:.3f} pawns ({mae_cp:.0f} cp)"
                )
        
        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies)
        avg_mae_pawns = np.mean(epoch_mae_pawns) if epoch_mae_pawns else 0.0
        avg_endgame = np.mean(endgame_accuracies) if endgame_accuracies else 0.0
        avg_tactical = np.mean(tactical_accuracies) if tactical_accuracies else 0.0
        
        scheduler.step()
        
        print(
            f"Epoch {epoch+1:3d} | Loss: {avg_loss:.6f} | "
            f"Acc: {avg_accuracy*100:.2f}% | EndGame: {avg_endgame*100:.2f}% | "
            f"Tactical: {avg_tactical*100:.2f}% | "
            f"MAE: {avg_mae_pawns:.3f} pawns ({avg_mae_pawns*100:.0f} cp) | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Model checkpoint
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            torch.save(model.state_dict(), output)
            print(f"  ✓ Best model saved (Accuracy: {avg_accuracy*100:.2f}%)")
            patience = 0
        else:
            patience += 1
        
        # Chess-specific early stopping
        if avg_accuracy > 0.94 and avg_endgame > 0.88:
            print(f"✓ Strong chess evaluation achieved! Accuracy: {avg_accuracy*100:.2f}% | Endgame: {avg_endgame*100:.2f}%")
            break
        
        if patience >= 25:
            print(f"Early stopping after {epoch+1} epochs")
            break
        
        print()
    
    print(f"Chess training complete! Best accuracy: {best_accuracy*100:.2f}%")
    print(f"Stockfish-like model saved to {output}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chess AI training - Stockfish-like evaluation")
    parser.add_argument("--dataset", "-d", default="dataset.json", help="Chess position database")
    parser.add_argument("--model", "-m", default=None, help="Pre-trained model for fine-tuning")
    parser.add_argument("--output", "-o", default="chess_model.pth", help="Output model path")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--device", "-dev", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--num-workers", "-nw", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only)")
    parser.add_argument("--cache", default=None, help="Optional SQLite cache path for pos_info/tensors")
    parser.add_argument("--cache-tensors", action="store_true", help="Also cache FEN tensors (very large)")
    
    args = parser.parse_args()
    
    device = args.device
    print(f"Chess Training System | Device: {device}")
    print("=" * 60)
    
    train(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_path=args.model,
        output=args.output,
        device=device,
        num_workers=args.num_workers,
        use_amp=args.amp,
        cache_path=args.cache,
        cache_tensors=args.cache_tensors
    )


