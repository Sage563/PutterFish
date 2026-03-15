#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import chess
from model import create_model as create_base_model, fen_to_tensor, eval_string_to_target, MATE_VALUE
import numpy as np


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


def train(
    dataset_path: str,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    model_path: str = None,
    output: str = "model.pth",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
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
        for line in f:
            try:
                data_dict = json.loads(line.strip())
                for fen, eval_val in data_dict.items():
                    data.append((fen, eval_val))
            except (json.JSONDecodeError, ValueError):
                continue
    
    print(f"Loaded {len(data)} chess positions from database")
    
    # Create chess-optimized dataset
    class ChessDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
            self.position_cache = {}
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            fen, eval_val = self.data[idx]
            board_tensor = fen_to_tensor(fen)
            target_value = torch.tensor(eval_string_to_target(eval_val), dtype=torch.float32)
            
            # Cache position analysis for loss weighting
            if fen not in self.position_cache:
                self.position_cache[fen] = analyze_position(fen)
            
            return board_tensor, target_value, self.position_cache[fen]
    
    dataset = ChessDataset(data)
    
    # Optimized DataLoader with chess learningparameters
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Model and optimizer
    if model_path:
        model = create_base_model(device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = create_base_model(device=device)
    
    model.train()
    
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
    accuracy_tol = 0.10 * MATE_VALUE
    endgame_tol = 0.08 * MATE_VALUE
    tactical_tol = 0.12 * MATE_VALUE
    policy_temp = 15.0 / MATE_VALUE
    
    print(f"Starting chess training for {epochs} epochs")
    print(f"Device: {device}")
    print(f"Model: Stockfish-like evaluation optimization")
    print()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_accuracies = []
        endgame_accuracies = []
        tactical_accuracies = []
        
        for batch_idx, (board_tensor, target_value, pos_info) in enumerate(train_loader):
            board_tensor = board_tensor.to(device)
            target_value = target_value.to(device).unsqueeze(1)
            batch_size_actual = board_tensor.size(0)
            
            optimizer.zero_grad(set_to_none=True)
            policy, predicted_value = model(board_tensor)
            
            # CHESS-SPECIFIC LOSS WEIGHTING
            # Weight endgames higher (critical accuracy needed)
            is_endgame = torch.tensor([p['is_endgame'] for p in pos_info], 
                                     dtype=torch.float32, device=device).unsqueeze(1)
            is_tactical = torch.tensor([p['is_tactical'] for p in pos_info], 
                                      dtype=torch.float32, device=device).unsqueeze(1)
            material_weight = torch.tensor([1.0 + (p['material_diff'] / 100.0) for p in pos_info],
                                          dtype=torch.float32, device=device).unsqueeze(1)
            
            # Endgames need 2x weight (higher eval precision needed)
            endgame_weight = 1.0 + (is_endgame * 1.5)
            # Tactical positions need sharper evaluation
            tactical_weight = 1.0 + (is_tactical * 0.8)
            # Material imbalance = more critical evaluation
            combined_weight = endgame_weight * tactical_weight * material_weight
            
            # VALUE LOSS - Multiple loss functions for robustness
            error = torch.abs(predicted_value - target_value)
            
            # MSE for overall accuracy
            mse = (mse_loss(predicted_value, target_value) * combined_weight).mean()
            
            # Huber loss for outlier resistance (chess has some wild evaluations)
            huber = (huber_loss(predicted_value, target_value) * combined_weight).mean()
            
            # MAE for centipawn accuracy (like Stockfish)
            mae = (mae_loss(predicted_value, target_value) * combined_weight).mean()
            
            value_loss = 0.50 * mse + 0.30 * huber + 0.20 * mae
            
            # TD LEARNING - Deep temporal difference for chess reasoning
            td_error = error * combined_weight.squeeze()
            td_loss = td_error.mean() + (td_error ** 2).mean() * 0.15
            
            # ADVANTAGE ESTIMATION (GAE) - Chess-specific
            advantage = (target_value - predicted_value).detach() * combined_weight
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
            target_prob = torch.softmax(target_value.abs() * policy_temp, dim=0)
            policy_divergence = torch.nn.functional.kl_div(
                policy_logprobs,
                target_prob.expand_as(policy_logprobs),
                reduction='mean'
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
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # CHESS EVALUATION METRICS
            error = torch.abs(predicted_value - target_value)
            accuracy = (error < accuracy_tol).float().mean().item()
            
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
            
            # Log only every 50 batches (reduce overhead)
            if (batch_idx + 1) % 50 == 0:
                recent_loss = np.mean(epoch_losses[-50:])
                recent_acc = np.mean(epoch_accuracies[-50:])
                print(
                    f"Epoch {epoch+1:3d} | Batch {batch_idx+1:4d}/{len(train_loader):4d} | "
                    f"Loss: {recent_loss:.6f} | Acc: {recent_acc*100:5.2f}%"
                )
        
        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies)
        avg_endgame = np.mean(endgame_accuracies) if endgame_accuracies else 0.0
        avg_tactical = np.mean(tactical_accuracies) if tactical_accuracies else 0.0
        
        scheduler.step()
        
        print(
            f"Epoch {epoch+1:3d} | Loss: {avg_loss:.6f} | "
            f"Acc: {avg_accuracy*100:.2f}% | EndGame: {avg_endgame*100:.2f}% | "
            f"Tactical: {avg_tactical*100:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}"
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
    #defualt command python train_model.py --dataset dataset.json --model chess_model.pth --output chess_model.pth --epochs 20 --batch-size 64 --learning-rate 0.0001 --device cuda  
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
        device=device
    )


