#!/usr/bin/env python3

import torch
from model import Cputterfish, create_model as create_base_model, fen_to_tensor
import argparse

MATE_VALUE = 10.0

def load_model(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Cputterfish:
    model = create_base_model(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def interpret_eval(eval_score: float) -> str:
    if eval_score >= MATE_VALUE * 0.9:
        return "M1+ (White winning)"
    elif eval_score <= -MATE_VALUE * 0.9:
        return "M1- (Black winning)"
    elif eval_score > 0.75:
        return "Winning (White)"
    elif eval_score > 0.6:
        return "Much better (White)"
    elif eval_score > 0.55:
        return "Better (White)"
    elif eval_score > -0.55 and eval_score <= 0.55:
        return "Roughly equal"
    elif eval_score > -0.6:
        return "Better (Black)"
    elif eval_score > -0.75:
        return "Much better (Black)"
    elif eval_score > -0.9:
        return "Winning (Black)"
    else:
        return "M1- (Black winning)"


def evaluate_position(fen: str, model: Cputterfish, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    model.eval()
    board_tensor = fen_to_tensor(fen).unsqueeze(0).to(device)
    
    with torch.no_grad():
        policy, value = model(board_tensor)
    
    eval_score = float(value.item())
    
    return {
        "fen": fen,
        "eval_score": eval_score,
        "interpretation": interpret_eval(eval_score)
    }


def eval_single_position(fen: str, model_path: str = "chess_model_rl.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device=device)
    
    print(f"Evaluating position: {fen}")
    result = evaluate_position(fen, model, device=device)
    
    print(f"\nResult:")
    print(f"  FEN: {result['fen']}")
    print(f"  Eval Score: {result['eval_score']:.4f}")
    print(f"  Interpretation: {result['interpretation']}")
    
    return result


def eval_multiple_positions(positions: list, model_path: str = "chess_model_rl.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device=device)
    
    print(f"\nEvaluating {len(positions)} position(s)...\n")
    print(f"{'#':<4} {'Eval Score':<12} {'Interpretation':<30} {'FEN':<60}")
    print("-" * 110)
    
    for idx, fen in enumerate(positions, 1):
        result = evaluate_position(fen, model, device=device)
        print(
            f"{idx:<4} {result['eval_score']:<12.4f} {result['interpretation']:<30} "
            f"{result['fen']:<60}"
        )
    
    return [evaluate_position(fen, model, device=device) for fen in positions]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate chess positions with trained model")
    parser.add_argument("--model", "-m", default="chess_model_rl.pth", help="Path to trained model")
    parser.add_argument("--fen", "-f", default=None, help="FEN position to evaluate")
    parser.add_argument("--test", "-t", action="store_true", help="Run test with sample positions")
    
    args = parser.parse_args()
    
    if args.test:
        test_positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "8/8/8/4k3/8/8/8/4K3 w - - 0 1",
            "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq e6 0 1",
        ]
        
        print("=" * 110)
        print("Chess AI Position Evaluator - Test Mode")
        print("=" * 110)
        eval_multiple_positions(test_positions, model_path=args.model)
        
    elif args.fen:
        print("=" * 110)
        print("Chess AI Position Evaluator")
        print("=" * 110)
        eval_single_position(args.fen, model_path=args.model)
        
    else:
        print("Usage:")
        print("  Single position:")
        print('    python evaluate.py -f "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"')
        print("\n  Test with sample positions:")
        print("    python evaluate.py -t")
        print("\n  Use custom model:")
        print('    python evaluate.py -m my_model.pth -f "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"')

