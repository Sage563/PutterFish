#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MATE_VALUE = 10.0


def fen_to_tensor(fen: str) -> torch.Tensor:
    import chess
    board = chess.Board(fen)
    tensor = torch.zeros((112, 8, 8), dtype=torch.float32)
    
    piece_to_idx = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            piece_idx = piece_to_idx[piece.piece_type]
            color_offset = 0 if piece.color == chess.WHITE else 6
            channel = piece_idx + color_offset
            tensor[channel, row, col] = 1.0
    
    return tensor


def eval_string_to_target(eval_str: str) -> float:
    if isinstance(eval_str, str):
        eval_str = eval_str.strip()
        if eval_str.startswith("M"):
            return MATE_VALUE
        elif eval_str.startswith("-M"):
            return -MATE_VALUE
        else:
            try:
                cp = float(eval_str)
                return cp / 500.0
            except ValueError:
                return 0.0
    elif isinstance(eval_str, (int, float)):
        return float(eval_str) / 500.0
    return 0.0


class ResidualBlock(nn.Module):
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class Cputterfish(nn.Module):
    
    def __init__(
        self,
        num_residual_blocks: int = 20,
        channels: int = 256,
        policy_channels: int = 73,
    ):
        super().__init__()
        
        self.num_residual_blocks = num_residual_blocks
        self.channels = channels
        self.policy_channels = policy_channels
        
        self.input_conv = nn.Conv2d(112, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels, kernel_size=3)
            for _ in range(num_residual_blocks)
        ])
        
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 8 * 8 * policy_channels)
        
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor):
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        for block in self.residual_blocks:
            x = block(x)
        
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = policy.view(-1, 8, 8, self.policy_channels)
        
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = self.value_fc2(value)
        
        return policy, value
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Cputterfish:
    model = Cputterfish(
        num_residual_blocks=20,
        channels=256,
        policy_channels=73,
    )
    model = model.to(device)
    return model


def save_model(model: Cputterfish, filepath: str) -> None:
    torch.save(model.state_dict(), filepath)


def load_model(model: Cputterfish, filepath: str, device: str = "cuda") -> Cputterfish:
    model.load_state_dict(torch.load(filepath, map_location=device))
    return model
