#!/usr/bin/env python3

import os
import sys
import threading
import torch
from model import Cputterfish, create_model, fen_to_tensor

_device_override = None  # "cpu", "cuda", or None for auto
_EVAL_CACHE_MAX = 131072  # default; overridden by set_nn_cache_size
_eval_cache = {}
_eval_cache_lock = threading.Lock()


def set_backend(name: str):
    """Set NN backend: 'auto', 'cpu', or 'cuda'. Clears model cache so next load uses new device."""
    global _device_override, _model_cache
    old = _device_override
    if name == "cuda":
        _device_override = "cuda"
    elif name == "cpu":
        _device_override = "cpu"
    else:
        _device_override = None  # auto
    if _device_override != old:
        _model_cache.clear()


def set_nn_cache_size(size: int):
    """Set max number of positions in NN eval cache (NNCacheSize)."""
    global _EVAL_CACHE_MAX
    _EVAL_CACHE_MAX = max(0, min(999999999, size))


def _device():
    if _device_override:
        return _device_override
    return "cuda" if torch.cuda.is_available() else "cpu"


device = "cuda" if torch.cuda.is_available() else "cpu"
_model_cache = {}  # model_path -> loaded model


def _resource_path(relative_path: str) -> str:
    """Resolve path for bundled PyInstaller exe (frozen) or normal run."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


def _get_model(model_path: str):
    """Load and cache model by path."""
    resolved = _resource_path(model_path)
    dev = _device()
    if resolved not in _model_cache:
        m = create_model(device=dev)
        m.load_state_dict(torch.load(resolved, map_location=dev))
        m.eval()
        _model_cache[resolved] = m
    return _model_cache[resolved]


def predict_eval(model_path: str = "models/model.pth", fen: str = "") -> float:
    """Predict evaluation for a FEN position. Returns raw float. Uses cache for speed."""
    if not fen:
        return 0.0
    dev = _device()
    key = (model_path, fen)
    with _eval_cache_lock:
        if key in _eval_cache:
            return _eval_cache[key]
    m = _get_model(model_path)
    board_tensor = fen_to_tensor(fen).unsqueeze(0).to(dev)
    with torch.inference_mode():
        _, value = m(board_tensor)
    v = float(value.item())
    with _eval_cache_lock:
        if _EVAL_CACHE_MAX > 0 and len(_eval_cache) >= _EVAL_CACHE_MAX:
            for _ in range(_EVAL_CACHE_MAX // 2):
                _eval_cache.pop(next(iter(_eval_cache)), None)
        if _EVAL_CACHE_MAX > 0:
            _eval_cache[key] = v
    return v
