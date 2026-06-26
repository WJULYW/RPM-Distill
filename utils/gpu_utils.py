"""GPU utilities with safe CPU fallback."""

from typing import Optional, Union

import torch

try:
    import pynvml
except Exception:
    pynvml = None


def _parse_gpu_index(gpu_index: Optional[Union[str, int]]) -> Optional[int]:
    if gpu_index is None:
        return None
    if isinstance(gpu_index, int):
        return gpu_index

    text = str(gpu_index).strip().lower()
    if text in {"", "auto", "none", "cpu", "cuda"}:
        return None
    if text.startswith("cuda:"):
        text = text.split(":", 1)[1]
    try:
        return int(text)
    except ValueError:
        return None


def auto_select_gpu(min_free_memory_gb: float = 20, preferred_gpu: Optional[int] = None) -> Optional[str]:
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        return None

    if pynvml is None:
        print("pynvml is not available, fallback to cuda:0.")
        return "0"

    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        if gpu_count <= 0:
            return None

        if preferred_gpu is not None and 0 <= preferred_gpu < gpu_count:
            return str(preferred_gpu)

        best_gpu = 0
        best_free_gb = -1.0
        for gpu_id in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = meminfo.free / 1024 / 1024 / 1024
            if free_gb > best_free_gb:
                best_free_gb = free_gb
                best_gpu = gpu_id

        if best_free_gb < min_free_memory_gb:
            print(
                f"[WARN] No GPU has >= {min_free_memory_gb:.1f} GB free memory. "
                f"Using cuda:{best_gpu} ({best_free_gb:.1f} GB free)."
            )
        return str(best_gpu)
    except Exception as exc:
        print(f"[WARN] GPU query failed ({exc}), fallback to cuda:0.")
        return "0"


def get_device(gpu_index: Optional[Union[str, int]] = None, min_free_memory_gb: float = 20) -> torch.device:
    if str(gpu_index).strip().lower() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")

    preferred_gpu = _parse_gpu_index(gpu_index)
    selected_gpu = auto_select_gpu(min_free_memory_gb=min_free_memory_gb, preferred_gpu=preferred_gpu)
    if selected_gpu is None:
        return torch.device("cpu")
    return torch.device(f"cuda:{selected_gpu}")


def print_gpu_info() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
    if pynvml is None:
        print("pynvml is not available, cannot query GPU memory details.")
        return

    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        for gpu_id in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = meminfo.total / 1024 / 1024 / 1024
            used_gb = meminfo.used / 1024 / 1024 / 1024
            free_gb = meminfo.free / 1024 / 1024 / 1024
            print(f"GPU {gpu_id}: {name}")
            print(f"  Total: {total_gb:.1f} GB")
            print(f"  Used:  {used_gb:.1f} GB")
            print(f"  Free:  {free_gb:.1f} GB")
    except Exception as exc:
        print(f"[WARN] Failed to print GPU info: {exc}")
