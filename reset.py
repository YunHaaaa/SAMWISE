import torch

torch.cuda.empty_cache()  # PyTorch 내부 캐시 비우기
torch.cuda.ipc_collect()  # inter-process communication 메모리 정리
