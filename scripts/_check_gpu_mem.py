import torch
t = torch.cuda.get_device_properties(0).total_memory
print(f"TOTAL_GPU_BYTES={t}")
print(f"TOTAL_GPU_GIB={t / (1024**3):.2f}")
