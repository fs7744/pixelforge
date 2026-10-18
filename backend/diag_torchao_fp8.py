"""torchao fp8 验证 2：量化后 .to(cuda) 搬运（模拟 accelerate offload 钩子）+ 计时。"""
import time

import torch
import torch.nn as nn
from torchao.quantization import Float8WeightOnlyConfig, quantize_

torch.manual_seed(0)

# 模拟 transformer：多个 Linear 组成的序列
mod = nn.Sequential(*[nn.Linear(3072, 3072) for _ in range(60)]).to(torch.bfloat16)
x = torch.randn(2, 512, 3072, device="cuda", dtype=torch.bfloat16)

ref = mod.cuda()(x).float()
del mod
torch.cuda.empty_cache()

mod = nn.Sequential(*[nn.Linear(3072, 3072) for _ in range(60)]).to(torch.bfloat16)
quantize_(mod, Float8WeightOnlyConfig())
# 模拟 offload 钩子：量化后再搬上 GPU
mod.to("cuda")
out = mod(x).float()
diff = (out - ref).abs().max().item()
print(f"quantize->cuda->forward OK, max_diff={diff:.4f}")

# 计时：模拟 40 步 × 每步多次 matmul（每步每层 4 次 Linear）
mod_cpu = nn.Sequential(*[nn.Linear(3072, 3072) for _ in range(60)]).to(torch.bfloat16)
quantize_(mod_cpu, Float8WeightOnlyConfig())
t0 = time.time()
for step in range(3):
    mod_cpu.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    t_move = time.time() - t0
    for i in range(4):
        mod_cpu(x)
    torch.cuda.synchronize()
    print(f"step {step}: cpu->gpu move {t_move*1000:.0f}ms, 4x matmul over 60 layers "
          f"{(time.time()-t0-t_move)*1000:.0f}ms")
    mod_cpu.to("cpu")
    t0 = time.time()
print("VALIDATION 2 OK")
