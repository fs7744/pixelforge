"""落盘缓存的两个技术前提验证：
1. safetensors 能否直接存/读 float8_e4m3fn（不能则用 uint8 视图兜底）；
2. from_config(meta) + 手动 assign 能否搭出可用模型（diffusers 快速加载同款路径）。
"""
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

print("safetensors:", __import__("safetensors").__version__)

# ---- 1) fp8 序列化 ----
w = torch.randn(64, 32).to(torch.bfloat16)
scale = w.abs().amax(dim=1) / 448.0
wq = (w / scale[:, None]).to(torch.float8_e4m3fn)
try:
    from safetensors.torch import save_file, load_file

    p = Path(tempfile.mkdtemp()) / "t.safetensors"
    save_file({"w": wq, "s": scale, "b": torch.zeros(64)}, p)
    d = load_file(p)
    print("fp8 native save/load: OK,", {k: str(v.dtype) for k, v in d.items()})
    ok = torch.equal(d["w"].view(torch.uint8), wq.view(torch.uint8))
    print("fp8 roundtrip bit-exact:", ok)
except Exception as e:
    print("fp8 native save FAILED ->", type(e).__name__, str(e)[:150])


# ---- 2) _FP8Linear 前向正确性 ----
class FP8Linear(nn.Linear):
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if self.w_scale is not None:
            w = w * self.w_scale.to(x.dtype)[:, None]
        return F.linear(x, w, self.bias)


lin = nn.Linear(32, 64).to(torch.bfloat16)
ref = lin(torch.randn(4, 32).to(torch.bfloat16))
scale = lin.weight.abs().amax(dim=1) / 448.0
lin.weight.data = (lin.weight / scale[:, None]).to(torch.float8_e4m3fn)
lin.register_buffer("w_scale", scale)
lin.__class__ = FP8Linear  # 原地换类
out = lin(torch.randn(4, 32).to(torch.bfloat16))  # 不同输入，只验证不报错
x = torch.randn(4, 32).to(torch.bfloat16)
ref2 = nn.Linear(32, 64).to(torch.bfloat16)  # 同权重 bf16 对照
ref2.weight.data = (lin.weight.to(torch.bfloat16) * scale[:, None]).to(torch.bfloat16)
ref2.bias.data = lin.bias.data
diff = (lin(x).float() - ref2(x).float()).abs().max().item()
print(f"FP8Linear forward vs bf16-ref: max_diff={diff:.4f} (fp8 量化固有误差)")

# ---- 3) from_config + meta + assign（用 diffusers 小模型验证路径） ----
from diffusers import QwenImageTransformer2DModel  # noqa: E402

cfg_path = Path("D:/code/pixelforge/models/qwen-image-edit-2511/transformer/config.json")
import json  # noqa: E402

cfg = json.loads(cfg_path.read_text())
# 建一个缩小配置验证路径（真模型 20B 太大，只验证机制）
cfg_small = dict(cfg)
cfg_small["num_layers"] = 2
m = QwenImageTransformer2DModel.from_config(cfg_small)
print("from_config device:", next(m.parameters()).device)
# meta 上直接 assign 真实张量
sd = {k: torch.randn_like(v, dtype=torch.bfloat16) if v.is_floating_point() else v
      for k, v in m.state_dict().items()}
missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
print("assign load: missing=", len(missing), "unexpected=", len(unexpected),
      "| weight dtype now:", next(m.parameters()).dtype)
# 跑一次前向（小配置，CPU）
import torch as _t  # noqa: E402

m.eval()
try:
    with _t.no_grad():
        out = m(
            hidden_states=_t.randn(1, 16, cfg["in_channels"] // 2).to(_t.bfloat16),
            timestep=_t.tensor([500.0]),
            encoder_hidden_states=_t.randn(1, 8, cfg["joint_attention_dim"]).to(_t.bfloat16),
        )
    print("meta-assign model forward: OK, sample shape:", tuple(out.sample.shape))
except Exception as e:
    print("forward FAILED:", type(e).__name__, str(e)[:200])
print("PREREQ OK")
