import torch
import numpy as np

from transformer_from_scratch.PositionWiseFeedForward import PositionWiseFeedForward

# ========= PositionWiseFeedForward 小例子 =========
print("\n=== PositionWiseFeedForward 示例 ===")

# 构造一个形状为 (batch_size=2, seq_len=3, d_model=4) 的输入
batch_size, seq_len, d_model, d_ff = 2, 3, 4, 8
inp = torch.randn(batch_size, seq_len, d_model, )

print("输入张量形状:", inp.shape)
print("输入张量内容:")
print(inp)

# 实例化 FFN 层
ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)

# 前向传播
out = ffn(inp)

print("\n输出张量形状:", out.shape)
print("输出张量内容:")
print(out)

# 对比同一位置在 batch 内的行为（参数共享）
pos_b0 = inp[0, 0]
pos_b1 = inp[1, 0]

print("\n第 0 个位置（batch 0）的输入向量:", pos_b0)
print("第 0 个位置（batch 1）的输入向量:", pos_b1)

with torch.no_grad():
    out_b0 = ffn(pos_b0.unsqueeze(0).unsqueeze(0))  # 形状变为 (1,1,4)
    out_b1 = ffn(pos_b1.unsqueeze(0).unsqueeze(0))

print("\n第 0 个位置（batch 0）的输出向量:", out_b0[0, 0])
print("第 0 个位置（batch 1）的输出向量:", out_b1[0, 0])

print("=== 示例结束 ===")
