import torch
import math
import matplotlib.pyplot as plt

def get_sinusoidal_positional_encoding(seq_len, d_model):
    """计算正弦位置编码。"""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(
        0, seq_len, dtype=torch.float
    ).unsqueeze(1)
    # 形状: (seq_len, 1)

    # 用于计算频率的项
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float()
        * (-math.log(10000.0) / d_model)
    )
    # 形状: (d_model/2)

    # 对偶数索引计算正弦，对奇数索引计算余弦
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # 添加批处理维度（可选，通常稍后完成）
    # pe = pe.unsqueeze(0) # 形状: (1, seq_len, d_model)
    return pe

# 示例：生成序列长度100、模型维度128的编码
seq_len = 100
d_model = 128
fixed_pe = get_sinusoidal_positional_encoding(seq_len, d_model)
# 形状: (100, 128)

print(f"固定位置嵌入的形状: {fixed_pe.shape}")
# 输出: 固定位置嵌入的形状:
# torch.Size([100, 128])

# 可视化前几个维度
plt.figure(figsize=(10, 5))
# 绘制维度 0, 2, 4, 6
for i in range(0, 8, 2):
    plt.plot(fixed_pe[:, i].numpy(), label=f'维度 {i} (sin)')
# 绘制维度 1, 3, 5, 7
for i in range(1, 9, 2):
     plt.plot(
         fixed_pe[:, i].numpy(),
         label=f'维度 {i} (cos)',
         linestyle='--'
     )
plt.ylabel("值")
plt.xlabel("位置")
plt.title("正弦位置编码（前8个维度）")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()