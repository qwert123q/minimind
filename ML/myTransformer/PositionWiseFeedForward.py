import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """实现位置感知前馈网络（FFN）模块。"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        初始化 PositionWiseFeedForward 模块。

        参数:
            d_model (int): 输入和输出特征的维度。
            d_ff (int): 内部层的维度。
            dropout (float): Dropout 概率。默认值为 0.1。
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FFN 模块的前向传播。

        参数:
            x (torch.Tensor): 形状为 (batch_size, seq_len, d_model) 的输入张量。

        返回:
            torch.Tensor: 形状为 (batch_size, seq_len, d_model) 的输出张量。
        """
        # 应用第一个线性层，然后是激活函数，接着是 Dropout，最后是第二个线性层
        # x 形状: (batch_size, seq_len, d_model)
        x = self.linear1(x)      # -> (batch_size, seq_len, d_ff)
        x = self.activation(x)   # -> (batch_size, seq_len, d_ff)
        # Dropout 有时可以放在激活函数之后或第二个线性层之后
        # 我们在这里将其放在第二个线性层之后，与一些实践保持一致。
        x = self.linear2(x)      # -> (batch_size, seq_len, d_model)
        x = self.dropout(x)      # -> (batch_size, seq_len, d_model)
        return x
    
# 示例用法:
d_model = 512  # 模型维度
d_ff = 2048    # 内部维度（通常为 4 * d_model）
dropout_rate = 0.1
batch_size = 4
seq_len = 10

# 创建一个示例输入张量
input_tensor = torch.randn(batch_size, seq_len, d_model)

# 实例化 FFN 层
ffn_layer = PositionWiseFeedForward(d_model, d_ff, dropout_rate)

# 将输入通过 FFN 层
output_tensor = ffn_layer(input_tensor)

print(f"输入形状: {input_tensor.shape}")
print(f"输出形状: {output_tensor.shape}")

# 验证输出维度与 d_model 匹配
assert output_tensor.shape == (batch_size, seq_len, d_model)