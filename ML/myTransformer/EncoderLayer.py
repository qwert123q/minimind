import torch
import torch.nn as nn
import copy
from MultiHeadAttention import MultiHeadAttention
from PositionWiseFeedForward import PositionWiseFeedForward 

# 假设MultiHeadAttention和PositionwiseFeedForward类已在其他地方定义
# from .attention import MultiHeadAttention
# from .feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """
    表示Transformer编码器的一层。

    它由一个多头自注意力机制，后接一个
    位置全连接前馈网络组成。残差连接
    和层归一化在每个子层之后应用。
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        参数：
            d_model: 输入和输出的维度
                     （嵌入维度）。
            num_heads: 注意力头的数量。
            d_ff: 前馈网络内部层的维度。
            dropout: dropout比率。
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        将输入通过编码器层。

        参数：
            x: 层的输入张量 (batch_size, seq_len, d_model)。
            mask: 自注意力机制的掩码（可选）。
                  通常用于忽略填充标记。
                  形状为 (batch_size, 1, seq_len) 或
                  (batch_size, seq_len, seq_len)。

        返回：
            层的输出张量 (batch_size, seq_len, d_model)。
        """
        # 1. 多头自注意力
        attn_output, _ = self.self_attn(query=x, key=x, value=x, mask=mask)
        # 应用残差连接和层归一化
        x = self.norm1(x + self.dropout(attn_output)) # Add -> Norm

        # 2. 位置前馈网络
        ff_output = self.feed_forward(x)
        # 应用残差连接和层归一化
        x = self.norm2(x + self.dropout(ff_output)) # Add -> Norm

        return x
