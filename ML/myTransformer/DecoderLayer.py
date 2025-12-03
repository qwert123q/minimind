import torch
import torch.nn as nn
import copy
from MultiHeadAttention import MultiHeadAttention
from PositionWiseFeedForward import PositionWiseFeedForward

# 假设MultiHeadAttention和PositionwiseFeedForward类已在其他地方定义
# from .attention import MultiHeadAttention
# from .feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    """
    表示Transformer解码器的一层。

    它由带掩码的自注意力、交叉注意力（关注
    编码器输出）和一个位置前馈网络组成。残差
    连接和层归一化在每个子层之后应用。
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        参数：
            d_model: 输入和输出的维度。
            num_heads: 注意力头的数量。
            d_ff: 前馈网络内部层的维度。
            dropout: dropout比率。
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                src_mask: torch.Tensor | None = None,
                tgt_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        将输入通过解码器层。

        参数：
            x: 解码器层的输入张量
               (batch_size, tgt_seq_len, d_model)。
            memory: 编码器堆栈的输出张量
                    (batch_size, src_seq_len, d_model)。
            src_mask: 交叉注意力（编码器-解码器注意力）层的掩码，用于忽略
                      编码器输出中的填充标记（可选）。
                      形状为 (batch_size, 1, src_seq_len)。
            tgt_mask: 带掩码自注意力层的掩码，结合了
                      前瞻掩码和目标填充掩码（可选）。
                      形状为 (batch_size, tgt_seq_len, tgt_seq_len)。

        返回：
            层的输出张量
            (batch_size, tgt_seq_len, d_model)。
        """
        # 1. 带掩码多头自注意力
        # 目标掩码 (tgt_mask) 避免关注
        # 未来位置。
        self_attn_output, _ = self.self_attn(query=x,
                                             key=x,
                                             value=x,
                                             mask=tgt_mask)
        # 应用残差连接和层归一化
        x = self.norm1(x + self.dropout(self_attn_output)) # Add -> Norm

        # 2. 多头交叉注意力（编码器-解码器注意力）
        # 查询来自解码器 (x)，键/值来自编码器
        # 输出 (memory)。
        # 源掩码 (src_mask) 避免关注
        # 编码器输出中的填充。
        cross_attn_output, _ = self.cross_attn(query=x,
                                               key=memory,
                                               value=memory,
                                               mask=src_mask)
        # 应用残差连接和层归一化
        x = self.norm2(x + self.dropout(cross_attn_output)) # Add -> Norm

        # 3. 位置前馈网络
        ff_output = self.feed_forward(x)
        # 应用残差连接和层归一化
        x = self.norm3(x + self.dropout(ff_output)) # Add -> Norm

        return x