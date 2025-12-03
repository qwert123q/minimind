import torch
import torch.nn as nn
import math
from transformer_components import scaled_dot_product_attention

# 假设scaled_dot_product_attention已在上一节中定义
# def scaled_dot_product_attention(q, k, v, mask=None):
#     d_k = q.size(-1)
#     scores = torch.matmul(q, k.transpose(-2, -1)) / \
#              math.sqrt(d_k)
#     if mask is not None:
#         # 使用一个大的负值
#         scores = scores.masked_fill(mask == 0, -1e9)
#     attn_weights = torch.softmax(scores, dim=-1)
#     output = torch.matmul(attn_weights, v)
#     return output, attn_weights

class MultiHeadAttention(nn.Module):
    """ 实现多头注意力机制。 """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            "Embedding dimension must be divisible by number of heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q、K、V投影的线性层。
        # 为了效率，我们使用一个单一的线性层，
        # 投影到embed_dim * 3然后分割结果。
        # 另外，也可以使用单独的层。
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout) # Dropout层（可选）

        self._reset_parameters()

    def _reset_parameters(self):
        # 对线性层使用Xavier均匀初始化
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, query, key, value, mask=None):
        """
        多头注意力的前向传播。
        Args:
            query (torch.Tensor): 查询张量，
                                  形状 (batch_size, seq_len_q, embed_dim)
            key (torch.Tensor): 键张量，
                                形状 (batch_size, seq_len_k, embed_dim)
            value (torch.Tensor): 值张量，
                                  形状 (batch_size, seq_len_v, embed_dim)
                                注意：通常seq_len_k == seq_len_v。
            mask (torch.Tensor, optional): 掩码张量，用于阻止
                                           对某些位置的注意力。
                                           形状 (batch_size, 1, seq_len_q,
                                                  seq_len_k) 或类似
                                           可广播形状。
        Returns:
            torch.Tensor: 输出张量，
                          形状 (batch_size, seq_len_q, embed_dim)
            torch.Tensor: 注意力权重，
                          形状 (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size, seq_len_q, _ = query.size()
        # 值序列长度必须匹配
        _, seq_len_k, _ = key.size()
        _, seq_len_v, _ = value.size()
        assert seq_len_k == seq_len_v

        # 1. 使用组合线性层投影Q、K、V
        qkv = self.qkv_proj(query) # 投影查询
        # 我们分别投影键和值，以防
        # 它们在编码器-解码器注意力中具有不同的源长度，
        # 尽管此处我们假设自注意力（查询=键=值）
        # 为了通用性，我们假设键和值可能存在独立的输入。
        # 如果查询、值是相同的张量（自注意力），
        # 这比一次投影并分割效率略低，
        # 但更灵活。
        k_proj = self.qkv_proj(key) # 投影键
        v_proj = self.qkv_proj(value) # 投影值

        # 将组合投影分割成Q、K、V
        # qkv形状: (batch_size, seq_len, embed_dim * 3) ->
        #            3个(batch_size, seq_len, embed_dim)形状的张量
        q, k, v = qkv.chunk(3, dim=-1)

        # 如果使用单独的层或只对查询进行不同投影的替代方案：
        # q = self.q_proj(query)
        # k = self.k_proj(key)
        # v = self.v_proj(value)

        # 2. 重塑Q、K、V以进行多头计算
        # 从(batch_size, seq_len, embed_dim)重塑为
        # (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size,
                   seq_len_q,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size,
                   seq_len_k,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size,
                   seq_len_v,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)

        # 3. 对每个头应用缩放点积注意力
        # 掩码需要能够正确广播。
        # 如果掩码形状是(batch_size, seq_len_q, seq_len_k)，则需要
        # 为头维度进行unsqueezing：
        # (batch_size, 1, seq_len_q, seq_len_k)
        if mask is not None:
             # (batch_size, seq_len_q, seq_len_k)
             if mask.dim() == 3:
                 # 添加头维度: (batch_size, 1, seq_len_q, seq_len_k)
                 mask = mask.unsqueeze(1)
             # (seq_len_q, seq_len_k) - 所有批次使用相同掩码
             elif mask.dim() == 2:
                 # 添加批次和头维度: (1, 1, seq_len_q, seq_len_k)
                 mask = mask.unsqueeze(0).unsqueeze(0)
        # 确保掩码形状兼容：
        # (batch_size, num_heads, seq_len_q, seq_len_k)或可广播

        # attn_output形状: (batch_size, num_heads, seq_len_q, head_dim)
        # attn_weights形状: (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask
        )

        # 4. 拼接头并将结果投影回embed_dim
        # 转置并重塑以组合头：
        # (batch_size, seq_len_q, num_heads * head_dim)
        # num_heads * head_dim = embed_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )

        # 应用最终线性投影
        output = self.out_proj(attn_output)

        # 应用dropout（可选）
        output = self.dropout(output)

        return output, attn_weights