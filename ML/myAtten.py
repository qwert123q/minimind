import torch
import torch.nn as nn
import math


class myAtten(nn.Module):
    def __init__(self,embed_dim,num_heads,head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)

        # 1. 线性投影Q、K、V
        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        nums_heads = self.num_heads
        head_dim = self.head_dim
        
        atten_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        value = torch.matmul(atten_scores, v)
        # 3. 拼接头并将结果投影回embed_dim
        # 转置并重塑以组合头：
        # (batch_size, seq_len_q, num_heads * head_dim)
        # num_heads * head_dim = embed_dim
        value = value.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )
        # 4. 应用最终线性投影
        output = self.out_proj(value)
        return output, atten_scores
    
if __name__ == '__main__':
    embed_dim = 512
    num_heads = 8
    head_dim = embed_dim // num_heads
    atten = myAtten(embed_dim, num_heads, head_dim)
    query = torch.randn(1, 10, embed_dim)
    key = torch.randn(1, 10, embed_dim)
    value = torch.randn(1, 10, embed_dim)
    output, atten_scores = atten(query, key, value)
    print(output.shape)
    print(atten_scores.shape)