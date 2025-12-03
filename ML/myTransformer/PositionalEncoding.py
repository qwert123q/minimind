import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term) # pe[:, 0, 0::2]是index
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # 使用 register_buffer 以便 pe 不作为模型参数
        self.register_buffer('pe', pe)
        print(f"pe shape: {self.pe.shape}")

    def forward(self, x):
        """
        参数：
            x: 张量，形状为 [batch_size, seq_len, embedding_dim]
        """
        # 将位置编码调整为批处理格式 [batch_size, seq_len, embedding_dim]
        # 原始 pe 形状为 [max_len, 1, embedding_dim]。我们需要 [1, seq_len, embedding_dim] 或兼容形状。
        # 对当前序列长度切片 pe 并转置。
        # 形状变为 [1, seq_len, embedding_dim]
        pe_for_seq = self.pe[:x.size(1), :].permute(1, 0, 2)
        print(f"pe_for_seq shape: {pe_for_seq.shape}")
        x = x + pe_for_seq
        print(f"x shape after pe: {x.shape}")
        return self.dropout(x)
    

if __name__ == "__main__":
    d_model = 512
    max_len = 100
    x = torch.randn(2, max_len, d_model)
    pe = PositionalEncoding(d_model, max_len=max_len)
    print(pe(x))
