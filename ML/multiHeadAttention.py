import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head) -> None:
        super().__init__()
        self.nums_head = nums_head

        # 一般来说，
        self.head_dim = hidden_dim // nums_head
        self.hidden_dim = hidden_dim

        # 一般默认有 bias，需要时刻主意，hidden_dim = head_dim * nums_head，所以最终是可以算成是 n 个矩阵
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # gpt2 和 bert 类都有，但是 llama 其实没有
        self.att_dropout = nn.Dropout(0.1)
        # 输出时候的 proj
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None):
        # 需要在 mask 之前 masked_fill
        # x shape is (batch, seq, hidden_dim)
        # attention_mask shape is (batch, seq)

        batch_size, seq_len, _ = x.size()

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # shape 变成 （batch_size, num_head, seq_len, head_dim）
        q_state = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)
        k_state = K.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        v_state = V.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        # 主意这里需要用 head_dim，而不是 hidden_dim
        attention_weight = (
            q_state @ k_state.transpose(-1, -2) / math.sqrt(self.head_dim)
        )
        print(type(attention_mask))
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0, float("-1e20")
            )

        # 第四个维度 softmax
        attention_weight = torch.softmax(attention_weight, dim=3)
        print(attention_weight)

        attention_weight = self.att_dropout(attention_weight)
        output_mid = attention_weight @ v_state

        # 重新变成 (batch, seq_len, num_head, head_dim)
        # 这里的 contiguous() 是相当于返回一个连续内存的 tensor，一般用了 permute/tranpose 都要这么操作
        # 如果后面用 Reshape 就可以不用这个 contiguous()，因为 view 只能在连续内存中操作
        # 变成 (batch, seq, hidden_dim)
        output = output_mid.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        return output


attention_mask = (
    torch.tensor(
        [
            [0, 1],
            [0, 0],
            [1, 0],
        ]
    )
    .unsqueeze(1)
    .unsqueeze(2)
    .expand(3, 8, 2, 2)
)

x = torch.rand(3, 2, 128)
net = MultiHeadAttention(128, 8)
net(x, attention_mask).shape