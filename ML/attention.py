# 导入相关需要的包
import math
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings(action="ignore")


class SelfAttV1(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttV1, self).__init__()
        self.hidden_dim = hidden_dim
        # 一般 Linear 都是默认有 bias
        # 一般来说， input dim 的 hidden dim
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        # 打印输入形状
        print("Input shape:", X.shape)
        
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)
        
        # 打印Q, K, V的形状
        print("Q shape:", Q.shape)
        print("K shape:", K.shape)
        print("V shape:", V.shape)

        # 计算注意力得分
        attention_value = torch.matmul(Q, K.transpose(-1, -2))
        print("Attention value shape:", attention_value.shape)
        
        # 计算注意力权重
        attention_wight = torch.softmax(
            attention_value / math.sqrt(self.hidden_dim), dim=-1
        )
        print("Attention weights shape:", attention_wight.shape)
        print("Attention weights:", attention_wight)
        
        # 计算最终输出
        output = torch.matmul(attention_wight, V)
        print("Output shape:", output.shape)
        return output


# 测试代码
print("=== 开始测试自注意力模块 ===")
X = torch.rand(3, 2, 4)  # 3个样本，每个样本长度为2，特征维度为4
print("Test input:", X)
net = SelfAttV1(4)
result = net(X)
print("Final output:", result)
print("=== 测试完成 ===")