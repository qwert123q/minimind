import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim, dropout=0.1):
        super().__init__()
        self.down_project = nn.Linear(d_model, bottleneck_dim)
        self.non_linear = nn.GELU() # 常见选择，也可以是ReLU等。
        self.up_project = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # 将up_project的权重初始化为零或接近零
        # 这使得适配器在初始时表现为恒等函数
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        # x 是来自上一层的输入（例如，MHA或FFN的输出）
        adapter_input = x
        x = self.down_project(x)
        x = self.non_linear(x)
        x = self.up_project(x)
        x = self.dropout(x)
        # 添加残差连接
        output = adapter_input + x
        return output

# Transformer层前向传播中的使用示例
# 假设 `self.mha_adapter` 和 `self.ffn_adapter` 是
# Adapter的实例
# hidden_states = ... MHA的输出 ...
# adapted_mha_output = self.mha_adapter(hidden_states)
# hidden_states = layer_norm(adapted_mha_output + residual_mha_input)
# # 添加主残差和层归一化
#
# feed_forward_output = ... FFN的输出 ...
# adapted_ffn_output = self.ffn_adapter(feed_forward_output)
# hidden_states = layer_norm(adapted_ffn_output + residual_ffn_input)
# # 添加主残差和层归一化