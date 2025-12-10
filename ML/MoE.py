import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """一个简单的前馈网络专家"""
    # 专家就是简单的mlp网络
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU() # 或者GeLU、SwiGLU等

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class MoELayer(nn.Module):
    """专家混合层"""
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # 专家池
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(num_experts)]
        )

        # 门控网络（学习将token路由到专家）
        # 输入：token表示，输出：每个专家的分数 (num_experts)
        self.gating_network = nn.Linear(d_model, num_experts)

    def forward(self, x):

        batch_size, seq_len, d_model = x.shape
        # 重塑为 (批次*序列长度, d_model)。二维token（第几个样本x这句的第几个token）-> 一维token
        x_reshaped = x.view(-1, d_model)

        # 1. 从门控网络获取路由权重
        # logits 形状：(批次*序列长度, 专家数量)
        router_logits = self.gating_network(x_reshaped)
        # 对专家进行Softmax
        routing_weights = F.softmax(router_logits, dim=1)

        # 2. 选择Top-k专家并获取其权重
        # top_k_weights, top_k_indexes 形状：
        # (批次*序列长度, top_k)
        # 举例：top_k_indexes[0, 1]表示第 1 个token的第 2 个选中专家的索引
        top_k_weights, top_k_indexes = torch.topk(
            routing_weights, self.top_k, dim=1
        )

        # 再归一化Top-k权重：本来每个 token 对所有专家都有一个分配比例，但我只保留前 k 个最重要的，然后把它们重新拉到和为 1
        top_k_weights_norm = (
            top_k_weights / top_k_weights.sum(dim=1, keepdim=True)
        )

        # 3. 计算专家输出（简化 -
        #    实际实现为提高效率更为复杂）
        # 初始化最终输出张量
        final_output = torch.zeros_like(x_reshaped)

        # 这是一个简化的循环，
        # 高效实现使用scatter/gather操作
        for i in range(batch_size * seq_len):
            token_input = x_reshaped[i]
            # 表示当前处理的是该token选中的第几个专家
            for k in range(self.top_k):
                expert_idx = top_k_indexes[i, k].item()
                expert_weight = top_k_weights_norm[i, k]

                # 计算所选专家的输出
                expert_output = self.experts[expert_idx](token_input)

                # 累加加权输出
                final_output[i] += expert_weight * expert_output

        # 重塑回原形状
        return final_output.view(
            batch_size, seq_len, d_model
        ) # 重塑回原形状
    

"""
假设配置：batch_size=1, seq_len=2, d_model=4
x = torch.tensor([[
    [0.1, 0.2, 0.3, 0.4],  # token0
    [0.5, 0.6, 0.7, 0.8]   # token1
]], dtype=torch.float32)



1. 输入重塑：[[0.1,0.2,0.3,0.4], [0.5,0.6,0.7,0.8]]

2. 门控网络输出示例：
router_logits = [[0.5, 1.2, 0.8, 0.3], [0.9, 0.4, 1.5, 0.7]]
routing_weights = [[0.15, 0.45, 0.30, 0.10], [0.20, 0.10, 0.50, 0.20]]  # 已归一化

3. Top-k=2选择：
top_k_weights = [[0.45, 0.30], [0.50, 0.20]]
top_k_indexes = [[1, 2], [2, 3]]  # token0选专家1、2；token1选专家2、3

4. 重新归一化：
top_k_weights_norm = torch.tensor([
    [0.6, 0.4],  # token 0 对专家1的权重0.6，专家2的权重0.4
    [0.7, 0.3]   # token 1 对专家2的权重0.7，专家3的权重0.3
])

token_input = x_reshaped[0]  # [0.1, 0.2, 0.3, 0.4]

5. 对于token0：
# 处理第1个选中专家（k=0）：
expert_idx = top_k_indexes[0, 0].item()  # 1
expert_weight = top_k_weights_norm[0, 0]  # 0.6
expert_output = self.experts[1](token_input)  # 假设输出 [0.2, 0.3, 0.4, 0.5]
final_output[0] += 0.6 * [0.2, 0.3, 0.4, 0.5]  # final_output[0] = [0.12, 0.18, 0.24, 0.30]

# 处理第2个选中专家（k=1）：
expert_idx = top_k_indexes[0, 1].item()  # 2
expert_weight = top_k_weights_norm[0, 1]  # 0.4
expert_output = self.experts[2](token_input)  # 假设输出 [0.3, 0.4, 0.5, 0.6]
final_output[0] += 0.4 * [0.3, 0.4, 0.5, 0.6]  # final_output[0] = [0.12+0.12, 0.18+0.16, 0.24+0.20, 0.30+0.24] = [0.24, 0.34, 0.44, 0.54]

6. 最终输出
"""