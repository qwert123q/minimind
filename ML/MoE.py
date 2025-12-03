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
        # top_k_weights, top_k_indices 形状：
        # (批次*序列长度, top_k)
        top_k_weights, top_k_indices = torch.topk(
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
            for k in range(self.top_k):
                expert_idx = top_k_indices[i, k].item()
                expert_weight = top_k_weights_norm[i, k]

                # 计算所选专家的输出
                expert_output = self.experts[expert_idx](token_input)

                # 累加加权输出
                final_output[i] += expert_weight * expert_output

        # 重塑回原形状
        return final_output.view(
            batch_size, seq_len, d_model
        ) # 重塑回原形状