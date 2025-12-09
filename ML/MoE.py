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
假设：

batch_size = 1
seq_len = 2（一句话里有两个 token）
d_model = 4（每个 token 向量4维）
num_experts = 3（3个专家）
top_k = 2（每个 token 只选2个专家）

1.输入 x 形状：(1, 2, 4)，可以写成：

token0: [1.0, 0.5, -0.2, 0.3]
token1: [-0.1, 0.7, 0.4, 0.0]

2.展平后变成 (2, 4)，两行分别是上述两个 token。

3.每个 token 通过 gating_network 得到 3 个 logit，例如：

对 token0：router_logits[0] = [2.0, 1.0, -1.0]
对 token1：router_logits[1] = [0.5, 1.5, 0.0]

4.softmax 后得到 routing_weights，例如：

token0 对专家的权重（近似）：
expert0: 0.71
expert1: 0.26
expert2: 0.03
token1 对专家的权重（近似）：
expert0: 0.24
expert1: 0.65
expert2: 0.11

5.top_k = 2，所以只取最大两个专家：

token0:
indices: [0, 1]（专家0和1）
weights: [0.71, 0.26]
token1:
indices: [1, 0]（专家1和0）
weights: [0.65, 0.24]
再归一化，让2个权重和为1（这里已经差不多接近1，就略过）。

6.对 token0：

用专家0处理：y0 = Expert0(token0) 得到一个 4 维向量
用专家1处理：y1 = Expert1(token0) 得到一个 4 维向量
最终输出：out_token0 = 0.71 * y0 + 0.26 * y1（忽略了专家2）
对 token1 类似：

y1' = Expert1(token1)
y0' = Expert0(token1)
out_token1 = 0.65 * y1' + 0.24 * y0'

7.把 out_token0 和 out_token1 堆成 (1, 2, 4) 返回。
"""