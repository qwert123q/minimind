import torch

# 占位输入序列（batch_size=1, seq_len=5, features=10）
input_seq = torch.randn(1, 5, 10)
# 初始隐藏状态（batch_size=1, hidden_size=20）
h_prev = torch.zeros(1, 20)
# 简单的RNN单元（非实际实现）
rnn_cell = lambda input_t, h_prev: torch.tanh(
    input_t @ torch.randn(10, 20) + h_prev @ torch.randn(20, 20)
) # 简化版

hidden_states = []
# 顺序处理循环
for t in range(input_seq.shape[1]): # 遍历序列长度
    input_t = input_seq[:, t, :]
    h_t = rnn_cell(input_t, h_prev)
    hidden_states.append(h_t)
    h_prev = h_t # 更新隐藏状态以用于下一步

# hidden_states 现在包含每个时间步的状态
# 注意：时间步't'的计算明确依赖于
# 't-1'的结果