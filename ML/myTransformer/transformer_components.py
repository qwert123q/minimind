import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算缩放点积注意力。

    参数：
        query: 查询张量；形状为 (batch_size, num_heads, seq_len_q, d_k)
               或 (batch_size, seq_len_q, d_k) (如果为单头)。
        key: 键张量；形状为 (batch_size, num_heads, seq_len_k, d_k)
             或 (batch_size, seq_len_k, d_k) (如果为单头)。
        value: 值张量；形状为 (batch_size, num_heads, seq_len_v, d_v)
               或 (batch_size, seq_len_v, d_v) (如果为单头)。
               注意：seq_len_k 和 seq_len_v 必须相同。
        mask: 可选的掩码张量；形状应可广播到
              (batch_size, num_heads, seq_len_q, seq_len_k)。
              `True` 或 `1` 的位置将被掩盖（设为 -inf）。

    返回：
        包含以下内容的元组：
        - output：注意力输出张量；
                  形状为 (batch_size, num_heads, seq_len_q, d_v)
                  或 (batch_size, seq_len_q, d_v) (如果为单头)。
        - attention_weights：注意力权重张量；
                             形状为 (batch_size, num_heads, seq_len_q, seq_len_k)
                             或 (batch_size, seq_len_q, seq_len_k) (如果为单头)。
    """
    # 确保维度与矩阵乘法兼容
    # K 需要形状 (..., d_k, seq_len_k) 才能与 Q (..., seq_len_q, d_k) 进行矩阵乘法
    # 结果形状: (..., seq_len_q, seq_len_k)
    d_k = query.size(-1)
    scores = (torch.matmul(query, key.transpose(-2, -1))
              / math.sqrt(d_k))

    # 如果提供了掩码，则应用掩码（将掩盖位置设为一个大的负值）
    if mask is not None:
        # 确保掩码具有兼容的维度或可以广播
        # 常见掩码形状：(batch_size, 1, 1, seq_len_k) 用于填充掩码
        #                    (batch_size, 1, seq_len_q, seq_len_k) 用于组合掩码
        # 我们添加一个大的负值，而不是直接使用布尔掩码
        # 以确保与各种 PyTorch 版本和操作兼容。
        # 当掩码为 True（或 1）时，我们希望用 -inf 替换分数。
        scores = scores.masked_fill(mask == True, float('-inf'))
        # 或者使用一个大的负数，如 -1e9

    # 应用 softmax 以获得注意力概率
    # Softmax 应用于最后一个维度 (seq_len_k)
    attention_weights = F.softmax(scores, dim=-1)

    # 检查 softmax 后可能出现的 NaN，这可能发生在某一行中所有分数都为 -inf 的情况下
    # 这可能表明掩码或输入数据存在问题
    if torch.isnan(attention_weights).any():
        print("警告：在注意力权重中检测到 NaN。 "
              "请检查掩码或输入数据。")
        # （可选）处理 NaN，例如，将其设为 0，
        # 尽管这可能会隐藏潜在问题。
        # attention_weights = torch.nan_to_num(attention_weights)

    # 权重乘以值
    # 结果形状: (..., seq_len_q, d_v)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights

# 示例用法（为简单起见，假设为单头）
batch_size = 2
seq_len_q = 5 # 查询序列长度
seq_len_k = 7 # 键/值序列长度
d_k = 64      # 键/查询的维度
d_v = 128     # 值的维度

# 虚拟张量
query_tensor = torch.randn(batch_size, seq_len_q, d_k)
key_tensor = torch.randn(batch_size, seq_len_k, d_k)
value_tensor = torch.randn(batch_size, seq_len_k, d_v) # seq_len_k == seq_len_v

# 填充掩码示例（掩盖键/值序列的最后两个元素）
padding_mask = torch.zeros(batch_size, 1, seq_len_k, dtype=torch.bool)
padding_mask[:, :, -2:] = True # 掩盖位置 5 和 6

# 计算注意力
output_tensor, attention_weights_tensor = scaled_dot_product_attention(
    query_tensor,
    key_tensor,
    value_tensor,
    mask=padding_mask
)

print("输出形状:", output_tensor.shape) # 预期：[2, 5, 128]
print("注意力权重形状:", attention_weights_tensor.shape) # 预期：[2, 5, 7]

print("批次 0 中第一个查询的注意力权重 "
      "(最后两个键已被掩盖)：")
print(attention_weights_tensor[0, 0, :])