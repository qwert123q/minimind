# MiniMind 学习与实践路线（2 周 × 每天约 2 小时）

> 适用人群：有深度学习基础、懂 Transformer/LLM 原理，但不熟悉具体代码实现与工程细节的同学。  
> 目标：通过 MiniMind 项目系统掌握“小型 LLM 从 0 到 1”的完整实现与训练流程，并完成若干动手小项目。

---

## 一、项目概览：你能在这里学到什么

本仓库是一个 **从零实现 + 贯穿全流程** 的小型 LLM 教学/实战项目，你可以在这里学到：

- **模型结构实现**

  - 文件：`model/model_minimind.py`
  - 学习内容：如何从零实现 GPT 类模型（Embedding、Attention、MLP、RMSNorm、MoE 等），配置类 `MiniMindConfig`，以及顶层 `MiniMindForCausalLM` 的 `forward`。

- **数据与 Dataset 管线**

  - 文件：`dataset/lm_dataset.py`
  - `PretrainDataset`：自回归预训练数据构造（文本 → token → `X/Y/loss_mask`）。
  - `SFTDataset`：对话式监督微调（用 `apply_chat_template` 生成 prompt，只在 assistant 回复区域打 loss）。
  - `DPODataset` / `RLAIFDataset`：偏好学习与 RLAIF 数据的组织方式。

- **训练工程与脚手架**

  - 文件：
    - 预训练：`trainer/train_pretrain.py`
    - 其他阶段：`trainer/train_full_sft.py`、`train_dpo.py`、`train_ppo.py`、`train_grpo.py`、`train_spo.py`
    - 通用工具：`trainer/trainer_utils.py`
  - 学习内容：
    - 标准训练循环：DataLoader → 前向 → loss → 反向 → 梯度累积 → 混合精度 → 梯度裁剪 → checkpoint。
    - 分布式训练（DDP）的初始化、主进程判断、断点续训。

- **推理与部署**

  - 命令行评估：`eval_llm.py`
  - WebUI：`scripts/web_demo.py`
  - OpenAI API 协议服务端：`scripts/serve_openai_api.py`
  - 学习内容：如何加载权重、构造对话 prompt、使用 `generate` 推理，以及如何暴露成 WebUI/API。

- **与 Transformers 生态互通**
  - 文件：`scripts/convert_model.py`
  - 学习内容：如何从自定义 `.pth` 权重转换为 `transformers` 兼容格式，如何注册 `AutoModelForCausalLM`、保存 `config.json` 和权重文件，支持 `from_pretrained(..., trust_remote_code=True)`。

---

## 二、两周学习计划（精确到代码与文件）

### 第 1 周：模型 + 预训练主线

#### Day 1：整体鸟瞰 + 推理流程

**目标**：从命令行一路跟到模型 `forward`。

- 阅读内容：

  - `README.md` / `README_en.md` 中与 `eval_llm.py`、预训练、SFT 等相关的介绍（可搜索：`eval_llm.py`、`pretrain`）。
  - `eval_llm.py`：
    - `init_model(args)`
    - `main()`：`prompts`、`conversation`、`tokenizer.apply_chat_template`、`model.generate`。

- 建议动手：
  - 在终端运行一次推理（假设你已经有 `MiniMind2` 权重）：
    ```bash
    python eval_llm.py --load_from ./MiniMind2
    ```
  - 修改 `prompts` 列表，加入你自己的问题，看下模型回复。
  - 在 `main()` 中增加简单调试输出，观察张量形状：
    ```python
    print(type(inputs["input_ids"]), inputs["input_ids"].shape)
    ```

---

#### Day 2：模型配置与总体结构

**目标**：弄清模型“长什么样”，参数规模怎么来的。

- 阅读内容：`model/model_minimind.py`

  - `MiniMindConfig`：隐藏维度、层数、是否 MoE 等。
  - `MiniMindForCausalLM` 的 `__init__` 与 `forward`。
  - Block 类（如 `MiniMindBlock` / `DecoderLayer` 等），理解层的堆叠方式。

- 建议动手：

  - 写一个小脚本打印参数量（可在根目录下建 `play_model_size.py`）：

    ```python
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8)
    m = MiniMindForCausalLM(cfg)
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")
    ```

  - 对照 README 中的参数表，看看是否一致。

---

#### Day 3：Attention + MLP 细节

**目标**：把你熟悉的理论与实现逐行对齐。

- 阅读内容：继续看 `model/model_minimind.py`

  - Multi-head Attention 相关类及其 `forward`。
  - MLP / FFN 相关类及其 `forward`。

- 建议动手：
  - 为 Attention 的 `forward` 画一条 shape 流程线：  
    `X: [B, T, C] → q/k/v: [...] → [B, heads, T, head_dim] → attention_scores → output`。
  - 在关键位置加 `assert` 确保逻辑正确，例如：
    ```python
    assert q.shape[-1] == k.shape[-1]
    ```
  - 跑一次 `eval_llm.py` 确认没有触发断言，增强对实现的信心。

---

#### Day 4：预训练数据管线（PretrainDataset）

**目标**：搞清楚训练时一条样本是如何从磁盘变成 `X/Y/loss_mask` 的。

- 阅读内容：`dataset/lm_dataset.py` 中的 `PretrainDataset`

  - `__init__`：记录 `tokenizer`、`max_length`。
  - `load_data`：从 jsonl 里读出 `{"text": ...}`。
  - `__getitem__`：
    - 调用 tokenizer，得到 `input_ids`。
    - 构造 `X`（[:-1]）、`Y`（[1:]）、`loss_mask`，并转换为张量。

- 建议动手：

  - 在根目录下写 `play_pretrain_dataset.py`：

    ```python
    from transformers import AutoTokenizer
    from dataset.lm_dataset import PretrainDataset

    tok = AutoTokenizer.from_pretrained("model")  # 或 README 推荐路径
    ds = PretrainDataset("your_pretrain.jsonl", tok, max_length=32)

    x, y, mask = ds[0]
    print("x:", x.shape, "y:", y.shape, "mask:", mask.shape)
    print("decoded x:", tok.decode(x, skip_special_tokens=False))
    print("mask:", mask[:20])
    ```

  - 通过解码 `x`，对照 `mask`，理解哪些 token 被用于计算 loss。

---

#### Day 5：训练工具（trainer_utils.py）

**目标**：掌握训练脚手架：学习率、DDP、checkpoint、模型加载。

- 阅读内容：`trainer/trainer_utils.py`

  - `get_lr`：余弦学习率调度。
  - `init_distributed_mode`：从环境变量判断是否启用 DDP，设置 `LOCAL_RANK` 对应的 GPU。
  - `setup_seed`：多进程统一随机种子设置。
  - `lm_checkpoint`：
    - 保存：模型半精度权重、优化器状态、epoch/step、world_size、wandb 等。
    - 恢复：在 GPU 数量变化时调整 step。
  - `init_model`：根据 config 与权重路径加载模型与 tokenizer。

- 建议动手：
  - 单独写一段代码调用 `get_lr`，画出一个简单的曲线或在控制台打印几个关键 step 的值，理解其变化趋势。
  - 尝试手动调用 `lm_checkpoint` 加载已有的 `_resume.pth` 文件，打印其中的 `epoch` 和 `step`。

---

#### Day 6：预训练训练脚本（train_pretrain.py）

**目标**：吃透一个 epoch 内每一步的逻辑。

- 阅读内容：`trainer/train_pretrain.py`

  - `train_epoch`：
    - 数据取出：`for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1)`。
    - 学习率更新：`get_lr(...)` + 对 `optimizer.param_groups` 的赋值。
    - 前向：`res = model(X)`（注意 `res.logits` 和 `res.aux_loss`）。
    - loss：
      - `nn.CrossEntropyLoss(reduction='none')` → reshape → 应用 `loss_mask`。
      - 加上 `res.aux_loss`（如 MoE 的负载均衡 loss）。
      - 再除以 `accumulation_steps`。
    - 梯度累积与混合精度：
      - `with autocast_ctx:`
      - `scaler.scale(loss).backward()`
      - 每隔 `accumulation_steps` 做一次 `step`、`clip_grad_norm_`、`zero_grad`。
    - 保存模型：
      - 半精度保存 state_dict。
      - 调用 `lm_checkpoint(...)`。
  - `if __name__ == "__main__":`：解析参数、构造 DataLoader、初始化模型/优化器/scaler、分布式初始化等。

- 建议动手：
  - 在 `train_epoch` 的日志打印处增加 `perplexity` 估计：
    ```python
    import math
    ...
    ppl = math.exp(current_loss) if current_loss < 20 else float('inf')
    Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) '
           f'loss:{current_loss:.6f} ppl:{ppl:.2f} lr:{current_lr:.6e}')
    ```
  - 用很小的 toy 数据和 batch 试跑一次：
    ```bash
    python trainer/train_pretrain.py --epochs 1 --batch_size 2 --max_seq_len 64 --data_path your_pretrain.jsonl
    ```

---

#### Day 7：总结与回顾

**目标**：用自己的话复述第 1 周内容。

- 建议内容：
  - 画一张图或写一段话，总结从 “磁盘上的 jsonl 文本” → `PretrainDataset` → `DataLoader` → `model.forward` → `loss` 的完整路径。
  - 总结 `MiniMindForCausalLM` 的大致结构（Embedding → Blocks → LM Head）和关键配置项。
  - 写出你理解的“预训练一步中发生的事情”的伪代码。

---

### 第 2 周：SFT/DPO/RL + LoRA + 部署

#### Day 8：SFT 数据集 + 训练

**目标**：掌握对话式监督微调如何落在代码上。

- 阅读内容：

  - `dataset/lm_dataset.py` 中的 `SFTDataset`：
    - `_create_chat_prompt`：如何用 `apply_chat_template` 拼成对话 prompt。
    - `_generate_loss_mask`：如何只在 assistant 回复区间打 `1`。
    - `__getitem__`：`prompt` → `input_ids` → `X/Y/loss_mask`。
  - `trainer/train_full_sft.py`：整体结构与 `train_pretrain.py` 的相似与差异（主要在数据和 loss）。

- 建议动手：
  - 写脚本抽一条 SFT 样本，打印：
    - 原始 `conversations`。
    - 生成的 `prompt` 文本。
    - 对应的 `loss_mask` 中 `1` 所在位置（可以只看前 200 个 token）。
  - 思考：如果你想只训练助手的“最后一句回复”，可以如何改 `_generate_loss_mask` 的逻辑。

---

#### Day 9：LoRA 适配

**目标**：理解如何在现有模型上叠加 LoRA 层，只训练少量参数。

- 阅读内容：

  - `model/model_lora.py`：
    - LoRA 层的定义（如 `LoRALinear`）。
    - `apply_lora(model)` / `load_lora(model, path)`。
  - `trainer/train_lora.py`：
    - 查看如何加载 base 模型，只训练 LoRA 参数。

- 建议动手：
  - 写脚本对比：
    - 原始模型可训练参数量：`sum(p.numel() for p in model.parameters() if p.requires_grad)`。
    - 开启 LoRA 时的可训练参数量。
  - 在 `model_lora.py` 中尝试修改一个配置（比如默认的 rank `r`），新开一个 LoRA 训练脚本实验一下。

---

#### Day 10：DPO 数据集 + 训练思想

**目标**：理解偏好学习（preference learning）的数据和 loss 形式。

- 阅读内容：

  - `dataset/lm_dataset.py` 中 `DPODataset`：
    - `__getitem__`：从一条 jsonl 的 `chosen` / `rejected` 构造 `x_*/y_*/mask_*`。
    - `_generate_loss_mask`：继续沿用 SFT 的 mask 逻辑。
  - `trainer/train_dpo.py`：
    - 找到 DPO loss 的实现部分，留意：
      - 用模型计算 `logprob(chosen)` 与 `logprob(rejected)`。
      - 使用 β 值和 log prob 差构造 loss。

- 建议动手：
  - 打印一条 DPO 样本的 chosen/rejected 文本，直观感受“偏好差异”。
  - 在 `train_dpo.py` 的 loss 计算附近加调试输出，打印某几个 batch 的 `chosen_logprob` / `rejected_logprob` / `loss`。

---

#### Day 11：RL（PPO/GRPO/SPO）代码粗读

**目标**：理解 RL 阶段的代码结构，而不是细节数学推导。

- 阅读内容：

  - 浏览 `trainer/train_ppo.py`、`trainer/train_grpo.py`、`trainer/train_spo.py`：
    - 看看它们如何：
      - 使用当前 policy 生成回复（rollout）。
      - 计算 reward（可能来自奖励模型或启发式规则）。
      - 计算 advantage、更新 policy。

- 建议动手：
  - 用自己的话写一份：“在这个仓库中，一次 PPO 训练循环的大致 5 个步骤”。

---

#### Day 12：Transformers 格式转换与集成

**目标**：理解如何把自定义模型权重转换为 `transformers` 兼容格式。

- 阅读内容：`scripts/convert_model.py`

  - `convert_torch2transformers_minimind`：
    - 注册 AutoClass：`MiniMindConfig.register_for_auto_class()` 等。
    - 加载 `.pth` 权重，构造 `MiniMindForCausalLM`，保存为 `save_pretrained(...)`。
    - 保存 tokenizer。
  - `convert_torch2transformers_llama`：如何用 LLaMA 结构做兼容。

- 建议动手：
  - 在笔记中梳理：
    - `.pth` → `MiniMindForCausalLM` → `save_pretrained` → `config.json` + 权重文件。
    - `AutoModelForCausalLM.from_pretrained(--load_from, trust_remote_code=True)` 如何利用这些文件。

---

#### Day 13：API / WebUI 部署

**目标**：学习如何把模型包装成 WebUI 或 OpenAI 风格 API。

- 阅读内容：

  - `scripts/serve_openai_api.py`：
    - 看启动 HTTP 服务的方式、路由定义、如何在请求中调用模型。
  - `scripts/web_demo.py`：
    - 看 `streamlit` 如何构建简单聊天界面，如何在按钮事件中调用模型。

- 建议动手：
  - 如果环境允许，运行：
    ```bash
    cd scripts
    streamlit run web_demo.py
    ```
  - 修改 WebUI 中默认的 system prompt、temperature，观察输出变化。

---

#### Day 14：收尾项目 + 总结

**目标**：做一次完整的“小改动 + 跑通 + 总结”，把两周的理解固化下来。

可以从以下几个小项目中任选其一或组合：

- **选项 A：自定义学习率调度器**

  - 在 `trainer/trainer_utils.py` 中新增 `get_lr_with_warmup`：
    - 前若干步线性 warmup，从 0 → `lr`。
    - 之后使用余弦衰减。
  - 在 `trainer/train_pretrain.py` 中替换原来的 `get_lr` 调用，训练几十步对比 loss 曲线。

- **选项 B：新增 PlainTextDataset（下面有详细代码）**

  - 在 `dataset/lm_dataset.py` 中新增 `PlainTextDataset`，支持从纯 txt 多行文本中读取样本。
  - 写一个简化版预训练脚本，仅利用 `PlainTextDataset`，在小语料上跑一小段。

- **选项 C：模型结构小改动**
  - 在 MLP 中增加一个 gate（例如 `x = x * torch.sigmoid(Wg(x))`），通过新 config 字段控制是否启用。
  - 确保旧 config 能加载（给默认值），跑几步训练验证路径正确。

最后，用 10–20 分钟写下总结：

- 描述这个项目的整体 pipeline。
- 描述从数据到 loss 的完整路径。
- 描述至少一个你自己改过/实现过的小功能。

---

## 三、实践项目示例：PlainTextDataset + mini 预训练脚本

下面是一个相对简单、又很有帮助的实践项目：**为预训练阶段增加一个可以读取纯 txt 文本的 Dataset，并写一个极简预训练脚本**。  
你可以对照当前 `PretrainDataset` 的实现，理解和对比两种数据预处理方式。

> 提示：以下代码只是示例，建议你手动敲一遍，有助于加深记忆。

### 1. 在 `dataset/lm_dataset.py` 中新增 PlainTextDataset

可以在文件中 `PretrainDataset` 之后新增：

```python
class PlainTextDataset(Dataset):
    """
    从纯文本文件中读取数据进行预训练。

    支持两种基本模式：
    1）按行：每一行作为一个独立样本；
    2）拼接模式：把所有文本拼接成一个长序列，再切成若干 fixed-length 片段。
    这里先实现按行版本，便于理解。
    """

    def __init__(self, txt_path, tokenizer, max_length=512, shuffle=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lines = self._load_lines(txt_path)

        if shuffle:
            random.shuffle(self.lines)

    def _load_lines(self, path):
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        text = self.lines[index]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze(0)

        # pad 位置不计算 loss
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = input_ids[:-1].long()
        Y = input_ids[1:].long()
        loss_mask = loss_mask[1:].long()
        return X, Y, loss_mask
```

你可以对比 `PretrainDataset` 的实现，注意：

- 二者都使用了 `tokenizer(...)` 和 `max_length`。
- 都构造了 `X = input_ids[:-1]`、`Y = input_ids[1:]`。
- `loss_mask` 用于忽略 padding 区域。

如果你想进一步练习，可以自己实现“长文本拼接再切片”的版本：

- 思路：把 `lines` 拼到一起 → 用 tokenizer 编码成一个超长 `input_ids` → 按 `max_length` 切成多段样本。

### 2. 编写一个极简预训练脚本（mini_train_plaintext.py）

在项目根目录新建一个文件，例如 `mini_train_plaintext.py`，用于在小语料上快速验证：

```python
import os
import math

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PlainTextDataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 准备 tokenizer、数据与 DataLoader
    tokenizer = AutoTokenizer.from_pretrained("model")  # 或 README 推荐路径

    dataset = PlainTextDataset(
        txt_path="data/toy_pretrain.txt",  # 你准备的一小份 txt
        tokenizer=tokenizer,
        max_length=128,
        shuffle=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # 2. 构造一个小配置的 MiniMind 模型
    lm_config = MiniMindConfig(
        hidden_size=512,
        num_hidden_layers=4,
        max_seq_len=128,
        use_moe=False,
    )
    model = MiniMindForCausalLM(lm_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    model.train()

    # 3. 训练若干步，观察 loss 变化
    for step, (X, Y, loss_mask) in enumerate(loader, start=1):
        X = X.to(device)
        Y = Y.to(device)
        loss_mask = loss_mask.to(device)

        outputs = model(X)
        logits = outputs.logits

        # logits: [B, T, V] → [B*T, V]；Y: [B, T] → [B*T]
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            Y.view(-1),
        ).view(Y.size())

        loss = (loss * loss_mask).sum() / loss_mask.sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            ppl = math.exp(loss.item()) if loss.item() < 20 else float("inf")
            print(f"step {step} | loss {loss.item():.4f} | ppl {ppl:.2f}")

        if step >= 50:
            break


if __name__ == "__main__":
    main()
```

使用方式（示例）：

1. 准备一个小文本文件，例如 `data/toy_pretrain.txt`：
   ```text
   今天天气很好，我们一起学习大模型代码。
   预训练的目标是预测下一个 token。
   这是一个用于测试的简易语料文件。
   ```
2. 运行：
   ```bash
   python mini_train_plaintext.py
   ```
3. 观察输出的 loss / ppl 是否逐渐下降（不必很稳定，只要能运行且有趋势即可）。

通过这个小脚本，你可以在不修改主训练代码的情况下，快速实验：

- 对预训练目标的理解是否正确；
- 模型的前向与损失计算是否符合预期；
- 不同学习率、batch_size、max_length 对收敛速度的影响。

---

## 四、其他可选的动手项目（摘要）

除了上面详细展开的 PlainTextDataset + mini 预训练脚本，你还可以根据兴趣选择：

- **学习率实验**（`trainer/trainer_utils.py` + `trainer/train_pretrain.py`）

  - 新增带 warmup 的 `get_lr_with_warmup`，对比不同调度策略下 loss 曲线差异。

- **SFT loss mask 变体**（`SFTDataset`）

  - 修改 `_generate_loss_mask`，尝试：
    - 只训练助手最后一句；
    - 或者增加对 system 提示的训练。

- **模型结构小改动**（`model/model_minimind.py`）
  - 在 MLP 中增加 gate 或更换激活函数（如 ReLU → SiLU/GELU）。
  - 加一个 config 开关来控制是否启用，体会“新增结构 + 配置 + 兼容旧权重”的流程。

你可以把自己的实验记录（命令、 loss 曲线截图、思考）继续写在本文件的后面，形成你的专属 MiniMind 学习档案。
