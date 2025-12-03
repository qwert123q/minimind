import torch
import torch.nn as nn
import math
from PositionalEncoding import PositionalEncoding
from EncoderLayer import EncoderLayer
from DecoderLayer import DecoderLayer

class TransformerModel(nn.Module):
    """
    完整的 Transformer 模型实现。
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int, nhead: int, num_encoder_layers: int,
                 num_decoder_layers: int, dim_feedforward: int,
                 dropout: float = 0.1, max_len: int = 5000):
        """
        参数：
            src_vocab_size: 源词汇表大小。
            tgt_vocab_size: 目标词汇表大小。
            d_model: 嵌入和模型层的维度。
            nhead: 注意力头数量。
            num_encoder_layers: 堆叠编码器层的数量。
            num_decoder_layers: 堆叠解码器层的数量。
            dim_feedforward: 前馈网络隐藏层的维度。
            dropout: Dropout 比率。
            max_len: 位置编码的最大序列长度。
        """
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # 对层列表使用 nn.ModuleList
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 最终的线性层，用于将解码器输出投影到词汇表大小
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # 可选：目标嵌入和最终线性层之间的权重绑定
        # self.tgt_tok_emb.weight = self.generator.weight # 需要相同维度

        self._reset_parameters()

    def _reset_parameters(self):
        """初始化 Transformer 模型中的参数。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None,
               src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        将源序列通过编码器堆栈。

        参数：
            src: 源序列张量（batch_size, src_seq_len）。
            src_mask: 源序列注意力掩码（src_seq_len, src_seq_len）。
                      如果需要，可防止注意力集中于未来位置（编码器通常不需要）。
            src_key_padding_mask: 源序列中填充标记的掩码（batch_size, src_seq_len）。

        返回：
            编码器输出张量（batch_size, src_seq_len, d_model）。
        """
        # 嵌入标记并添加位置编码
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        # 通过每个编码器层
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask=src_mask,
                           src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: torch.Tensor = None,
               memory_mask: torch.Tensor = None,
               tgt_key_padding_mask: torch.Tensor = None,
               memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        将目标序列和编码器 memory 通过解码器堆栈。

        参数：
            tgt: 目标序列张量（batch_size, tgt_seq_len）。
            memory: 编码器输出张量（batch_size, src_seq_len, d_model）。
            tgt_mask: 目标序列自注意力掩码（tgt_seq_len, tgt_seq_len）。
                      防止注意力集中于未来位置。
            memory_mask: 编码器-解码器注意力掩码（tgt_seq_len, src_seq_len）。
                         通常不需要，除非需要特定的交叉注意力掩码。
            tgt_key_padding_mask: 目标序列中填充标记的掩码（batch_size, tgt_seq_len）。
            memory_key_padding_mask: 源序列中填充标记的掩码，用于编码器-解码器注意力（batch_size, src_seq_len）。

        返回：
            解码器输出张量（batch_size, tgt_seq_len, d_model）。
        """
        # 嵌入标记并添加位置编码
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # 通过每个解码器层
        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        return output

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Transformer 模型的完整前向传播。

        参数：
            src: 源序列张量（batch_size, src_seq_len）。
            tgt: 目标序列张量（batch_size, tgt_seq_len）。
            src_mask: 源序列注意力掩码。
            tgt_mask: 目标序列自注意力掩码。
            memory_mask: 编码器-解码器注意力掩码。
            src_key_padding_mask: 源序列的填充掩码。
            tgt_key_padding_mask: 目标序列的填充掩码。
            memory_key_padding_mask: 用于交叉注意力的源序列填充掩码。

        返回：
            输出对数张量（batch_size, tgt_seq_len, tgt_vocab_size）。
        """
        memory = self.encode(src, src_mask, src_key_padding_mask)
        decoder_output = self.decode(tgt, memory, tgt_mask, memory_mask,
                                     tgt_key_padding_mask,
                                     memory_key_padding_mask)
        logits = self.generator(decoder_output)
        return logits