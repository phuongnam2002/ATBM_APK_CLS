import torch
import math as mt
import torch.nn as nn
from transformers import PreTrainedTokenizer
from components.models.module import MLPLayer


class Position_Encoding(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.encoding = torch.zeros(args.max_seq_len, args.d_model, requires_grad=False, device=args.device)

        self.pos = torch.arange(0, args.max_seq_len, device=args.device)  # (seq_len,)
        self.pos = self.pos.float().unsqueeze(dim=1)  # (seq_len, 1)

        indexs_odd = torch.arange(1, args.d_model, step=2, device=args.device).float()
        indexs_even = torch.arange(0, args.d_model, step=2, device=args.device).float()

        self.encoding[:, 0::2] = torch.sin(self.pos / (10000 ** (indexs_even / args.d_model)))
        self.encoding[:, 1::2] = torch.cos(self.pos / (10000 ** (indexs_odd / args.d_model)))

    def forward(self, x):
        seq_len = x.size(1)
        return self.encoding[:seq_len, :]  # (batch_size, seq_len, d_model)


class Embedding(nn.Embedding):
    def __init__(self, args, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer.vocab_size, args.d_model, padding_idx=tokenizer.pad_token_id)


class TransformerEmbedding(nn.Module):
    def __init__(self, args, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.args = args
        self.pe = Position_Encoding(args=args)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.embedding = Embedding(args=args, tokenizer=tokenizer)

    def forward(self, x):
        x_pe = self.pe(x)  # (batch_size, seq_len, d_model)
        x_embed = self.embedding(x).to(self.args.device)  # (batch_size, seq_len, d_model)

        x = x_embed + x_pe

        return x  # (batch_size, seq_len, d_model)


class Norm(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gamma = nn.Parameter(torch.ones(args.d_model))
        self.beta = nn.Parameter(torch.zeros(args.d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        x = (x - mean) / torch.sqrt(var + self.args.eps)
        x = self.gamma * x + self.beta

        return x  # (batch_size, seq_len, d_model)


class ScaleDotProductAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch, head, seq_len, d_model = q.size()

        k_t = k.transpose(2, 3)  # batch, head, d_model, seq_len
        score = (q @ k_t) / mt.sqrt(d_model)  # batch, head, seq_len, seq_len

        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))

        score = self.softmax(score)

        v = score @ v  # batch, head, seq_len, d_model

        return v, score


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.attention = ScaleDotProductAttention(args=args)
        self.w_q = nn.Linear(args.d_model, args.d_model)
        self.w_k = nn.Linear(args.d_model, args.d_model)
        self.w_v = nn.Linear(args.d_model, args.d_model)
        self.w_concat = nn.Linear(args.d_model, args.d_model)

    def forward(self, q, k, v, mask=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = self.split(q)
        k = self.split(k)
        v = self.split(v)

        outputs, attentions = self.attention(q, k, v, mask=mask)

        outputs = self.concat(outputs)
        outputs = self.w_concat(outputs)

        return outputs

    def split(self, x):
        batch, seq_len, d_model = x.size()

        assert d_model % self.args.num_heads == 0
        d_tensor = d_model // self.args.num_heads

        x = x.view(batch, seq_len, self.args.num_heads, d_tensor).transpose(1, 2)

        return x

    def concat(self, x):
        batch, num_heads, seq_len, d_tensor = x.size()

        x = x.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return x


class PointWiseFeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.linear1 = nn.Linear(args.d_model, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, args.d_model)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = MultiHeadAttention(args=args)
        self.norm1 = Norm(args=args)
        self.dropout1 = nn.Dropout(p=args.dropout_rate)

        self.ffn = PointWiseFeedForward(args=args)
        self.norm2 = Norm(args=args)
        self.dropout2 = nn.Dropout(p=args.dropout_rate)

    def forward(self, x, s_mask):
        residual = x
        x = self.attention(q=x, k=x, v=x, mask=s_mask)

        x = self.dropout1(x)
        x = self.norm1(residual + x)

        residual = x
        x = self.ffn(x)

        x = self.dropout2(x)
        x = self.norm2(x + residual)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention1 = MultiHeadAttention(args=args)
        self.norm1 = Norm(args=args)
        self.dropout1 = nn.Dropout(p=args.dropout_rate)

        self.attention2 = MultiHeadAttention(args=args)
        self.norm2 = Norm(args=args)
        self.dropout2 = nn.Dropout(p=args.dropout_rate)

        self.ffn = PointWiseFeedForward(args=args)
        self.norm3 = Norm(args=args)
        self.dropout3 = nn.Dropout(p=args.dropout_rate)

    def forward(self, dec, enc, t_mask, s_mask):
        residual = dec
        x = self.attention1(q=dec, k=dec, v=dec, mask=t_mask)

        x = self.dropout1(x)
        x = self.norm1(residual + x)

        if enc is not None:
            residual = x
            x = self.attention2(q=x, k=enc, v=enc, mask=s_mask)

            x = self.dropout2(x)
            x = self.norm2(x + residual)

        residual = x
        x = self.ffn(x)

        x = self.dropout3(x)
        x = self.norm3(x + residual)

        return x  # batch_size, seq_len, d_model


class Encoder(nn.Module):
    def __init__(self, args, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.args = args
        self.embed = TransformerEmbedding(args=args, tokenizer=tokenizer)

        self.layers = nn.ModuleList([EncoderLayer(args=args) for i in range(args.num_layers)])

    def forward(self, x, s_mask):
        x = self.embed(x)

        for layer in self.layers:
            x = layer(x, s_mask=s_mask)

        return x  # batch_size, seq_len, d_model


class Decoder(nn.Module):
    def __init__(self, args, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.args = args
        self.embed = TransformerEmbedding(args=args, tokenizer=tokenizer)

        self.layers = nn.ModuleList([DecoderLayer(args=args) for i in range(args.num_layers)])

        self.linear = nn.Linear(args.d_model, tokenizer.vocab_size)

    def forward(self, x, enc, t_mask, s_mask):
        x = self.embed(x)

        for layer in self.layers:
            x = layer(x, enc, t_mask, s_mask)

        outputs = self.linear(x)

        return outputs


class Transformers(nn.Module):
    def __init__(self, args, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.mlp = MLPLayer(args.hidden_size)
        self.loss = nn.BCEWithLogitsLoss()
        self.encoder = Encoder(args=args, tokenizer=tokenizer)
        self.decoder = Decoder(args=args, tokenizer=tokenizer)

    def forward(self, src, trg, labels, train=True):
        src_mask = self.make_pad_mask(src, src)

        src_trg_mask = self.make_pad_mask(trg, src)

        trg_mask = self.make_pad_mask(trg, trg) * self.no_peak_mask(trg)

        enc_src = self.encoder(src, src_mask)

        outputs = self.decoder(trg, enc_src, trg_mask, src_trg_mask)

        outputs = self.mlp(outputs)

        if train == False:
            return outputs

        outputs = torch.gather(outputs, 1, labels)

        loss = self.loss(outputs, labels.float())

        return loss

    def make_pad_mask(self, q, k):
        seq_len = q.size(1)

        k = k.ne(self.tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)  # batch_size, 1, 1, seq_len
        k = k.repeat(1, 1, seq_len, 1)  # batch_size, 1, seq_len, seq_len

        q = q.ne(self.tokenizer.pad_token_id).unsqueeze(1).unsqueeze(3)  # batch_size, 1, seq_len,1
        q = q.repeat(1, 1, 1, seq_len)  # batch_size, 1, seq_len, seq_len

        mask = k & q

        return mask

    def no_peak_mask(self, q):
        seq_len = q.size(1)

        mask = torch.tril(torch.ones(seq_len, seq_len)).type(torch.BoolTensor).to(self.args.device)

        return mask
