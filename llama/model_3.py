# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear


def edl_reg(evi, q, y, k):
    alpha = evi * (q / (1 - q)) * k + 1
    # dirichlet parameters after removal of non-misleading evidence (from the label)
    alpha = y + (1 - y) * alpha

    # uniform dirichlet distribution
    beta = torch.ones_like(alpha)

    sum_alpha = (q / (1 - q)) * k + k
    sum_beta = beta.sum(-1)

    t1 = sum_alpha.lgamma() - sum_beta.lgamma()
    t2 = (alpha.lgamma() - beta.lgamma()).sum(-1)
    t3 = alpha - beta
    t4 = alpha.digamma() - sum_alpha.digamma().unsqueeze(-1)

    kl = t1 - t2 + (t3 * t4).sum(-1)
    return kl


def edl_mse(evi, q, y, k):
    u = 1 / (1 + q / (1 - q))
    p = (evi * (q / (1 - q)) + 1 / k) / (1 + q / (1 - q))
    t1 = (y - p).pow(2).sum(-1)
    t2 = ((p * (1 - p)) / ((1 + q / (1 - q)) + 1 / k)).sum(-1)
    mse = t1 + t2 / k
    return mse, u


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = True

    max_batch_size: int = 32
    max_seq_len: int = 2048
    adapter_len: int=10
    adapter_layer: int=30
    precision: str='bf16'


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.max_feats = args.max_feats
        self.precision = args.precision

        self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)).cuda()
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)).cuda()
        self.gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, video_start=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_k = self.wk(adapter).view(1, adapter_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = self.wv(adapter).view(1, adapter_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            xk = torch.cat([adapter_k, xk], dim=1)
            xv = torch.cat([adapter_v, xv], dim=1)
            extra_mask = torch.zeros(1, 1, seqlen, adapter_len).to(mask)
            mask = torch.cat([extra_mask, mask], dim=-1)
        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        if adapter is not None:            
            adapter_scores = (F.softmax(scores[..., :adapter_len].float(), dim=-1) * self.gate.tanh()).type_as(xq)
            if video_start is not None:
                vt_scores = scores[..., adapter_len:].clone()
                vt_scores = F.softmax(vt_scores.float(), dim=-1).type_as(xq)
            else:
                vt_scores = F.softmax(scores[..., adapter_len:], dim=-1)
            scores = torch.cat([adapter_scores, vt_scores], dim=-1)
        else:
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, video_start=None):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter, video_start)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, args, tokenizer):
        super().__init__()
        params.max_feats = args.max_feats
        params.bias = args.bias
        self.args = args
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.max_feats = args.max_feats

        self.tok_embeddings = Embedding(params.vocab_size, params.dim)

        self.adapter_query = Embedding(params.adapter_len * params.adapter_layer, params.dim)

        self.visual_proj = Linear(768, params.dim, bias=False)
        self.temporal_emb = Embedding(self.max_feats, params.dim)
        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

        self.train_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.infer_criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        self.sd_criterion = nn.KLDivLoss(reduction='batchmean')

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)

        self.video_label = torch.arange(1, self.max_feats)
        self.tau = args.tau

        if self.args.edl_mse:
            self.scale_proj = Linear(params.vocab_size, 1, bias=True) # 如果没有bias呢？

        self.tokenizer = tokenizer
        
    def edl(self, output, label, label_mask):
        output = output[label_mask.bool(), :]
        label = label[
            label_mask.bool()
        ]
        k = output.shape[-1]
        y = F.one_hot(label, k)

        p = torch.sigmoid(self.scale_proj(output)).to(torch.float32)
        evi = F.softmax(output, dim=-1).to(torch.float32)

        y = y.to(torch.float32)
        mse_loss, u = edl_mse(evi, p, y, k)
        mse_loss = mse_loss.mean()
        reg_loss = edl_reg(evi, p, y, k)
        reg_loss = reg_loss.mean()
        return mse_loss, reg_loss, u

    def forward(self, data, inference=False):
        loss = {}

        # adapter
        adapter = self.adapter_query.weight.reshape(
            -1, self.adapter_len, self.params.dim
        ).unsqueeze(1)

        ######## main-vqa ########
        vqa_id = data["text_id"]["vqa"].cuda()
        vqa_label = data["label"]["vqa"].cuda()
        vqa_video_start = data["video_start"]["vqa"][0]

        bsz, n_options, seqlen = vqa_id.shape

        # video
        video = data["video"].cuda()

        _video_feature = self.visual_proj(video)
        if inference:
            _video_feature = (
                _video_feature.unsqueeze(1)
                .repeat(1, n_options, 1, 1)
                .view(-1, _video_feature.shape[-2], _video_feature.shape[-1])
            )
        video_feature = (_video_feature + self.temporal_emb.weight[None, :, :]).half()

        vqa_id = vqa_id.reshape(-1, seqlen)
        vqa_label = vqa_label.reshape(-1, seqlen)
        vqa_label = vqa_label[:, 1:].flatten()

        vaq_id = data["text_id"]["vaq"].cuda()
        vaq_label = data["label"]["vaq"].cuda()
        vaq_video_start = data["video_start"]["vaq"][0]

        vaq_id = vaq_id.reshape(-1, seqlen)
        vaq_label = vaq_label.reshape(-1, seqlen)
        vaq_label = vaq_label[:, 1:].flatten()

        with torch.no_grad():
            vqa_h = self.tok_embeddings(vqa_id)
            vaq_h = self.tok_embeddings(vaq_id)

        freqs_cis = self.freqs_cis.to(vqa_h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=vqa_h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(vqa_h)
        start_pos = 0

        vqa_h = vqa_h.clone()
        vqa_h[:, vqa_video_start : vqa_video_start + self.max_feats] = video_feature

        for i, layer in enumerate(self.layers[-1 * self.adapter_layer :]):
            vqa_h = layer(
                vqa_h, start_pos, freqs_cis, mask, adapter[i].half(), vqa_video_start
            )

        vqa_h = self.norm(vqa_h)
        vqa_output = self.output(vqa_h)
        vqa_output = vqa_output[:, :-1, :].reshape(-1, self.vocab_size)
        if self.args.edl_mse:
            # edl
            label_mask = data["label_mask"]["vqa"][:, :, 1:].flatten().cuda()  # [b*127]
            vqa_mse_loss, vqa_reg_loss, u = self.edl(vqa_output, vqa_label, label_mask)
            loss["edl_mse"] = vqa_mse_loss * self.args.edl_mse
            loss["edl_reg"] = vqa_reg_loss * self.args.edl_reg
        else:
            loss["vqa"] = self.train_criterion(vqa_output, vqa_label)

        ######## vaq ########
        if (self.args.vaq or self.args.aqa) and not inference:

            vaq_h = vaq_h.clone()
            vaq_h[:, vaq_video_start : vaq_video_start + self.max_feats] = video_feature

            for i, layer in enumerate(self.layers[-1 * self.adapter_layer :]):
                vaq_h = layer(
                    vaq_h,
                    start_pos,
                    freqs_cis,
                    mask,
                    adapter[i].half(),
                    vaq_video_start,
                )

            if self.args.aqa:
                vaq_h_ = vaq_h.clone()

            vaq_h = self.norm(vaq_h)
            vaq_output = self.output(vaq_h)
            vaq_output = vaq_output[:, :-1, :].reshape(-1, self.vocab_size)
            loss["vaq"] = self.train_criterion(vaq_output, vaq_label) * self.args.vaq

        ######## aqa ########
        if self.args.aqa and not inference:

            vqa_question_start = data["question_start"]["vqa"][0]
            vaq_prefix_index = data["prefix_index"]["vaq"]
            qlen = data["qlen"]["vqa"]

            with torch.no_grad():
                aqa_h = self.tok_embeddings(vqa_id)

            aqa_h = aqa_h.clone()
            aqa_h[:, vqa_video_start : vqa_video_start + self.max_feats] = video_feature

            vaq_h_ = self.norm(vaq_h_)
            vaq_output_ = self.output(vaq_h_)

            try:
                for i in range(bsz):
                    aqa_h[i, vqa_question_start : vqa_question_start + qlen[i]] = (
                        torch.matmul(
                            F.gumbel_softmax(
                                vaq_output_[
                                    i,
                                    vaq_prefix_index[i] : vaq_prefix_index[i] + qlen[i],
                                ],
                                tau=1,
                                hard=True,
                            ),
                            self.tok_embeddings.weight,
                        )
                    )
            except:
                print("max length overflow")

            for i, layer in enumerate(self.layers[-1 * self.adapter_layer :]):
                aqa_h = layer(
                    aqa_h,
                    start_pos,
                    freqs_cis,
                    mask,
                    adapter[i].half(),
                    vqa_video_start,
                )

            aqa_h = self.norm(aqa_h)
            aqa_output = self.output(aqa_h)

            if self.args.edl_mse and self.args.aqa_gate:
                _, _, u = self.edl(aqa_output, vqa_label, label_mask)
                vqa_label = vqa_label.reshape(bsz, -1)
                loss["aqa"] = torch.tensor(0)
                u = u.reshape(bsz, -1).mean(-1)
                w = 1 - u
                for i in range(bsz):
                    loss["aqa"] = (
                        loss["aqa"]
                        + self.train_criterion(aqa_output[i][:-1], vqa_label[i]) * w[i]
                    )
                loss["aqa"] = loss["aqa"] * self.args.aqa
            else:
                aqa_output = aqa_output[:, :-1, :].reshape(-1, self.vocab_size)
                loss["aqa"] = self.train_criterion(aqa_output, vqa_label) * self.args.aqa

        # infer & return
        if inference:
            if self.args.edl_mse:
                label_mask = (
                    data["label_mask"]["vqa"][:, :, 1:].flatten().cuda()
                )
                _, _, u = self.edl(vqa_output, vqa_label, label_mask)
                try:
                    u = u.reshape(bsz, n_options, -1).mean(-1)
                except:
                    u = torch.zeros(bsz, n_options).cuda()
            else:
                u = torch.zeros(bsz, n_options).cuda()
            logits = self.infer_criterion(vqa_output, vqa_label)
            logits = logits.reshape(bsz, n_options, -1)
            return logits, u
        else:
            return loss
