# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
import torch
import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm
import numpy as np
from AR.models.utils import make_pad_mask
from AR.models.utils import (
    topk_sampling,
    sample,
    logits_to_probs,
    multinomial_sample_one_no_sync,
    dpo_loss,
    make_reject_y, 
    get_batch_logps
)
from AR.modules.embedding import SinePositionalEmbedding
from AR.modules.embedding import TokenEmbedding
from AR.modules.transformer import LayerNorm
from AR.modules.transformer import TransformerEncoder
from AR.modules.transformer import TransformerEncoderLayer
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy
from mlx.utils import tree_flatten
default_config = {"model":{
    "embedding_dim": 512,
    "hidden_dim": 512,
    "head": 8,
    "n_layer": 12,
    "num_codebook": 8,
    "dropout": 0.0,
    "vocab_size": 1024 + 1,
    "phoneme_vocab_size": 512,
    "EOS": 1024,
}}


class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=3):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = config["model"]["dropout"]
        self.EOS = config["model"]["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        # should be same as num of kmeans bin
        # assert self.EOS == 1024
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(
            self.embedding_dim, self.phoneme_vocab_size, self.p_dropout
        )
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True
        )
        self.ar_audio_embedding = TokenEmbedding(
            self.embedding_dim, self.vocab_size, self.p_dropout
        )
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True
        )

        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_head,
                dim_feedforward=self.model_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=self.num_layers,
            norm=LayerNorm(self.model_dim) if norm_first else None,
        )

        self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)

        self.ar_accuracy_metric = MulticlassAccuracy(
            self.vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,
        )

    def make_input_data(self, x, x_lens, y, y_lens, bert_feature):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.swapaxes(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.astype(mx.int64)
        codes = y.astype(mx.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max().item()
        y_len = y_lens.max().item()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_padding_mask = mx.concatenate([x_mask, y_mask], axis=1)

        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = mx.pad(
            mx.zeros((x_len, x_len), dtype=mx.bool_),
            [(0, 0), (0, y_len)],
            constant_values=True,
        )
        
        y_attn_mask = mx.pad(
            mx.triu(
                mx.ones((y_len, y_len), dtype=mx.bool_),
                k=1,
            ),
            [(0, 0), (x_len, 0)],
            constant_values=False,
        )

        xy_attn_mask = mx.concatenate([x_attn_mask, y_attn_mask], axis=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            mx.tile(ar_xy_padding_mask.reshape(bsz, 1, 1, src_len)
            ,(1, self.num_head, 1, 1))
            .reshape(bsz * self.num_head, 1, src_len)
        )
        xy_attn_mask = mx.logical_or(xy_attn_mask,_xy_padding_mask)
        new_attn_mask = mx.zeros_like(xy_attn_mask).astype(x.dtype)
        new_attn_mask = mx.where(xy_attn_mask, float("-inf"), new_attn_mask)
        xy_attn_mask = new_attn_mask
        # x 和完整的 y 一次性输入模型
        xy_pos = mx.concatenate([x, y_pos], axis=1)

        return xy_pos, xy_attn_mask, targets

    def __call__(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids
        y: semantic_ids
        """

        reject_y, reject_y_lens = make_reject_y(y, y_lens)

        xy_pos, xy_attn_mask, targets = self.make_input_data(x, x_lens, y, y_lens, bert_feature)

        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        x_len = x_lens.max().item()
        logits = self.ar_predict_layer(xy_dec[:, x_len:])

        ###### DPO #############
        reject_xy_pos, reject_xy_attn_mask, reject_targets = self.make_input_data(x, x_lens, reject_y, reject_y_lens, bert_feature)

        reject_xy_dec, _ = self.h(
            (reject_xy_pos, None),
            mask=reject_xy_attn_mask,
        )
        reject_logits = self.ar_predict_layer(reject_xy_dec[:, x_len:])

        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum

        A_logits, R_logits = get_batch_logps(logits, reject_logits, targets, reject_targets)

        return logits, targets, A_logits, R_logits

    def forward_old(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids
        y: semantic_ids
        """
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.swapaxes(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.astype(mx.int64)
        codes = y.astype(mx.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max().item()
        y_len = y_lens.max().item()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_padding_mask = mx.concatenate([x_mask, y_mask], axis=1)
        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = mx.pad(
            mx.zeros((x_len, x_len), dtype=mx.bool_),
           [(0, 0), (0, y_len)],
            constant_values=True,
        )
        y_attn_mask = mx.pad(
            mx.triu(
                mx.ones((y_len, y_len), dtype=mx.bool_),
                k=1,
            ),
            [(0, 0), (x_len, 0)],
            constant_values=False,
        )
        xy_attn_mask = mx.concatenate([x_attn_mask, y_attn_mask], axis=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            mx.tile(ar_xy_padding_mask.reshape(bsz, 1, 1, src_len)
            , (1, self.num_head, 1, 1))
            .reshape(bsz * self.num_head, 1, src_len)
        )
        xy_attn_mask = mx.logical_or(xy_attn_mask, _xy_padding_mask)
        new_attn_mask = mx.zeros_like(xy_attn_mask).astype(x.dtype)
        new_attn_mask = mx.where(xy_attn_mask, float("-inf"),new_attn_mask)
        xy_attn_mask = new_attn_mask
        # x 和完整的 y 一次性输入模型
        xy_pos = mx.concatenate([x, y_pos], axis=1)
        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        logits = self.ar_predict_layer(xy_dec[:, x_len:]).transpose(0, 2, 1)
        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum
        return logits, targets

    # 需要看下这个函数和 forward 的区别以及没有 semantic 的时候 prompts 输入什么
    def infer(
        self,
        x,
        x_lens,
        prompts,
        bert_feature,
        top_k: int = -100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.swapaxes(1, 2))
        x = self.ar_text_position(x)

        # AR Decoder
        y = prompts
        prefix_len = y.shape[1]
        x_len = x.shape[1]
        x_attn_mask =  mx.zeros((x_len, x_len), dtype=mx.bool_)
        stop = False
        for _ in tqdm(range(1500)):
            y_emb = self.ar_audio_embedding(y)
            y_pos = self.ar_audio_position(y_emb)
            # x 和逐渐增长的 y 一起输入给模型
            xy_pos = mx.concatenate([x, y_pos], axis=1)
            y_len = y.shape[1]
            x_attn_mask_pad = mx.pad(
                x_attn_mask,
                [(0, 0), (0, y_len)],
                constant_values=True,
            )
            y_attn_mask = mx.pad(
                mx.triu(mx.ones((y_len, y_len), dtype=mx.bool_), k=1),
                [(0, 0), (x_len, 0)],
                constant_values=False,
            )
            xy_attn_mask = mx.concatenate([x_attn_mask_pad, y_attn_mask], axis=0)

            xy_dec, _ = self.h(
                (xy_pos, None),
                mask=xy_attn_mask,
            )
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(
                logits, top_k=top_k, top_p=1.0, temperature=temperature
            )

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if mx.argmax(logits, axis=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                # print(torch.argmax(logits, dim=-1)[0] == self.EOS, samples[0, 0] == self.EOS)
                stop = True
            if stop:
                if prompts.shape[1] == y.shape[1]:
                    y = mx.concatenate([y, mx.zeros_like(samples)], axis=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break
            # 本次生成的 semantic_ids 和之前的 y 构成新的 y
            # print(samples.shape)#[1,1]#第一个1是bs
            # import os
            # os._exit(2333)
            y = mx.concatenate([y, samples], axis=1)
        return y

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = mx.pad(y, [(0,0)]*(y.ndim-1)+[(0, 1)], constant_values=0) + eos_id * mx.pad(
            y_mask_int, [(0,0)]*(y_mask_int.ndim-1)+[(0, 1)], constant_values=1
        )
        # 错位
        return targets[:, :-1], targets[:, 1:]

    def infer_panel(
        self,
        x,  #####全部文本token
        x_lens,
        prompts,  ####参考音频token
        bert_feature,
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.swapaxes(1, 2))
        x = self.ar_text_position(x)

        # AR Decoder
        y = prompts
        
        x_len = x.shape[1]
        x_attn_mask = mx.zeros((x_len, x_len), dtype=mx.bool_)
        stop = False
        # print(1111111,self.num_layers)
        cache = {
            "all_stage": self.num_layers,
            "k": [None] * self.num_layers,  ###根据配置自己手写
            "v": [None] * self.num_layers,
            # "xy_pos":None,##y_pos位置编码每次都不一样的没法缓存，每次都要重新拼xy_pos.主要还是写法原因，其实是可以历史统一一样的，但也没啥计算量就不管了
            "y_emb": None,  ##只需要对最新的samples求emb，再拼历史的就行
            # "logits":None,###原版就已经只对结尾求再拼接了，不用管
            # "xy_dec":None,###不需要，本来只需要最后一个做logits
            "first_infer": 1,
            "stage": 0,
        }
        ###################  first step ##########################
        if y is not None:
            y_emb = self.ar_audio_embedding(y)
            y_len = y_emb.shape[1]
            prefix_len = y.shape[1]
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = mx.concatenate([x, y_pos], axis=1)
            cache["y_emb"] = y_emb
            ref_free = False
        else:
            y_emb = None
            y_len = 0
            prefix_len = 0
            y_pos = None
            xy_pos = x
            y = mx.zeros((x.shape[0], 0), dtype=mx.int32)
            ref_free = True

        x_attn_mask_pad = mx.pad(
                    x_attn_mask,
                    [(0, 0), (0, y_len)],  ###xx的纯0扩展到xx纯0+xy纯1，(x,x+y)
                    constant_values=True,
                )
        y_attn_mask = mx.pad(  ###yy的右上1扩展到左边xy的0,(y,x+y)
            mx.triu(mx.ones((y_len, y_len), dtype=mx.bool_), k=1),
            [(0, 0), (x_len, 0)],
            constant_values=False,
        )
        xy_attn_mask = mx.concatenate([x_attn_mask_pad, y_attn_mask], axis=0)
        

        for idx in tqdm(range(1500)):
            
            xy_dec, _ = self.h((xy_pos, None), mask=xy_attn_mask, cache=cache)
            logits = self.ar_predict_layer(
                xy_dec[:, -1]
            )  ##不用改，如果用了cache的默认就是只有一帧，取最后一帧一样的
            # samples = topk_sampling(logits, top_k=top_k, top_p=1.0, temperature=temperature)
            if(idx==0):###第一次跑不能EOS否则没有了
                logits = logits[:, :-1]  ###刨除1024终止符号的概率
            samples = mx.expand_dims(sample(
                logits[0], y, top_k=top_k, top_p=top_p, repetition_penalty=1.35, temperature=temperature
            )[0],0)
            # 本次生成的 semantic_ids 和之前的 y 构成新的 y
            # print(samples.shape)#[1,1]#第一个1是bs
            y = mx.concatenate([y, samples], axis=1) 

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if mx.argmax(logits, axis=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                # print(torch.argmax(logits, dim=-1)[0] == self.EOS, samples[0, 0] == self.EOS)
                stop = True
            if stop:
                # if prompts.shape[1] == y.shape[1]:
                #     y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                #     print("bad zero prediction")
                if y.shape[1]==0:
                    y = mx.concatenate([y, mx.zeros_like(samples)], axis=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break
            
            ####################### update next step ###################################
            cache["first_infer"] = 0
            if cache["y_emb"] is not None:
                y_emb = mx.concatenate(
                    [cache["y_emb"], self.ar_audio_embedding(y[:, -1:])], axis = 1
                )
                cache["y_emb"] = y_emb
                y_pos = self.ar_audio_position(y_emb)
                xy_pos = y_pos[:, -1:]
            else:
                y_emb = self.ar_audio_embedding(y[:, -1:])
                cache["y_emb"] = y_emb
                y_pos = self.ar_audio_position(y_emb)
                xy_pos = y_pos
            y_len = y_pos.shape[1]

            ###最右边一列（是错的）
            # xy_attn_mask=torch.ones((1, x_len+y_len), dtype=torch.bool,device=xy_pos.device)
            # xy_attn_mask[:,-1]=False
            ###最下面一行（是对的）
            xy_attn_mask = mx.zeros(
                (1, x_len + y_len), dtype=mx.bool_
            )
        if ref_free:
            return y[:, :-1], 0
        return y[:, :-1], idx-1
