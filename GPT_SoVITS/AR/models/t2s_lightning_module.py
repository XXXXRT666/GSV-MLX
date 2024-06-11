# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_lightning_module.py
# reference: https://github.com/lifeiteng/vall-e
import os, sys

import mlx.optimizers

now_dir = os.getcwd()
sys.path.append(now_dir)
from typing import Dict, Tuple
import mlx
import torch
from pytorch_lightning import LightningModule
from AR.models.t2s_model import Text2SemanticDecoder
from AR.modules.lr_schedulers import WarmupCosineLRSchedule
from AR.modules.optim import ScaledAdam
from AR.models.utils import dpo_loss
import mlx.nn as nn
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

class Text2SemanticLightningModule(LightningModule):
    def __init__(self, config, output_dir, is_train=True):
        super().__init__()
        self.config = config
        self.top_k = 3
        self.model = Text2SemanticDecoder(config=config, top_k=self.top_k)
        pretrained_s1 = config.get("pretrained_s1")
        if pretrained_s1 and is_train:
            # print(self.load_state_dict(torch.load(pretrained_s1,map_location="cpu")["state_dict"]))
            print(
                self.load_state_dict(
                    torch.load(pretrained_s1, map_location="cpu")["weight"]
                )
            )
        if is_train:
            self.automatic_optimization = False
            self.save_hyperparameters()
            self.eval_dir = output_dir / "eval"
            self.eval_dir.mkdir(parents=True, exist_ok=True)

    def training_step(self, batch: Dict, batch_idx: int):
        opt = self.optimizers()
        self.scheduler = self.lr_schedulers()
        loss_fn = self.loss_fn_dpo if self.config["train"].get("if_dpo",False)==True else self.loss_fn_old
        input_batch = (
            batch["phoneme_ids"],
            batch["phoneme_ids_len"],
            batch["semantic_ids"],
            batch["semantic_ids_len"],
            batch["bert_feature"],
        )
        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        loss, grads = loss_and_grad_fn(self.model, input_batch)

        if batch_idx > 0 and batch_idx % 4 == 0:
            opt.update(self.model, grads)
            mx.eval(self.model.parameters(), opt.state)


    def validation_step(self, batch: Dict, batch_idx: int):
        return

    # # get loss
    # loss, acc = self.model.forward(
    #     batch['phoneme_ids'], batch['phoneme_ids_len'],
    #     batch['semantic_ids'], batch['semantic_ids_len'],
    #     batch['bert_feature']
    # )
    #
    # self.log(
    #     "val_total_loss",
    #     loss,
    #     on_step=True,
    #     on_epoch=True,
    #     prog_bar=True,
    #     sync_dist=True)
    # self.log(
    #     f"val_top_{self.top_k}_acc",
    #     acc,
    #     on_step=True,
    #     on_epoch=True,
    #     prog_bar=True,
    #     sync_dist=True)
    #
    # # get infer output
    # semantic_len = batch['semantic_ids'].size(1)
    # prompt_len = min(int(semantic_len * 0.5), 150)
    # prompt = batch['semantic_ids'][:, :prompt_len]
    # pred_semantic = self.model.infer(batch['phoneme_ids'],
    #                                  batch['phoneme_ids_len'], prompt,
    #                                  batch['bert_feature']
    #                                  )
    # save_name = f'semantic_toks_{batch_idx}.pt'
    # save_path = os.path.join(self.eval_dir, save_name)
    # torch.save(pred_semantic.detach().cpu(), save_path)

    def configure_optimizers(self):
        model_parameters = self.model.trainable_parameters()
        parameters_names = []
        parameters_names.append(
            [name_param_pair[0] for name_param_pair in tree_flatten(self.model.trainable_parameters(),prefix=".self.model")]
        )
        lm_opt = mlx.optimizers.Adamax(
            learning_rate=0.01,
            betas=(0.9, 0.95),
        )

        return {
            "optimizer": lm_opt,
            "lr_scheduler": {
                "scheduler": WarmupCosineLRSchedule(
                    init_lr=self.config["optimizer"]["lr_init"],
                    peak_lr=self.config["optimizer"]["lr"],
                    end_lr=self.config["optimizer"]["lr_end"],
                    warmup_steps=self.config["optimizer"]["warmup_steps"],
                    total_steps=self.config["optimizer"]["decay_steps"],
                )
            },
        }


    def loss_fn_dpo(self, input_batch):
        logits, targets ,A_logits, R_logits=self.model(*input_batch)
        loss_1 = nn.losses.cross_entropy(
            logits.transpose(0, 2, 1), targets, reduction="sum"
            )
        loss_2, _, _ = dpo_loss(
            A_logits, R_logits, 0, 0, 0.2, reference_free=True
            )
        acc = self.acc_dpo(self, logits, targets)
        self.log_(loss_1 + loss_2, acc)
        return loss_1 + loss_2

    def loss_fn_old(self,input_batch):
        logits, targets = self.model.forward_old(*input_batch)
        loss = nn.losses.cross_entropy(
            logits, targets, reduction="sum"
            )
        acc = self.acc_old(self, logits, targets)
        self.log_(loss, acc)   
        return loss

    def acc_dpo(self:Text2SemanticDecoder, logits, targets):
        acc = self.ar_accuracy_metric(
            torch.tensor(np.array(logits.transpose(0, 2, 1))), torch.tensor(np.array(targets))
            ).item()
        return acc

    def acc_old(self:Text2SemanticDecoder, logits, targets):
        acc = self.ar_accuracy_metric(
            torch.tensor(np.array(logits)), torch.tensor(np.array(targets))
            ).item()
        return acc
    
    def log_(self, loss, acc):

        self.log(
                "total_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
        )
        self.log(
                "lr",
                self.scheduler.last_lr()[0],
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
        )
        self.log(
                f"top_{self.top_k}_acc",
                acc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
        )
        return
