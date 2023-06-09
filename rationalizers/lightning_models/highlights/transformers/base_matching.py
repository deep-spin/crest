import logging
import os
from itertools import chain
from random import shuffle

import numpy as np
import torch
import torchmetrics
import wandb
from torch import nn
from transformers import AutoModel

from rationalizers import constants
from rationalizers.builders import build_optimizer, build_scheduler
from rationalizers.explainers import available_explainers
from rationalizers.explainers.sparsemap_matching import split_pairs
from rationalizers.lightning_models.highlights.base import BaseRationalizer
from rationalizers.modules.matchings_utils import submul, apply_multiple
from rationalizers.modules.metrics import evaluate_rationale
from rationalizers.modules.sentence_encoders import LSTMEncoder, MaskedAverageEncoder
from rationalizers.utils import (
    get_z_stats, freeze_module, masked_average, get_rationales, unroll, get_html_rationales,
    save_rationales, is_trainable, get_ext_mask
)

shell_logger = logging.getLogger(__name__)


class TransformerBaseMatchingRationalizer(BaseRationalizer):

    def __init__(
        self,
        tokenizer: object,
        nb_classes: int,
        is_multilabel: bool,
        h_params: dict,
    ):
        """
        :param tokenizer (object): torchnlp tokenizer object
        :param nb_classes (int): number of classes used to create the last layer
        :param multilabel (bool): whether the problem is multilabel or not (it depends on the dataset)
        :param h_params (dict): hyperparams dict. See docs for more info.
        """
        super().__init__(tokenizer, nb_classes, is_multilabel, h_params)

        # manual update constants
        constants.update_constants(self.tokenizer)
        self.has_countertfactual_flow = False

        ########################
        # hyperparams
        ########################
        # factual:
        self.ff_gen_arch = h_params.get("gen_arch", "bert-base-multilingual-cased")
        self.ff_gen_emb_requires_grad = h_params.get("gen_emb_requires_grad", False)
        self.ff_gen_encoder_requires_grad = h_params.get("gen_encoder_requires_grad", True)
        self.ff_gen_use_decoder = h_params.get("gen_use_decoder", False)
        self.ff_pred_arch = h_params.get("pred_arch", "bert-base-multilingual-cased")
        self.ff_pred_emb_requires_grad = h_params.get("pred_emb_requires_grad", False)
        self.ff_pred_encoder_requires_grad = h_params.get("pred_encoder_requires_grad", True)
        self.ff_pred_output_requires_grad = h_params.get("pred_output_requires_grad", True)
        self.ff_shared_gen_pred = h_params.get("shared_gen_pred", False)
        self.ff_dropout = h_params.get("dropout", 0.1)
        self.ff_selection_vector = h_params.get("selection_vector", 'zero')
        self.ff_selection_mask = h_params.get("selection_mask", True)
        self.ff_selection_faithfulness = h_params.get("selection_faithfulness", True)
        self.ff_lbda = h_params.get('ff_lbda', 1.0)
        self.ff_faithful = h_params.get('ff_faithful', True)

        # explainer:
        self.explainer_fn = h_params.get("explainer_fn", "attention")
        self.explainer_activation = h_params.get("explainer_activation", "sparsemax")
        self.explainer_pre_mlp = h_params.get("explainer_pre_mlp", True)
        self.explainer_requires_grad = h_params.get("explainer_requires_grad", True)
        self.explainer_mask_token_type_id = h_params.get("explainer_mask_token_type_id", None)
        self.temperature = h_params.get("temperature", 1.0)

        ########################
        # useful vars
        ########################
        self.ff_z = None
        self.ff_z_dist = None
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id

        ########################
        # build factual flow
        ########################
        # generator module
        self.ff_gen_hf = AutoModel.from_pretrained(self.ff_gen_arch)
        self.ff_gen_emb_layer = self.ff_gen_hf.shared if 't5' in self.ff_gen_arch else self.ff_gen_hf.embeddings
        self.ff_gen_encoder = self.ff_gen_hf.encoder
        self.ff_gen_decoder = self.ff_gen_hf.decoder if hasattr(self.ff_gen_hf, 'decoder') else None
        self.ff_gen_hidden_size = self.ff_gen_hf.config.hidden_size

        # explainer
        explainer_cls = available_explainers[self.explainer_fn]
        self.explainer = explainer_cls(h_params, self.ff_gen_hidden_size)
        self.explainer_mlp = nn.Sequential(
            nn.Linear(self.ff_gen_hidden_size, self.ff_gen_hidden_size),
            nn.Tanh(),
        )

        # faithfulness
        if self.ff_faithful:
            self.projection_x1 = nn.Sequential(
                nn.Linear(self.ff_gen_hidden_size, self.ff_gen_hidden_size), nn.ReLU()
            )
            self.projection_x2 = nn.Sequential(
                nn.Linear(self.ff_gen_hidden_size*2, self.ff_gen_hidden_size), nn.ReLU()
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(4 * 2 * self.ff_gen_hidden_size, self.ff_gen_hidden_size), nn.ReLU()
            )

        # predictor module
        self.ff_pred_encoder = LSTMEncoder(self.ff_gen_hidden_size, self.ff_gen_hidden_size, bidirectional=True)
        self.ff_pred_hidden_size = self.ff_gen_hidden_size * 2
        self.ff_pred_emb_layer = self.ff_gen_emb_layer
        self.ff_pred_hf = None
        self.ff_pred_decoder = None

        # predictor output layer
        self.ff_output_layer = nn.Sequential(
            nn.Linear(4 * self.ff_pred_hidden_size, self.ff_pred_hidden_size),
            nn.Tanh(),
            nn.Dropout(self.ff_dropout),
            nn.Linear(self.ff_pred_hidden_size, self.nb_classes),
            nn.Sigmoid() if not self.is_multilabel else nn.LogSoftmax(dim=-1),
        )

        ########################
        # weights details
        ########################
        # initialize params using xavier initialization for weights and zero for biases
        # (weights of these modules might be loaded later)
        self.init_weights(self.explainer_mlp)
        self.init_weights(self.explainer)
        self.init_weights(self.ff_output_layer)

        # freeze embedding layers
        if not self.ff_gen_emb_requires_grad:
            freeze_module(self.ff_gen_emb_layer)
        if not self.ff_pred_emb_requires_grad:
            freeze_module(self.ff_pred_emb_layer)

        # freeze models and set to eval mode to disable dropout
        if not self.ff_gen_encoder_requires_grad:
            freeze_module(self.ff_gen_encoder)
            if self.ff_gen_decoder is not None:
                freeze_module(self.ff_gen_decoder)

        # freeze models and set to eval mode to disable dropout
        if not self.ff_pred_encoder_requires_grad:
            freeze_module(self.ff_pred_encoder)
            if self.ff_pred_decoder is not None:
                freeze_module(self.ff_pred_decoder)

        # freeze output layers
        if not self.ff_pred_output_requires_grad:
            freeze_module(self.ff_output_layer)

        # freeze explainer
        if not self.explainer_requires_grad:
            freeze_module(self.explainer_mlp)
            freeze_module(self.explainer)

        # share generator and predictor for the factual flow
        if self.ff_shared_gen_pred:
            assert self.ff_gen_arch == self.ff_pred_arch
            self.ff_pred_hf = self.ff_gen_hf
            self.ff_pred_hidden_size = self.ff_gen_hidden_size
            self.ff_pred_emb_layer = self.ff_gen_emb_layer
            self.ff_pred_encoder = self.ff_gen_encoder
            self.ff_pred_decoder = self.ff_gen_decoder

        ########################
        # loss functions
        ########################
        criterion_cls = nn.MSELoss if not self.is_multilabel else nn.NLLLoss
        self.ff_criterion = criterion_cls(reduction="none")

        ########################
        # logging
        ########################
        # manual check requires_grad for all modules
        for name, module in self.named_children():
            shell_logger.info('is_trainable({}): {}'.format(name, is_trainable(module)))

    def configure_optimizers(self):
        """Configure optimizers and lr schedulers for Trainer."""
        ff_params = chain(
            self.ff_gen_emb_layer.parameters(),
            self.ff_gen_encoder.parameters(),
            self.ff_gen_decoder.parameters() if self.ff_gen_decoder is not None else [],
            self.ff_pred_emb_layer.parameters() if not self.ff_shared_gen_pred else [],
            self.ff_pred_encoder.parameters() if not self.ff_shared_gen_pred else [],
            self.ff_output_layer.parameters(),
            self.explainer_mlp.parameters(),
            self.explainer.parameters(),
        )
        parameters = [{"params": ff_params, 'lr': self.hparams['lr']}]
        optimizer = build_optimizer(parameters, self.hparams)
        scheduler = build_scheduler(optimizer, self.hparams)
        output = {"optimizer": optimizer}
        if scheduler is not None:
            output["scheduler"] = scheduler
            output["monitor"] = self.hparams['monitor']  # not sure we need this
        return output

    def forward(
        self,
        x: torch.LongTensor,
        mask: torch.BoolTensor = None,
        token_type_ids: torch.LongTensor = None,
        current_epoch=None,
    ):
        """
        Compute forward-pass.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param token_type_ids: mask tensor for explanation positions. torch.LongTensor of shape [B, T]
        :param current_epoch: int represents the current epoch.
        :return: (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat)
        """
        # factual flow
        z, y_hat = self.get_factual_flow(x, mask=mask, token_type_ids=token_type_ids)
        # return everything as output (useful for computing the loss)
        return z, y_hat

    def get_factual_flow(self, x, mask=None, token_type_ids=None, z=None):
        """
        Compute the factual flow.

        :param x: input ids tensor. torch.LongTensor of shape [B, T] or input vectors of shape [B, T, |V|]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param token_type_ids: mask tensor for explanation positions. torch.BoolTensor of shape [B, T]
        :param z: precomputed latent vector. torch.FloatTensor of shape [B, T] (default None)
        :return: z, y_hat
        """
        # get the embeddings of the generator
        if x.dim() == 2:
            # receive input ids
            gen_e = self.ff_gen_emb_layer(x)
        else:
            # receive one-hot vectors
            inputs_embeds = x @ self.ff_gen_emb_layer.word_embeddings.weight
            gen_e = self.ff_gen_emb_layer(inputs_embeds=inputs_embeds)

        # pass though the generator encoder
        if 't5' in self.ff_gen_arch:
            # t5 encoder and decoder receives word_emb and a raw mask
            gen_h = self.ff_gen_encoder(
                inputs_embeds=gen_e,
                attention_mask=mask
            )
            if self.ff_gen_use_decoder:
                gen_h = self.ff_gen_decoder(
                    inputs_embeds=gen_e,
                    attention_mask=mask,
                    encoder_hidden_states=gen_h.last_hidden_state,
                )
        else:
            # bert encoder receives word_emb + pos_embs and an extended mask
            ext_mask = get_ext_mask(mask)
            gen_h = self.ff_gen_encoder(
                hidden_states=gen_e,
                attention_mask=ext_mask
            )

        # get final hidden states
        # selected_layers = list(map(int, self.selected_layers.split(',')))
        # gen_h = torch.stack(gen_h)[selected_layers].mean(dim=0)
        gen_h = gen_h.last_hidden_state

        # pass through the explainer
        gen_h = self.explainer_mlp(gen_h) if self.explainer_pre_mlp else gen_h

        # split into two inputs
        token_type_ids = 1 - torch.cumprod(~torch.eq(x, self.eos_token_id), dim=-1)
        token_type_ids = token_type_ids.masked_fill(~mask, 2)
        gen_h1, gen_h2 = split_pairs(gen_h, token_type_ids, padding_value=self.pad_token_id)
        mask_h1, mask_h2 = split_pairs(mask, token_type_ids, padding_value=False)
        z, z_dist = self.explainer(gen_h1, gen_h2, mask_h1, mask_h2) if z is None else z

        # save vars (useful for some methods, e.g., hardkuma and info bottleneck)
        self.ff_z = z
        self.ff_z_dist = z_dist

        x1_align = torch.matmul(z, gen_h2)
        x2_align = torch.matmul(z.transpose(-1, -2), gen_h1)

        if self.ff_faithful:
            x1_combined = x1_align
            x2_combined = torch.cat([gen_h2, x2_align], -1)
            x1_combined = self.projection_x1(x1_combined)
            x2_combined = self.projection_x2(x2_combined)

        else:
            x1_combined = torch.cat([gen_h1, x1_align, submul(gen_h1, x1_align)], -1)
            x2_combined = torch.cat([gen_h2, x2_align, submul(gen_h2, x2_align)], -1)
            x1_combined = self.projection(x1_combined)
            x2_combined = self.projection(x2_combined)

        x1_compose, _ = self.ff_pred_encoder(x1_combined, mask_h1)
        x2_compose, _ = self.ff_pred_encoder(x2_combined, mask_h1)
        x1_rep = apply_multiple(x1_compose)
        x2_rep = apply_multiple(x2_compose)
        summary = torch.cat([x1_rep, x2_rep], -1)

        # pass through the final mlp
        y_hat = self.ff_output_layer(summary)

        return z, y_hat

    def get_factual_loss(self, y_hat, y, z, mask, prefix):
        """
        Compute loss for the factual flow.

        :param y_hat: predictions from SentimentPredictor. Torch.Tensor of shape [B, C]
        :param y: tensor with gold labels. torch.BoolTensor of shape [B]
        :param z: latent selection vector. torch.FloatTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param prefix: prefix for loss statistics (train, val, test)
        :return: tuple containing:
            `loss cost (torch.FloatTensor)`: the result of the loss function
            `loss stats (dict): dict with loss statistics
        """
        stats = {}
        loss_vec = self.ff_criterion(y_hat, y)  # [B] or [B,C]

        # masked average
        if loss_vec.dim() == 2:
            loss = (loss_vec * mask.float()).sum(-1) / mask.sum(-1).float()  # [1]
        else:
            loss = loss_vec.mean()

        # main loss for p(y | x, z)
        stats["mse" if not self.is_multilabel else "criterion"] = loss.item()

        # latent selection stats
        num_0, num_c, num_1, total = get_z_stats(z, mask)
        stats[prefix + "_p0"] = num_0 / float(total)
        stats[prefix + "_pc"] = num_c / float(total)
        stats[prefix + "_p1"] = num_1 / float(total)
        stats[prefix + "_ps"] = (num_c + num_1) / float(total)
        stats[prefix + "_main_loss"] = loss.item()

        return loss, stats

    def training_step(self, batch: dict, batch_idx: int):
        """
        Compute forward-pass, calculate loss and log metrics.

        :param batch: The dict output from the data module with the following items:
            `input_ids`: torch.LongTensor of shape [B, T],
            `lengths`: torch.LongTensor of shape [B]
            `labels`: torch.LongTensor of shape [B, C]
            `tokens`: list of strings
        :param batch_idx: integer displaying index of this batch
        :return: pytorch_lightning.Result log object
        """
        input_ids = batch["input_ids"]
        mask = input_ids != constants.PAD_ID
        labels = batch["labels"]
        token_type_ids = batch.get("token_type_ids", None)
        prefix = "train"

        z, y_hat = self(input_ids, mask, token_type_ids=token_type_ids, current_epoch=self.current_epoch)

        # compute factual loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        loss, loss_stats = self.get_factual_loss(y_hat, y, z, mask, prefix=prefix)

        # logger=False because they are going to be logged via loss_stats
        self.log("train_ff_ps", loss_stats["train_ps"], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("train_ff_p1", loss_stats["train_p1"], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("train_sum_loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)

        metrics_to_wandb = {
            "train_ff_p1": loss_stats["train_p1"],
            "train_ff_ps": loss_stats["train_ps"],
            "train_ff_sum_loss": loss.item(),
        }
        if "cost_g" in loss_stats:
            metrics_to_wandb["train_cost_g"] = loss_stats["cost_g"]

        self.logger.log_metrics(metrics_to_wandb, self.global_step)

        # return the loss tensor to PTL
        return {"loss": loss, "ps": loss_stats["train_ps"], "p1": loss_stats["train_p1"]}

    def _shared_eval_step(self, batch: dict, batch_idx: int, prefix: str):
        input_ids = batch["input_ids"]
        mask = input_ids != constants.PAD_ID
        token_type_ids = batch.get("token_type_ids", None)
        labels = batch["labels"]

        # forward-pass
        z, y_hat = self(input_ids, mask, token_type_ids=token_type_ids, current_epoch=self.current_epoch)

        # compute factual loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        loss, ff_loss_stats = self.get_factual_loss(y_hat, y, z, mask, prefix=prefix)
        self.logger.agg_and_log_metrics(ff_loss_stats, step=None)

        # log metrics
        self.log(f"{prefix}_sum_loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)

        # get factual rationales
        z_1 = (z > 0).long()  # non-zero probs are considered selections
        ids_rationales, rationales = get_rationales(self.tokenizer, input_ids, z_1, batch["lengths"])
        pieces = [self.tokenizer.convert_ids_to_tokens(idxs) for idxs in input_ids.tolist()]

        # output to be stacked across iterations
        output = {
            f"{prefix}_sum_loss": loss.item(),
            f"{prefix}_ps": ff_loss_stats[prefix + "_ps"],
            f"{prefix}_p1": ff_loss_stats[prefix + "_p1"],
            f"{prefix}_ids_rationales": ids_rationales,
            f"{prefix}_rationales": rationales,
            f"{prefix}_pieces": pieces,
            f"{prefix}_z": z,
            f"{prefix}_predictions": y_hat,
            f"{prefix}_labels": y.tolist(),
            f"{prefix}_lengths": batch["lengths"].tolist(),
        }

        if "annotations" in batch.keys():
            output[f"{prefix}_annotations"] = batch["annotations"]

        if "mse" in ff_loss_stats.keys():
            output[f"{prefix}_mse"] = ff_loss_stats["mse"]

        if "is_original" in batch.keys():
            output[f"{prefix}_is_original"] = batch["is_original"]

        return output

    def _shared_eval_epoch_end(self, outputs: list, prefix: str):
        """
        PTL hook. Perform validation at the end of an epoch.

        :param outputs: list of dicts representing the stacked outputs from validation_step
        :param prefix: `val` or `test`
        """

        def shared_helper(stacked_outputs, flow='ff'):
            # sample a few examples to be logged in wandb
            idxs = list(range(sum(map(len, stacked_outputs[f"{prefix}_pieces"]))))
            shuffle(idxs)
            idxs = idxs[:10] if prefix != 'test' else idxs[:100]

            # useful functions
            select = lambda v: [v[i] for i in idxs]
            detach = lambda v: [v[i].detach().cpu() for i in range(len(v))]

            # log rationales
            if self.log_rationales_in_wandb:
                pieces = select(unroll(stacked_outputs[f"{prefix}_pieces"]))
                scores = detach(select(unroll(stacked_outputs[f"{prefix}_z"])))
                gold = select(unroll(stacked_outputs[f"{prefix}_labels"]))
                pred = detach(select(unroll(stacked_outputs[f"{prefix}_predictions"])))
                lens = select(unroll(stacked_outputs[f"{prefix}_lengths"]))
                html_string = get_html_rationales(pieces, scores, gold, pred, lens)
                self.logger.experiment.log({f"{prefix}_rationales": wandb.Html(html_string)})

            # save rationales
            if self.hparams.save_rationales:
                scores = detach(unroll(stacked_outputs[f"{prefix}_z"]))
                lens = unroll(stacked_outputs[f"{prefix}_lengths"])
                filename = os.path.join(self.hparams.default_root_dir, f'{prefix}_rationales.txt')
                shell_logger.info(f'Saving rationales in {filename}...')
                save_rationales(filename, scores, lens)

            # log metrics
            dict_metrics = {
                f"{prefix}_{flow}_ps": np.mean(stacked_outputs[f"{prefix}_ps"]),
                f"{prefix}_{flow}_p1": np.mean(stacked_outputs[f"{prefix}_p1"]),
                f"{prefix}_{flow}_sum_loss": np.mean(stacked_outputs[f"{prefix}_sum_loss"]),
            }

            # only evaluate rationales on the test set and if we have annotation (only for beer dataset)
            if prefix == "test" and "test_annotations" in stacked_outputs.keys():
                rat_metrics = evaluate_rationale(
                    stacked_outputs["test_ids_rationales"],
                    stacked_outputs["test_annotations"],
                    stacked_outputs["test_lengths"],
                )
                dict_metrics[f"{prefix}_{flow}_rat_precision"] = rat_metrics["macro_precision"]
                dict_metrics[f"{prefix}_{flow}_rat_recall"] = rat_metrics["macro_recall"]
                dict_metrics[f"{prefix}_{flow}_rat_f1"] = rat_metrics["f1_score"]

            # log classification metrics
            if self.is_multilabel:
                preds = torch.argmax(torch.cat(stacked_outputs[f"{prefix}_predictions"]), dim=-1)
                labels = torch.tensor(unroll(stacked_outputs[f"{prefix}_labels"]), device=preds.device)
                accuracy = torchmetrics.functional.accuracy(
                    preds, labels, num_classes=self.nb_classes, average="macro"
                )
                precision = torchmetrics.functional.precision(
                    preds, labels, num_classes=self.nb_classes, average="macro"
                )
                recall = torchmetrics.functional.recall(
                    preds, labels, num_classes=self.nb_classes, average="macro"
                )
                f1_score = 2 * precision * recall / (precision + recall)
                dict_metrics[f"{prefix}_{flow}_accuracy"] = accuracy
                dict_metrics[f"{prefix}_{flow}_precision"] = precision
                dict_metrics[f"{prefix}_{flow}_recall"] = recall
                dict_metrics[f"{prefix}_{flow}_f1score"] = f1_score
            else:
                dict_metrics[f"{prefix}_{flow}_mse"] = np.mean(stacked_outputs[f"{prefix}_mse"])

            # log all saved metrics
            for metric_name, metric_value in dict_metrics.items():
                shell_logger.info("{}: {:.4f}".format(metric_name, metric_value))
                self.log(
                    metric_name,
                    metric_value,
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                )

            # aggregate across epochs
            self.logger.agg_and_log_metrics(dict_metrics, self.current_epoch)

            return dict_metrics

        def split_outputs_by_flows(outputs):
            # assume that `outputs` is a list containing dicts with the same keys
            # stacked_outputs = {k: [x[k] for x in outputs] for k in outputs[0].keys()}
            stacked_outputs_ff = {k: [] for k in outputs[0].keys()}
            stacked_outputs_cf = {k: [] for k in outputs[0].keys()}
            has_cf = False
            for x in outputs:
                for k in x.keys():
                    if f'{prefix}_is_original' not in x.keys():
                        stacked_outputs_ff[k].append(x[k])
                    else:
                        ff_idxs = [i for i, is_original in enumerate(x[f'{prefix}_is_original']) if is_original]
                        cf_idxs = [i for i, is_original in enumerate(x[f'{prefix}_is_original']) if not is_original]
                        has_cf = len(cf_idxs) > 0 or has_cf
                        if isinstance(x[k], torch.Tensor):
                            if len(ff_idxs) > 0:
                                stacked_outputs_ff[k].append(torch.stack([x[k][i] for i in ff_idxs]))
                            if len(cf_idxs) > 0:
                                stacked_outputs_cf[k].append(torch.stack([x[k][i] for i in cf_idxs]))
                        elif isinstance(x[k], (tuple, list)):
                            if len(ff_idxs) > 0:
                                stacked_outputs_ff[k].append([x[k][i] for i in ff_idxs])
                            if len(cf_idxs) > 0:
                                stacked_outputs_cf[k].append([x[k][i] for i in cf_idxs])
                        else:
                            stacked_outputs_ff[k].append(x[k])
                            stacked_outputs_cf[k].append(x[k])
            stacked_outputs_cf = stacked_outputs_cf if has_cf else None
            return stacked_outputs_ff, stacked_outputs_cf

        # split outputs by flows
        stacked_outputs_ff, stacked_outputs_cf = split_outputs_by_flows(outputs)

        # evaluate ff
        dict_metrics = shared_helper(stacked_outputs_ff, flow='ff')

        if stacked_outputs_cf is not None:
            # evaluate cf
            dict_cf_metrics = shared_helper(stacked_outputs_cf, flow='cf')
            # merge metrics
            dict_metrics = {**dict_metrics, **dict_cf_metrics}

        return dict_metrics

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = None):
        input_ids = batch["input_ids"]
        mask = input_ids != constants.PAD_ID
        token_type_ids = batch.get("token_type_ids", None)
        labels = batch.get("labels", None)
        lengths = mask.long().sum(dim=-1)

        # forward-pass
        z, y_hat = self(input_ids, mask, token_type_ids=token_type_ids, current_epoch=self.current_epoch)

        # get rationales
        z = [z_[:l] for z_, l in zip(z, lengths)]

        # tokens
        orig_tokens = [self.tokenizer.convert_ids_to_tokens(idxs) for idxs in input_ids.tolist()]
        orig_tokens = [e[:l] for e, l in zip(orig_tokens, lengths)]

        output = {
            "tokens": orig_tokens,
            "labels": labels.tolist() if labels is not None else None,
            "predictions": y_hat,
            "z": z,
        }
        return output
