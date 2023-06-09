import logging
import os
from copy import deepcopy
from itertools import chain
from random import shuffle

import numpy as np
import torch
import torchmetrics
import wandb
from torch import nn
from transformers import AutoModel, RobertaForMaskedLM, BertForMaskedLM

from rationalizers import constants
from rationalizers.builders import build_optimizer, build_scheduler
from rationalizers.explainers import available_explainers
from rationalizers.lightning_models.highlights.base import BaseRationalizer
from rationalizers.lightning_models.utils import (
    prepend_label_for_mice_t5, make_input_for_t5,
    get_new_frequencies_of_gen_ids_from_t5, repeat_interleave_and_pad,
    merge_input_and_gen_ids, sample_from_logits
)
from rationalizers.modules.metrics import evaluate_rationale
from rationalizers.modules.sentence_encoders import LSTMEncoder, MaskedAverageEncoder
from rationalizers.utils import (
    get_z_stats, freeze_module, masked_average, get_rationales, unroll, get_html_rationales,
    save_rationales, save_counterfactuals, load_torch_object, is_trainable, get_ext_mask
)

shell_logger = logging.getLogger(__name__)


class CounterfactualTransformerSPECTRARationalizer(BaseRationalizer):

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
        self.has_countertfactual_flow = True

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

        # counterfactual:
        self.cf_gen_arch = h_params.get("cf_gen_arch", "bert-base-multilingual-cased")
        self.cf_gen_emb_requires_grad = h_params.get("cf_gen_emb_requires_grad", False)
        self.cf_gen_encoder_requires_grad = h_params.get("cf_gen_encoder_requires_grad", True)
        self.cf_gen_lm_head_requires_grad = h_params.get("cf_gen_lm_head_requires_grad", True)
        self.cf_gen_use_decoder = h_params.get("cf_gen_use_decoder", True)
        self.cf_dropout = h_params.get("cf_dropout", 0.1)
        self.cf_input_space = h_params.get("cf_input_space", 'embedding')
        self.cf_selection_vector = h_params.get("cf_selection_vector", 'zero')
        self.cf_selection_mask = h_params.get("cf_selection_mask", True)
        self.cf_use_reinforce = h_params.get('cf_use_reinforce', True)
        self.cf_use_reinforce_baseline = h_params.get('cf_use_reinforce_baseline', True)
        self.cf_lbda = h_params.get('cf_lbda', 1.0)
        self.cf_generate_kwargs = h_params.get('cf_generate_kwargs', dict())
        self.cf_prepend_label_for_mice = h_params.get("cf_prepend_label_for_mice", False)
        self.cf_task_for_mice = h_params.get("cf_task_for_mice", "imdb")
        self.cf_manual_sample = h_params.get("cf_manual_sample", True)
        self.cf_supervision = h_params.get("cf_supervision", 'none')
        self.penalty_seq2seq = h_params.get('penalty_seq2seq', 1.0)
        self.penalty_similarity = h_params.get('penalty_similarity', 1.0)
        self.penalty_diversity = h_params.get('penalty_diversity', 1.0)
        self.penalty_fluency = h_params.get('penalty_fluency', 1.0)
        self.cf_margin = h_params.get('cf_margin', 0.0)
        self.cf_margin_lbda = h_params.get('cf_margin_lbda', 1.0)

        # explainer:
        self.explainer_pre_mlp = h_params.get("explainer_pre_mlp", True)
        self.explainer_requires_grad = h_params.get("explainer_requires_grad", True)
        self.explainer_mask_token_type_id = h_params.get("explainer_mask_token_type_id", None)
        self.temperature = h_params.get("temperature", 1.0)

        # both
        self.share_generators = h_params.get("share_generators", False)

        ########################
        # useful vars
        ########################
        self.ff_z = None
        self.ff_z_dist = None
        self.cf_z = None
        self.cf_z_dist = None
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id

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
        explainer_cls = available_explainers['sparsemap']
        self.explainer = explainer_cls(h_params, self.ff_gen_hidden_size)
        self.explainer_mlp = nn.Sequential(
            nn.Linear(self.ff_gen_hidden_size, self.ff_gen_hidden_size),
            nn.Tanh(),
        )

        # predictor module
        if self.ff_pred_arch == 'lstm':
            self.ff_pred_encoder = LSTMEncoder(self.ff_gen_hidden_size, self.ff_gen_hidden_size, bidirectional=True)
            self.ff_pred_hidden_size = self.ff_gen_hidden_size * 2
            self.ff_pred_emb_layer = self.ff_gen_emb_layer
            self.ff_pred_hf = None
            self.ff_pred_decoder = None
        elif self.ff_pred_arch == 'masked_average':
            self.ff_pred_encoder = MaskedAverageEncoder()
            self.ff_pred_hidden_size = self.ff_gen_hidden_size
            self.ff_pred_emb_layer = self.ff_gen_emb_layer
            self.ff_pred_hf = None
            self.ff_pred_decoder = None
        else:
            self.ff_pred_hf = AutoModel.from_pretrained(self.ff_pred_arch)
            self.ff_pred_hidden_size = self.ff_pred_hf.config.hidden_size
            self.ff_pred_emb_layer = self.ff_pred_hf.shared if 't5' in self.ff_pred_arch else self.ff_pred_hf.embeddings
            self.ff_pred_encoder = self.ff_pred_hf.encoder
            self.ff_pred_decoder = self.ff_pred_hf.decoder if hasattr(self.ff_pred_hf, 'decoder') else None

        # predictor output layer
        self.ff_output_layer = nn.Sequential(
            nn.Linear(self.ff_pred_hidden_size, self.ff_pred_hidden_size),
            nn.Tanh(),
            nn.Dropout(self.ff_dropout),
            nn.Linear(self.ff_pred_hidden_size, self.nb_classes),
            nn.Sigmoid() if not self.is_multilabel else nn.LogSoftmax(dim=-1),
        )

        ########################
        # counterfactual flow
        ########################
        # for reinforce
        self.rf_n_points = 0
        self.rf_mean_baseline = 0
        self.cf_x_tilde = None
        self.cf_log_prob_x_tilde = None
        # for supervised sentence-similarity loss
        self.cf_emb = None
        self.cf_emb_tilde = None

        # counterfactual generator module
        if 't5' in self.cf_gen_arch:
            from transformers import T5Config, T5ForConditionalGeneration
            if 'mice' in self.cf_gen_arch:
                t5_config = T5Config.from_pretrained("t5-base", n_positions=512)
                self.cf_gen_hf = T5ForConditionalGeneration.from_pretrained("t5-base", config=t5_config)
                self.cf_gen_hf.load_state_dict(load_torch_object(self.cf_gen_arch), strict=False)
            else:
                self.cf_gen_hf = T5ForConditionalGeneration.from_pretrained(self.cf_gen_arch)
            self.cf_gen_emb_layer = self.cf_gen_hf.shared
            self.cf_gen_encoder = self.cf_gen_hf.encoder
            self.cf_gen_decoder = self.cf_gen_hf.decoder
            if self.cf_gen_lm_head_requires_grad != self.cf_gen_emb_requires_grad:
                self.cf_gen_hf.lm_head = deepcopy(self.cf_gen_hf.lm_head)  # detach lm_head from shared weights
            self.cf_gen_lm_head = self.cf_gen_hf.lm_head
            self.cf_gen_hidden_size = self.cf_gen_hf.config.hidden_size
        elif 'roberta' in self.cf_gen_arch:
            self.cf_gen_hf = RobertaForMaskedLM.from_pretrained(self.cf_gen_arch)
            if self.cf_gen_lm_head_requires_grad != self.cf_gen_emb_requires_grad:
                self.cf_gen_hf.lm_head = deepcopy(self.cf_gen_hf.lm_head)  # detach lm_head from shared weights
            self.cf_gen_emb_layer = self.cf_gen_hf.roberta.embeddings
            self.cf_gen_encoder = self.cf_gen_hf.roberta.encoder
            self.cf_gen_lm_head = self.cf_gen_hf.lm_head
            self.cf_gen_decoder = None
            self.cf_gen_hidden_size = self.cf_gen_hf.roberta.config.hidden_size
        elif 'bert' in self.cf_gen_arch:
            self.cf_gen_hf = BertForMaskedLM.from_pretrained(self.cf_gen_arch)
            if self.cf_gen_lm_head_requires_grad != self.cf_gen_emb_requires_grad:
                self.cf_gen_hf.cls = deepcopy(self.cf_gen_hf.cls)  # detach lm_head from shared weights
            self.cf_gen_emb_layer = self.cf_gen_hf.bert.embeddings
            self.cf_gen_encoder = self.cf_gen_hf.bert.encoder
            self.cf_gen_lm_head = self.cf_gen_hf.cls
            self.cf_gen_decoder = None
            self.cf_gen_hidden_size = self.cf_gen_hf.bert.config.hidden_size
        else:
            raise NotImplementedError

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
        if not self.cf_gen_emb_requires_grad:
            freeze_module(self.cf_gen_emb_layer)

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

        # freeze models and set to eval mode to disable dropout
        if not self.cf_gen_encoder_requires_grad:
            freeze_module(self.cf_gen_encoder)
            if self.cf_gen_decoder is not None:
                freeze_module(self.cf_gen_decoder)

        # the lm head is an independent factor, which we can freeze or not
        if not self.cf_gen_lm_head_requires_grad:
            # it should not be shared with the embedding layer
            assert id(self.cf_gen_lm_head.weight) != id(self.cf_gen_emb_layer.weight)
            freeze_module(self.cf_gen_lm_head)

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

        # shared factual and counterfactual generators (only the LM head remains separate)
        if self.share_generators:
            del self.cf_gen_hf
            del self.cf_gen_emb_layer
            del self.cf_gen_encoder
            del self.cf_gen_hidden_size
            self.cf_gen_hf = self.ff_gen_hf
            self.cf_gen_emb_layer = self.ff_gen_emb_layer
            self.cf_gen_encoder = self.ff_gen_encoder
            self.cf_gen_decoder = self.ff_gen_decoder
            self.cf_gen_hidden_size = self.ff_gen_hidden_size

        ########################
        # loss functions
        ########################
        criterion_cls = nn.MSELoss if not self.is_multilabel else nn.NLLLoss
        self.ff_criterion = criterion_cls(reduction="none")
        self.cf_criterion = criterion_cls(reduction="none")

        if 'fluency' in self.cf_supervision:
            from transformers import AutoModelForCausalLM
            self.lm_model = AutoModelForCausalLM.from_pretrained("gpt2")
            self.lm_model.eval()
            freeze_module(self.lm_model)

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
        cf_params = chain(
            self.cf_gen_emb_layer.parameters() if not self.share_generators else [],
            self.cf_gen_encoder.parameters() if not self.share_generators else [],
            self.cf_gen_decoder.parameters() if not self.share_generators and self.cf_gen_decoder is not None else [],
            self.cf_gen_lm_head.parameters(),
        )
        grouped_parameters = []
        if self.ff_lbda > 0:
            grouped_parameters += [{"params": ff_params, 'lr': self.hparams['lr']}]
        if self.cf_lbda > 0:
            grouped_parameters += [{"params": cf_params, 'lr': self.hparams['cf_lr']}]

        optimizer = build_optimizer(grouped_parameters, self.hparams)
        scheduler = build_scheduler(optimizer, self.hparams)
        output = {"optimizer": optimizer}
        if scheduler is not None:
            output["scheduler"] = scheduler
            output["monitor"] = self.hparams['monitor']  # not sure we need this
        return output

    def forward(
        self,
        x: torch.LongTensor,
        x_cf: torch.LongTensor = None,
        mask: torch.BoolTensor = None,
        mask_cf: torch.BoolTensor = None,
        token_type_ids: torch.BoolTensor = None,
        token_type_ids_cf: torch.BoolTensor = None,
        current_epoch=None,
    ):
        """
        Compute forward-pass.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param x_cf: counterfactual input ids tensor. torch.LongTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param mask_cf: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param token_type_ids: mask tensor for explanation positions. torch.BoolTensor of shape [B, T]
        :param token_type_ids_cf: mask tensor for explanation positions. torch.BoolTensor of shape [B, T]
        :param current_epoch: int represents the current epoch.
        :return: (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat)
        """
        # factual flow
        z, y_hat = self.get_factual_flow(
            x, mask=mask, token_type_ids=token_type_ids
        )

        # prepend label for mice supervision-mode
        if 'mice' in self.cf_gen_arch and self.cf_prepend_label_for_mice:
            x_cf, z, mask_cf = prepend_label_for_mice_t5(
                y_hat, x_cf, z, mask_cf, self.tokenizer, self.cf_task_for_mice
            )

        # counterfactual flow
        x_tilde, z_tilde, mask_tilde, y_tilde_hat = self.get_counterfactual_flow(
            x_cf, z, mask=mask_cf, token_type_ids=token_type_ids_cf
        )

        # return everything as output (useful for computing the loss)
        return (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat)

    def get_factual_flow(self, x, mask=None, token_type_ids=None, z=None, from_cf=False):
        """
        Compute the factual flow.

        :param x: input ids tensor. torch.LongTensor of shape [B, T] or input vectors of shape [B, T, |V|]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param token_type_ids: mask tensor for explanation positions. torch.BoolTensor of shape [B, T]
        :param z: precomputed latent vector. torch.FloatTensor of shape [B, T] (default None)
        :param from_cf: bool, whether the input is from the counterfactual flow (default False)
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
        if token_type_ids is not None and self.explainer_mask_token_type_id is not None:
            # focus only on the explanation positions of the first concatenated input
            e_mask = mask & (token_type_ids == self.explainer_mask_token_type_id)
        else:
            e_mask = mask
        gen_h = self.explainer_mlp(gen_h) if self.explainer_pre_mlp else gen_h
        z, z_dist = self.explainer(gen_h, e_mask) if z is None else z
        z_mask = (z * e_mask.float()).unsqueeze(-1)

        # save vars
        if from_cf:
            self.cf_z = z
            self.cf_z_dist = z_dist
        else:
            self.ff_z = z
            self.ff_z_dist = z_dist

        # decide if we pass embeddings or hidden states to the predictor
        if self.ff_selection_faithfulness is True:
            if x.dim() == 2:
                # receive input ids
                pred_e = self.ff_pred_emb_layer(x)
            else:
                # receive one-hot vectors
                inputs_embeds = x @ self.ff_pred_emb_layer.word_embeddings.weight
                pred_e = self.ff_pred_emb_layer(inputs_embeds=inputs_embeds)
        else:
            pred_e = gen_h

        # decide if we will use a <mask>/<pad> vector or a <zero> vector
        if self.ff_selection_vector == 'mask':
            if 't5' in self.gen_arch:
                # create an input with sentinel tokens for T5
                sentinel_ids = 32100 - (z > 0).long().cumsum(dim=-1)
                x_mask = torch.clamp(sentinel_ids, min=32000, max=32099)
            else:
                # create an input with full mask tokens for other archs
                x_mask = torch.ones(x.shape[:2], device=x.device) * self.mask_token_id
            pred_e_mask = self.ff_pred_emb_layer(x_mask)
        elif self.ff_selection_vector == 'pad':
            x_mask = torch.ones(x.shape[:2], device=x.device) * self.pad_token_id
            pred_e_mask = self.ff_pred_emb_layer(x_mask)
        else:
            pred_e_mask = torch.zeros_like(pred_e)

        # create the input that will be passed to the predictor
        pred_e = pred_e * z_mask + pred_e_mask * (1 - z_mask)

        # pass through the predictor
        if self.ff_pred_arch == 'lstm':
            _, summary = self.ff_pred_encoder(pred_e, mask)
        elif self.ff_pred_arch == 'masked_average':
            summary = self.ff_pred_encoder(pred_e, mask)
        else:
            if 't5' in self.ff_pred_arch:
                # decide if we will mask the self-attention of the predictor
                attn_mask = (1 - z_mask.squeeze(-1)).long() if self.ff_selection_mask else mask
                pred_h = self.ff_pred_encoder(
                    inputs_embeds=pred_e,
                    attention_mask=attn_mask
                )
                if self.ff_gen_use_decoder:
                    pred_h = self.ff_pred_decoder(
                        inputs_embeds=pred_e,
                        attention_mask=attn_mask,
                        encoder_hidden_states=pred_h.last_hidden_state,
                    )
            else:
                # decide if we will mask the self-attention of the predictor
                ext_mask = get_ext_mask(1 - z_mask.squeeze(-1)) if self.ff_selection_mask else get_ext_mask(mask)
                pred_h = self.ff_pred_encoder(
                    hidden_states=pred_e,
                    attention_mask=ext_mask
                )

            # get final hidden states
            # selected_layers = list(map(int, self.selected_layers.split(',')))
            # pred_h = torch.stack(pred_h)[selected_layers].mean(dim=0)
            pred_h = pred_h.last_hidden_state

            # compute a sentence vector
            summary = masked_average(pred_h, mask)

        # pass through the final mlp
        y_hat = self.ff_output_layer(summary)

        return z, y_hat

    def get_counterfactual_flow(self, x, z, mask=None, token_type_ids=None):
        """
        Compute the counterfactual flow.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param z: binary variables tensor. torch.FloatTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param token_type_ids: mask tensor for explanation positions. torch.BoolTensor of shape [B, T]
        :return: (x_tilde, z_tilde, mask_tilde, y_tilde_hat)
        """
        if self.cf_manual_sample:
            # reuse the factual flow to get a prediction for the counterfactual flow
            z_tilde, y_tilde_hat = self.get_factual_flow(x, mask=mask, token_type_ids=token_type_ids, from_cf=True)
            return x, z_tilde, mask, y_tilde_hat

        # prepare input for the generator LM
        e = self.cf_gen_emb_layer(x) if self.cf_input_space == 'embedding' else x

        # get <mask> vectors
        if 't5' in self.cf_gen_arch:
            # fix inputs for t5 (replace chunked masked positions by a single sentinel token)
            x, e, z, mask = make_input_for_t5(x, e, z, mask, pad_id=constants.PAD_ID)
            # create sentinel tokens
            sentinel_ids = 32100 - (z > 0).long().cumsum(dim=-1)
            # get sentinel embeddings
            e_mask = self.cf_gen_emb_layer(sentinel_ids)
        else:
            e_mask = self.cf_gen_emb_layer(torch.ones_like(x) * self.mask_token_id)

        # create mask for pretrained-LMs
        ext_mask = (1.0 - mask[:, None, None, :].to(self.dtype)) * -10000.0

        # get the new mask
        z_mask = (z * mask.float()).unsqueeze(-1)

        # set the input for the counterfactual encoder via a differentiable where
        s_bar = e_mask * z_mask + e * (1 - z_mask)

        # pass (1-z)-masked inputs
        if 't5' in self.cf_gen_arch:
            cf_gen_enc_out = self.cf_gen_encoder(inputs_embeds=s_bar, attention_mask=mask)
            h_tilde = cf_gen_enc_out.last_hidden_state
        else:
            cf_gen_enc_out = self.cf_gen_encoder(s_bar, ext_mask)
            h_tilde = cf_gen_enc_out.last_hidden_state

        # sample from the LM head
        x_tilde, logits = self._sample_from_lm(x, h_tilde, z_mask, mask, encoder_outputs=cf_gen_enc_out)

        # expand z to account for the new generated tokens (important for t5)
        x_tilde, z_tilde, mask_tilde, token_type_ids_tilde = self._expand_factual_inputs_from_x_tilde(
            x, z, mask, x_tilde, token_type_ids=token_type_ids
        )

        # reuse the factual flow to get a prediction for the counterfactual flow
        z_tilde, y_tilde_hat = self.get_factual_flow(
            x_tilde, mask=mask_tilde, token_type_ids=token_type_ids_tilde, from_cf=True
        )

        return x_tilde, z_tilde, mask_tilde, y_tilde_hat

    def _expand_factual_inputs_from_x_tilde(self, x, z, mask, x_tilde, token_type_ids=None):
        if 't5' in self.cf_gen_arch:
            # get the counts needed to expand the input_ids into generated_ids
            gen_counts = get_new_frequencies_of_gen_ids_from_t5(
                x, x_tilde, pad_id=constants.PAD_ID, eos_id=constants.EOS_ID,
            )
            # and vice-versa
            inp_counts = get_new_frequencies_of_gen_ids_from_t5(
                x_tilde, x, pad_id=constants.PAD_ID, eos_id=constants.EOS_ID
            )

            # expand x, z, mask according to gen_counts
            x_rep = repeat_interleave_and_pad(x, gen_counts, pad_id=constants.PAD_ID)
            z_tilde = repeat_interleave_and_pad(z, gen_counts, pad_id=constants.PAD_ID)
            mask_tilde = repeat_interleave_and_pad(mask, gen_counts, pad_id=constants.PAD_ID)
            if token_type_ids is not None:
                token_type_ids_tilde = repeat_interleave_and_pad(token_type_ids, gen_counts, pad_id=constants.PAD_ID)
            else:
                token_type_ids_tilde = None
            # expand x_tilde according to inp_counts
            x_tilde_rep = repeat_interleave_and_pad(x_tilde, inp_counts)

            # merge x_rep and x_tilde_rep into a single tensor
            x_tilde = merge_input_and_gen_ids(x_rep, x_tilde_rep, pad_id=constants.PAD_ID)

            # fix the corner case of generating fewer tokens than what was selected
            original_seq_len = z_tilde.shape[-1]
            expanded_seq_len = x_tilde.shape[-1]
            if original_seq_len > expanded_seq_len:
                z_tilde = z_tilde[:, :expanded_seq_len]
                mask_tilde = mask_tilde[:, :expanded_seq_len]
                if token_type_ids_tilde is not None:
                    token_type_ids_tilde = token_type_ids_tilde[:, :expanded_seq_len]

            # remove labels in case they were prepended for mice t5
            # (the prepended prompt has 6 tokens)
            if 'mice' in self.cf_gen_arch and self.cf_prepend_label_for_mice:
                x_tilde = x_tilde[:, 6:]
                z_tilde = z_tilde[:, 6:]
                mask_tilde = mask_tilde[:, 6:]
                if token_type_ids_tilde is not None:
                    token_type_ids_tilde = token_type_ids_tilde[:, 6:]

            # if we generated too much, there isn't much we can do besides truncating
            if x_tilde.shape[-1] > 512 and 'bert' in self.cf_pred_arch:
                x_tilde = x_tilde[:, :512]
                z_tilde = z_tilde[:, :512]
                mask_tilde = mask_tilde[:, :512]
                if token_type_ids_tilde is not None:
                    token_type_ids_tilde = token_type_ids_tilde[:, :512]

        else:  # otherwise our dimensions match, so we can reuse the same z and mask
            z_tilde = z
            mask_tilde = mask
            token_type_ids_tilde = token_type_ids

        return x_tilde, z_tilde, mask_tilde, token_type_ids_tilde

    def _sample_from_lm(self, x, h_tilde, z_mask, mask, encoder_outputs=None, x_cf=None):
        if self.cf_use_reinforce:
            if 't5' in self.cf_gen_arch:
                # recover hidden states from the encoder (.generate() changes the hidden states)
                encoder_hidden_states = h_tilde.clone()

                # deal with min and max length
                gen_kwargs = self.cf_generate_kwargs.copy()
                if 'max_length' not in self.cf_generate_kwargs:
                    gen_kwargs['max_length'] = 512
                if self.cf_generate_kwargs.get('min_length', None) == 'original':
                    # set the minimum length to be at least equal to the number of
                    # sentinel tokens (times 2 since T5 has to generate sentinels too)
                    num_sentinels = (z_mask > 0).long().sum(-1).min().item()
                    gen_kwargs['min_length'] = min(gen_kwargs['max_length'], num_sentinels * 2)

                # sample autoregressively
                gen_out = self.cf_gen_hf.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=mask.long(),
                    return_dict_in_generate=True,
                    output_scores=True,
                    **gen_kwargs
                )
                # clear memory because generation is done
                # torch.cuda.empty_cache()

                # idk why but t5 generates a pad symbol as the first token
                # so we cut it out for all samples in the batch
                # (this happens only for .sequences)
                x_tilde = gen_out.sequences[:, 1:]

                # get the logits for x_tilde
                cf_gen_dec_out = self.cf_gen_decoder(
                    input_ids=gen_out.sequences,
                    attention_mask=(gen_out.sequences != constants.PAD_ID).long(),
                    encoder_hidden_states=encoder_hidden_states
                )
                logits = self.cf_gen_lm_head(cf_gen_dec_out.last_hidden_state)[:, :-1]

            else:
                # sample directly from the output layer
                logits = self.cf_gen_lm_head(h_tilde)
                # x_tilde = logits.argmax(dim=-1)
                x_tilde = sample_from_logits(
                    logits=logits,
                    top_k=self.cf_generate_kwargs.get('top_k', 0),
                    top_p=self.cf_generate_kwargs.get('top_p', 1.0),
                    min_tokens_to_keep=self.cf_generate_kwargs.get('min_tokens_to_keep', 1.0),
                    num_samples=self.cf_generate_kwargs.get('num_return_sequences', 1),
                ).squeeze(-1)

                # get gen_ids only for <mask> positions
                z_1 = (z_mask > 0).squeeze(-1).long()
                x_tilde = z_1 * x_tilde + (1 - z_1) * x

            # save variables for computing REINFORCE loss later
            self.cf_x_tilde = x_tilde.clone()
            self.cf_log_prob_x_tilde = torch.log_softmax(logits, dim=-1)

        else:
            # use the ST-gumbel-softmax trick
            logits = self.cf_gen_lm_head(h_tilde)
            x_tilde = nn.functional.gumbel_softmax(logits, hard=True, dim=-1)
            # x_tilde.shape is (bs, seq_len, |V|)

            # get gen_ids only for <mask> positions
            z_1 = (z_mask > 0).long()
            x_one_hot = nn.functional.one_hot(x, num_classes=x_tilde.shape[-1])
            x_tilde = z_1 * x_tilde + (1 - z_1) * x_one_hot

            # save variables for computing penalties later
            self.cf_x_tilde = x_tilde.clone()
            self.cf_log_prob_x_tilde = torch.log_softmax(logits, dim=-1)

        return x_tilde, logits

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

    def get_counterfactual_loss(self, y_hat, y, z_tilde, mask_tilde, prefix, x_tilde=None, x_cf=None):
        """
        Compute loss for the counterfactual flow.

        :param y_hat: predictions from SentimentPredictor. Torch.Tensor of shape [B, C]
        :param y: tensor with gold labels. torch.BoolTensor of shape [B]
        :param z_tilde: latent selection vector. torch.FloatTensor of shape [B, T]
        :param mask_tilde: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param prefix: prefix for loss statistics (train, val, test)
        :param x_tilde: generated counterfactual input ids tensor.
                        torch.LongTensor of shape [B, T] for reinforce sampling
                        torch.LongTensor of shape [B, T, V] for gumbel-softmax sampling
        :param x_cf: gold counterfactual input ids tensor. torch.LongTensor of shape [B, T]
        :return: tuple containing:
            `loss cost (torch.FloatTensor)`: the result of the loss function
            `loss stats (dict): dict with loss statistics
        """
        def get_x_emb(x):
            # simple function that recovers the embeddings of given ids
            if x.dim() == 2:
                # receive input ids
                return self.cf_gen_emb_layer(x)
            else:
                # receive one-hot vectors
                inputs_embeds = x @ self.cf_gen_emb_layer.word_embeddings.weight
                return self.cf_gen_emb_layer(inputs_embeds=inputs_embeds)

        stats = {}
        loss_vec = self.cf_criterion(y_hat, y)  # [B] or [B,C]

        # main MSE loss for p(y | x, z)
        # masked average
        if loss_vec.dim() == 2:
            loss = (loss_vec * mask_tilde.float()).sum(-1) / mask_tilde.sum(-1).float()  # [1]
        else:
            loss = loss_vec.mean()

        # main loss for p(y | x, z)
        stats["cf_mse" if not self.is_multilabel else "cf_criterion"] = loss.item()
        main_loss = loss

        # ideas for later (penalties):
        # use an "adaptor" layer which is a pretrained LM, such that new logits ~= adaptor logits
        # logits = alpha * self.adaptor_logits(x) + (1 - alpha) * self.cf_flow_logits(x)

        # supervise logp_xtilde
        penalty = torch.zeros_like(loss)

        if 'similarity' in self.cf_supervision:
            # maximize similarity between emb(x_cf) and emb(x_tilde)
            x_cf_emb = get_x_emb(x_cf).mean(1)
            x_tilde_emb = get_x_emb(x_tilde).mean(1)
            cost = 1 - torch.nn.functional.cosine_similarity(x_cf_emb, x_tilde_emb, dim=-1)
            penalty += self.penalty_similarity * cost.mean()

        if 'diversity' in self.cf_supervision:
            # maximize diversity of several sampled x_tildes
            # penalty += det(1 / 1 + dist(x_tilde_i, x_tilde_j))
            assert isinstance(x_tilde, (list, tuple))
            gen_embs = torch.stack([get_x_emb(x).mean(1) for x in x_tilde])
            gen_embs_sq = gen_embs.transpose(0, 1) ** 2
            dist_matrix = torch.sqrt(gen_embs_sq @ gen_embs_sq.transpose(-1, -2))
            cost = torch.linalg.det(1 / (1 + dist_matrix))
            penalty += self.penalty_diversity * cost.mean()

        if 'fluency' in self.cf_supervision:
            # maximize the fluency of x_cf_tilde
            # lm = https://huggingface.co/distilgpt2
            # penalty += neg_perplexity_pretrained_lm(x_tilde)
            lm_labels = x_tilde if x_tilde.dim() == 2 else x_tilde.argmax(dim=-1)
            lm_labels = lm_labels.masked_fill(lm_labels == self.pad_token_id, -100).detach()
            if x_tilde.dim() == 2:
                outputs = self.lm_model(input_ids=x_tilde, labels=lm_labels)
            else:
                x_tilde_embeds = x_tilde @ self.lm_model.transformer.wte.weight
                outputs = self.lm_model(inputs_embeds=x_tilde_embeds, labels=lm_labels)
            cost = outputs.loss
            penalty += self.penalty_fluency * cost.mean()

        # unsupervised setting, use reinforce or gumbel-softmax (default)
        if self.cf_use_reinforce:
            # for BERT:
            # the <mask> is on <mask> table
            # the book   is on the    table
            # p1  p2     p3 p4 p5     p6
            # 0   1      0  0  1      0
            # p * z

            # for T5:
            # the <extra_id_0> is on    <extra_id_1>  table
            # the long book    is under the very nice table
            # p1  p2   p3      p4 ...
            # 0   1    0       0 0    1    0
            # 0   1    1       0 0    1 1 1 0
            # p * 1

            # opt 1: independent samples
            # P(gen_id | s)
            # p = softmax(logits_output)

            # opt 2: autoregressive samples
            # P(gen_id_i | gen_id_{0:i-1}, s)
            # p = cumprod(p)

            # recover the probabilities of x_tilde with opt 1.
            # log P(x_tilde | s; phi)
            if 't5' in self.cf_gen_arch:
                # x:        the           <input_id_0> is on the <input_id_1>
                # x_tilde: <input_id_0>  long book    <input_id_1> table </s>
                gen_ids = self.cf_x_tilde  # [B, T]
                # recover log prob of all sampled tokens (including sentinels)
                logp_xtilde = self.cf_log_prob_x_tilde  # [B, T, |V|]
                # compute sentinel and padding mask
                gen_mask = ~((gen_ids >= 32000) & (gen_ids <= 32099))  # non-sentinel
                gen_mask &= (gen_ids != constants.PAD_ID)  # non-padding
                # get probas of x_tilde [B, T]
                logp_xtilde = logp_xtilde.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
                # weighted average (ignore sentinels and pad)
                log_xtilde_scalar = (logp_xtilde * gen_mask.float()).sum(1) / gen_mask.float().sum(1)
            else:
                # x:        the  <mask> is on the <mask>
                # x_tilde:  the  book   is on the table
                gen_ids = self.cf_x_tilde  # [B, T]
                logp_xtilde = self.cf_log_prob_x_tilde  # [B, T, |V|]
                # compute only the log prob of new sampled tokens
                gen_mask = (gen_ids != constants.PAD_ID) & (z_tilde > 0)
                # get probas of x_tilde
                logp_xtilde = logp_xtilde.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
                # weighted average (ignore sentinels and pad)
                log_xtilde_scalar = (logp_xtilde * gen_mask.float()).sum(1) / gen_mask.float().sum(1)

            # compute generator loss
            cost_vec = loss_vec.detach()
            # cost_vec is neg reward
            cost_logpz = ((cost_vec - self.rf_mean_baseline) * log_xtilde_scalar).mean(0)

            # MSE with regularizers = neg reward
            obj = cost_vec.mean()
            stats["cf_obj"] = obj.item()

            # add baseline
            if self.cf_use_reinforce_baseline:
                self.rf_n_points += 1.0
                self.rf_mean_baseline += (cost_vec.detach().mean() - self.rf_mean_baseline) / self.rf_n_points

            # pred diff doesn't do anything if only 1 aspect being trained
            if not self.is_multilabel:
                pred_diff = y_hat.max(dim=1)[0] - y_hat.min(dim=1)[0]
                pred_diff = pred_diff.mean()
                stats["cf_pred_diff"] = pred_diff.item()

            # generator cost
            stats["cf_cost_g"] = cost_logpz.item()

            # predictor cost
            stats["cf_cost_p"] = loss.item()

            main_loss = loss + cost_logpz

        # add penalties
        main_loss = main_loss + penalty

        # latent selection stats
        num_0, num_c, num_1, total = get_z_stats(z_tilde, mask_tilde)
        stats[prefix + "_cf_p0"] = num_0 / float(total)
        stats[prefix + "_cf_pc"] = num_c / float(total)
        stats[prefix + "_cf_p1"] = num_1 / float(total)
        stats[prefix + "_cf_ps"] = (num_c + num_1) / float(total)
        stats[prefix + "_cf_main_loss"] = main_loss.item()
        stats[prefix + "_cf_penalty"] = penalty.item()

        return main_loss, stats

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
        cf_input_ids = batch["cf_input_ids"] if "cf_input_ids" in batch else input_ids
        cf_mask = cf_input_ids != constants.PAD_ID
        cf_labels = batch["cf_labels"] if "cf_labels" in batch else labels
        token_type_ids = batch.get("token_type_ids", None)
        cf_token_type_ids = batch.get("cf_token_type_ids", None)
        prefix = "train"
        self.stage = prefix

        # forward pass
        if self.cf_manual_sample is True or 'seq2seq' in self.cf_supervision:
            (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat) = self(
                input_ids, cf_input_ids, mask, cf_mask, token_type_ids, cf_token_type_ids,
                current_epoch=self.current_epoch
            )
        else:
            # fix inputs for supervised training
            (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat) = self(
                input_ids, input_ids, mask, mask, token_type_ids, token_type_ids,
                current_epoch=self.current_epoch
            )

        # compute factual loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        ff_loss, loss_stats = self.get_factual_loss(y_hat, y, z, mask, prefix=prefix)

        # compute counterfactual loss
        y_tilde_hat = y_tilde_hat if not self.is_multilabel else y_tilde_hat.view(-1, self.nb_classes)
        y_tilde = cf_labels if not self.is_multilabel else cf_labels.view(-1)
        cf_loss, cf_loss_stats = self.get_counterfactual_loss(
            y_tilde_hat, y_tilde, z_tilde, mask_tilde, prefix=prefix,
            x_tilde=x_tilde, x_cf=cf_input_ids,
        )

        # combine losses
        if self.cf_margin > 0:
            loss = self.ff_lbda * ff_loss + self.cf_lbda * cf_loss
            loss += self.cf_margin_lbda * torch.relu(ff_loss - cf_loss + self.cf_margin)
        else:
            loss = self.ff_lbda * ff_loss + self.cf_lbda * cf_loss

        # logger=False because they are going to be logged via loss_stats
        self.log("train_ff_ps", loss_stats["train_ps"], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("train_cf_ps", cf_loss_stats["train_cf_ps"], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("train_ff_sum_loss", ff_loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)
        self.log("train_cf_sum_loss", cf_loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)
        self.log("train_sum_loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)

        if self.is_multilabel:
            metrics_to_wandb = {
                "train_ff_p1": loss_stats["train_p1"],
                "train_cf_p1": cf_loss_stats["train_cf_p1"],
                "train_ff_ps": loss_stats["train_ps"],
                "train_cf_ps": cf_loss_stats["train_cf_ps"],
                "train_ff_sum_loss": loss_stats["criterion"],
                "train_cf_sum_loss": cf_loss_stats["cf_criterion"],
            }
        else:
            metrics_to_wandb = {
                "train_ff_p1": loss_stats["train_p1"],
                "train_cf_p1": cf_loss_stats["train_cf_p1"],
                "train_ff_ps": loss_stats["train_ps"],
                "train_cf_ps": cf_loss_stats["train_cf_ps"],
                "train_ff_sum_loss": loss_stats["mse"],
                "train_cf_sum_loss": cf_loss_stats["cf_mse"],
            }
        if "cost_g" in loss_stats:
            metrics_to_wandb["train_cost_g"] = loss_stats["cost_g"]
        if "cost_g" in cf_loss_stats:
            metrics_to_wandb["train_cf_cost_g"] = cf_loss_stats["cf_cost_g"]

        self.logger.log_metrics(metrics_to_wandb, self.global_step)

        # return the loss tensor to PTL
        return {"loss": loss, "ps": loss_stats["train_ps"], "cf_ps": cf_loss_stats["train_cf_ps"]}

    def _shared_eval_step(self, batch: dict, batch_idx: int, prefix: str):
        input_ids = batch["input_ids"]
        mask = input_ids != constants.PAD_ID
        labels = batch["labels"]
        cf_input_ids = batch["cf_input_ids"] if "cf_input_ids" in batch else input_ids
        cf_mask = cf_input_ids != constants.PAD_ID
        cf_labels = batch["cf_labels"] if "cf_labels" in batch else labels
        token_type_ids = batch.get("token_type_ids", None)
        cf_token_type_ids = batch.get("cf_token_type_ids", None)
        self.stage = prefix

        # forward pass
        if self.cf_manual_sample is True or 'seq2seq' in self.cf_supervision:
            (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat) = self(
                input_ids, cf_input_ids, mask, cf_mask, token_type_ids, cf_token_type_ids,
                current_epoch=self.current_epoch
            )
        else:
            # fix inputs for supervised training
            (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat) = self(
                input_ids, input_ids, mask, mask, token_type_ids, token_type_ids,
                current_epoch=self.current_epoch
            )

        # compute factual loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        ff_loss, ff_loss_stats = self.get_factual_loss(y_hat, y, z, mask, prefix=prefix)
        self.logger.agg_and_log_metrics(ff_loss_stats, step=None)

        # compute counterfactual loss
        y_tilde_hat = y_tilde_hat if not self.is_multilabel else y_tilde_hat.view(-1, self.nb_classes)
        y_tilde = cf_labels if not self.is_multilabel else cf_labels.view(-1)
        cf_loss, cf_loss_stats = self.get_counterfactual_loss(
            y_tilde_hat, y_tilde, z_tilde, mask_tilde, prefix=prefix,
            x_tilde=x_tilde, x_cf=cf_input_ids,
        )
        self.logger.agg_and_log_metrics(cf_loss_stats, step=None)

        # combine losses
        if self.cf_margin > 0:
            loss = self.ff_lbda * ff_loss + self.cf_lbda * cf_loss
            loss += self.cf_margin_lbda * torch.relu(ff_loss - cf_loss + self.cf_margin)
        else:
            loss = self.ff_lbda * ff_loss + self.cf_lbda * cf_loss

        # log metrics
        self.log(f"{prefix}_ff_sum_loss", ff_loss.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True,)
        self.log(f"{prefix}_cf_sum_loss", cf_loss.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True,)
        self.log(f"{prefix}_sum_loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)

        # get factual rationales
        z_1 = (z > 0).long()  # non-zero probs are considered selections
        ff_rat_ids, ff_rat_tokens = get_rationales(self.tokenizer, input_ids, z_1, batch["lengths"])
        ff_tokens = [self.tokenizer.convert_ids_to_tokens(idxs) for idxs in input_ids.tolist()]

        # get counterfactuals
        gen_ids = x_tilde if x_tilde.dim() == 2 else x_tilde.argmax(dim=-1)
        if 't5' not in self.cf_gen_arch and 'seq2seq' not in self.cf_supervision:
            z_1 = (z_tilde > 0).long()
            gen_ids = z_1 * gen_ids + (1 - z_1) * cf_input_ids
        cf_tokens = [self.tokenizer.convert_ids_to_tokens(idxs) for idxs in gen_ids.tolist()]
        cf_lengths = (gen_ids != constants.PAD_ID).long().sum(-1).tolist()

        # output to be stacked across iterations
        output = {
            f"{prefix}_ff_sum_loss": ff_loss.item(),
            f"{prefix}_cf_sum_loss": cf_loss.item(),
            f"{prefix}_ff_ps": ff_loss_stats[prefix + "_ps"],
            f"{prefix}_cf_ps": cf_loss_stats[prefix + "_cf_ps"],
            f"{prefix}_ids_rationales": ff_rat_ids,
            f"{prefix}_rationales": ff_rat_tokens,
            f"{prefix}_pieces": ff_tokens,
            f"{prefix}_tokens": batch["tokens"],
            f"{prefix}_z": z,
            f"{prefix}_predictions": y_hat,
            f"{prefix}_labels": y.tolist(),
            f"{prefix}_lengths": batch["lengths"].tolist(),
            f"{prefix}_cfs": cf_tokens,
            f"{prefix}_cf_labels": y_tilde.tolist(),
            f"{prefix}_cf_predictions": y_tilde_hat,
            f"{prefix}_cf_z": z_tilde,
            f"{prefix}_cf_lengths": cf_lengths,
        }

        if "annotations" in batch.keys():
            output[f"{prefix}_annotations"] = batch["annotations"]
        if "mse" in ff_loss_stats.keys():
            output[f"{prefix}_ff_mse"] = ff_loss_stats["mse"]
        if "mse" in cf_loss_stats.keys():
            output[f"{prefix}_cf_mse"] = cf_loss_stats["mse"]
        return output

    def _shared_eval_epoch_end(self, outputs: list, prefix: str):
        """
        PTL hook. Perform validation at the end of an epoch.

        :param outputs: list of dicts representing the stacked outputs from validation_step
        :param prefix: `val` or `test`
        """
        # assume that `outputs` is a list containing dicts with the same keys
        stacked_outputs = {k: [x[k] for x in outputs] for k in outputs[0].keys()}

        # sample a few examples to be logged in wandb
        idxs = list(range(sum(map(len, stacked_outputs[f"{prefix}_pieces"]))))
        shuffle(idxs)
        idxs = idxs[:10] if prefix != 'test' else idxs[:100]

        # useful functions
        select = lambda v: [v[i] for i in idxs]
        detach = lambda v: [v[i].detach().cpu() for i in range(len(v))]

        if self.log_rationales_in_wandb:
            # log rationales
            pieces = select(unroll(stacked_outputs[f"{prefix}_pieces"]))
            scores = detach(select(unroll(stacked_outputs[f"{prefix}_z"])))
            gold = select(unroll(stacked_outputs[f"{prefix}_labels"]))
            pred = detach(select(unroll(stacked_outputs[f"{prefix}_predictions"])))
            lens = select(unroll(stacked_outputs[f"{prefix}_lengths"]))
            html_string = get_html_rationales(pieces, scores, gold, pred, lens)
            self.logger.experiment.log({f"{prefix}_rationales": wandb.Html(html_string)})

            # log counterfactuals
            cfs = select(unroll(stacked_outputs[f"{prefix}_cfs"]))
            scores = detach(select(unroll(stacked_outputs[f"{prefix}_cf_z"])))
            gold = select(unroll(stacked_outputs[f"{prefix}_cf_labels"]))
            pred = detach(select(unroll(stacked_outputs[f"{prefix}_cf_predictions"])))
            lens = select(unroll(stacked_outputs[f"{prefix}_cf_lengths"]))
            html_string = get_html_rationales(cfs, scores, gold, pred, lens)
            self.logger.experiment.log({f"{prefix}_counterfactuals": wandb.Html(html_string)})

        # save rationales
        if self.hparams.save_rationales:
            # factual rationales
            scores = detach(unroll(stacked_outputs[f"{prefix}_z"]))
            lens = unroll(stacked_outputs[f"{prefix}_lengths"])
            filename = os.path.join(self.hparams.default_root_dir, f'{prefix}_ff_rationales.txt')
            shell_logger.info(f'Saving rationales in {filename}...')
            save_rationales(filename, scores, lens)

            # counterfactual rationales
            scores = detach(unroll(stacked_outputs[f"{prefix}_cf_z"]))
            lens = unroll(stacked_outputs[f"{prefix}_cf_lengths"])
            filename = os.path.join(self.hparams.default_root_dir, f'{prefix}_cf_rationales.txt')
            shell_logger.info(f'Saving rationales in {filename}...')
            save_rationales(filename, scores, lens)

        # save counterfactuals
        if self.hparams.save_counterfactuals:
            pieces = unroll(stacked_outputs[f"{prefix}_cfs"])
            lens = unroll(stacked_outputs[f"{prefix}_cf_lengths"])
            filename = os.path.join(self.hparams.default_root_dir, f'{prefix}_counterfactuals.txt')
            shell_logger.info(f'Saving counterfactuals in {filename}...')
            save_counterfactuals(filename, pieces, lens)

        # log metrics
        dict_metrics = {
            f"{prefix}_ff_ps": np.mean(stacked_outputs[f"{prefix}_ff_ps"]),
            f"{prefix}_ff_sum_loss": np.mean(stacked_outputs[f"{prefix}_ff_sum_loss"]),
            f"{prefix}_cf_ps": np.mean(stacked_outputs[f"{prefix}_cf_ps"]),
            f"{prefix}_cf_sum_loss": np.mean(stacked_outputs[f"{prefix}_cf_sum_loss"]),
        }

        # only evaluate rationales on the test set and if we have annotation (only for beer dataset)
        if prefix == "test" and "test_annotations" in stacked_outputs.keys():
            rat_metrics = evaluate_rationale(
                stacked_outputs["test_ids_rationales"],
                stacked_outputs["test_annotations"],
                stacked_outputs["test_lengths"],
            )
            dict_metrics[f"{prefix}_ff_rat_precision"] = rat_metrics["macro_precision"]
            dict_metrics[f"{prefix}_ff_rat_recall"] = rat_metrics["macro_recall"]
            dict_metrics[f"{prefix}_ff_rat_f1"] = rat_metrics["f1_score"]

        # log classification metrics
        if self.is_multilabel:
            preds = torch.argmax(torch.cat(stacked_outputs[f"{prefix}_predictions"]), dim=-1)
            labels = torch.tensor(unroll(stacked_outputs[f"{prefix}_labels"]), device=preds.device)
            cf_preds = torch.argmax(torch.cat(stacked_outputs[f"{prefix}_cf_predictions"]), dim=-1)
            cf_labels = torch.tensor(unroll(stacked_outputs[f"{prefix}_cf_labels"]), device=cf_preds.device)
            ff_accuracy = torchmetrics.functional.accuracy(
                preds, labels, num_classes=self.nb_classes, average="macro"
            )
            ff_precision = torchmetrics.functional.precision(
                preds, labels, num_classes=self.nb_classes, average="macro"
            )
            ff_recall = torchmetrics.functional.recall(
                preds, labels, num_classes=self.nb_classes, average="macro"
            )
            ff_f1_score = 2 * ff_precision * ff_recall / (ff_precision + ff_recall)
            cf_accuracy = torchmetrics.functional.accuracy(
                cf_preds, cf_labels, num_classes=self.nb_classes, average="macro"
            )
            cf_precision = torchmetrics.functional.precision(
                cf_preds, cf_labels, num_classes=self.nb_classes, average="macro"
            )
            cf_recall = torchmetrics.functional.recall(
                cf_preds, cf_labels, num_classes=self.nb_classes, average="macro"
            )
            cf_f1_score = 2 * cf_precision * cf_recall / (cf_precision + cf_recall)
            dict_metrics[f"{prefix}_ff_accuracy"] = ff_accuracy
            dict_metrics[f"{prefix}_ff_precision"] = ff_precision
            dict_metrics[f"{prefix}_ff_recall"] = ff_recall
            dict_metrics[f"{prefix}_ff_f1score"] = ff_f1_score
            dict_metrics[f"{prefix}_cf_accuracy"] = cf_accuracy
            dict_metrics[f"{prefix}_cf_precision"] = cf_precision
            dict_metrics[f"{prefix}_cf_recall"] = cf_recall
            dict_metrics[f"{prefix}_cf_f1score"] = cf_f1_score
        else:
            dict_metrics[f"{prefix}_ff_mse"] = np.mean(stacked_outputs[f"{prefix}_ff_mse"])
            dict_metrics[f"{prefix}_cf_mse"] = np.mean(stacked_outputs[f"{prefix}_cf_mse"])

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
