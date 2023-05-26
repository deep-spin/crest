import os
import random
import re
from argparse import Namespace

import numpy as np
import torch
from transformers import AutoTokenizer

from rationalizers import constants
from rationalizers.data_modules import available_data_modules
from rationalizers.lightning_models import available_models
from rationalizers.utils import load_ckpt_config


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_tokenizer(args):
    # 1. Load a huggingface tokenizer
    print('Loading tokenizer: {}...'.format(args.tokenizer))
    if args.tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = None
    return tokenizer


def load_data_module(args, tokenizer):
    # 2. Load data module
    print('Loading data module: {}...'.format(args.dm))
    dm_cls = available_data_modules[args.dm]
    dm = dm_cls(d_params=vars(args), tokenizer=tokenizer)
    dm.load_encoders(
        root_dir=os.path.dirname(args.ckpt),
        load_tokenizer=args.load_tokenizer and tokenizer is None,
        load_label_encoder=args.load_label_encoder,
    )
    constants.update_constants(dm.tokenizer)
    return dm


def load_model(args, dm):
    # 3. Load model
    print('Loading model from {}...'.format(args.ckpt))
    model_cls = available_models[args.model]
    model = model_cls.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        map_location=lambda storage, loc: storage,
        strict=False,
        tokenizer=dm.tokenizer,
        nb_classes=dm.nb_classes,
        is_multilabel=dm.is_multilabel,
        h_params=vars(args),
    )
    # set the model to eval mode
    model.log_rationales_in_wandb = False
    model.generation_mode = True
    model.eval()
    return model


def load(args):
    tokenizer = load_tokenizer(args)
    dm = load_data_module(args, tokenizer)
    model = load_model(args, dm)
    return tokenizer, dm, model


def get_args_from_ckpt(ckpt_path, new_args):
    old_args = load_ckpt_config(ckpt_path)
    args = Namespace(**{**old_args, **new_args})
    args.ckpt = ckpt_path
    return args


def trim(text):
    text = re.sub(r'\ +', ' ', text)
    text = re.sub(r'(</s> )+', '</s> ', text)
    text = text.replace('</s> </s>', '</s>')
    return text.strip()


def tokens_to_text(raw_tokens):
    texts = []
    for tks in raw_tokens:
        texts.append(' '.join(['<unk>' if t is None else t for t in tks]))
    return texts
