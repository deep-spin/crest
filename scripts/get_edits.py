import argparse
import os
from pprint import pprint

import datasets as hf_datasets
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer

from rationalizers.data_modules import available_data_modules
from rationalizers.utils import load_torch_object
from rationalizers.utils import unroll
from utils import configure_seed, load, get_args_from_ckpt, tokens_to_text


# import warnings
# warnings.filterwarnings("ignore")


def predict(
    ckpt_path,
    factual_path,
    cf_generate_kwargs,
    dm_name,
    dm_args,
    dataloader='train',
    verbose=True,
    disable_progress_bar=False,
    return_tokenizer=False,
    sparsemap_budget=None
):
    # disable hf_dataset progress bar
    if disable_progress_bar:
        hf_datasets.logging.disable_progress_bar()
    else:
        hf_datasets.logging.enable_progress_bar()
    if verbose:
        hf_datasets.logging.set_verbosity(20)
    else:
        hf_datasets.logging.set_verbosity(50)

    # load args
    base_args = dict(
        seed=0,
        load_tokenizer=False,
        load_label_encoder=False,
        save_rationales=True,
        save_edits=True,
        cf_classify_edits=False,
        cf_generate_kwargs=cf_generate_kwargs
    )
    new_args = {**base_args, **dm_args}

    # set a specific budget for sparsemap at test time
    if sparsemap_budget is not None:
        new_args['sparsemap_budget'] = sparsemap_budget

    args = get_args_from_ckpt(ckpt_path, new_args)

    # fix cf_explainer_mask_token_type_id
    if hasattr(args, 'explainer_mask_token_type_id') and args.explainer_mask_token_type_id == '':
        args.explainer_mask_token_type_id = None
    if hasattr(args, 'cf_explainer_mask_token_type_id') and args.cf_explainer_mask_token_type_id == '':
        args.cf_explainer_mask_token_type_id = None

    pprint(vars(args))
    pprint(dm_args)

    # set global seed
    configure_seed(args.seed)

    # load tokenizer and model
    tokenizer, _, model = load(args)

    # factual model
    if factual_path is not None:
        print("Loading factual rationalizer from {}...".format(factual_path))
        factual_state_dict = load_torch_object(factual_path)['state_dict']
        model.load_state_dict(factual_state_dict, strict=False)

    # load data module
    dm_cls = available_data_modules[dm_name]
    dm = dm_cls(d_params=dm_args, tokenizer=tokenizer)
    dm.prepare_data()
    dm.setup()
    
    # predict
    model.generation_mode = True
    model.log_rationales_in_wandb = False
    trainer = Trainer(accelerator='gpu', devices=1)
    if dataloader == 'train':
        outputs = trainer.predict(model, dm.train_dataloader(shuffle=False))
    elif dataloader == 'val':
        outputs = trainer.predict(model, dm.val_dataloader())
    else:
        outputs = trainer.predict(model, dm.test_dataloader())

    # empty cache (beam search uses a lot of caching)
    torch.cuda.empty_cache()

    # stack outputs
    outputs = {k: [x[k] for x in outputs] for k in outputs[0].keys()}

    if return_tokenizer:
        return outputs, tokenizer
    return outputs


def save_edits(
    fname,
    orig_texts,
    orig_labels,
    orig_predictions,
    orig_z,
    edits_texts,
    edits_labels,
    edits_predictions,
    edits_z_pre,
    edits_z_pos,
):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        print(f'Creating {dirname} directory.')
        os.makedirs(dirname)

    df = pd.DataFrame({
        'orig_texts': orig_texts,
        'orig_labels': orig_labels,
        'orig_predictions': orig_predictions,
        'orig_z': [z_.detach().cpu().tolist() for z_ in orig_z],
        'edits_texts': edits_texts,
        'edits_labels': edits_labels,
        'edits_predictions': edits_predictions,
        'edits_z_pre': [z_.detach().cpu().tolist() for z_ in edits_z_pre],
        'edits_z_pos': [z_.detach().cpu().tolist() for z_ in edits_z_pos],
    })
    df.to_csv(fname, sep='\t', index=False)
    print('Saved to:', fname)


def get_edits(
    ckpt_path,
    factual_path,
    dm_name,
    dm_dataloader,
    dm_args,
    cf_generate_kwargs,
    sparsemap_budget=None
):
    # set seed
    configure_seed(0)

    # empty cache (beam search uses a lot of caching)
    torch.cuda.empty_cache()

    # load tokenizer, data, and the model, and then get predictions for a specified dataloader ('train', 'val', 'test')
    outputs, tokenizer = predict(
        ckpt_path,
        factual_path,
        cf_generate_kwargs,
        dm_name,
        dm_args,
        dataloader=dm_dataloader,
        verbose=True,
        disable_progress_bar=False,
        return_tokenizer=True,
        sparsemap_budget=sparsemap_budget
    )

    # get originals
    orig_texts = tokens_to_text(unroll(outputs['texts']))
    orig_labels = unroll(outputs['labels'])
    orig_predictions = torch.cat(outputs['predictions']).argmax(dim=-1).tolist()  # predictions for original inputs
    orig_z = unroll(outputs['z'])  # the z given to the original input by the rationalizer

    # get edits
    edits_texts = tokens_to_text(unroll(outputs['edits']))
    edits_labels = unroll(outputs['edits_labels'])
    edits_predictions = torch.cat(outputs['edits_predictions']).argmax(dim=-1).tolist()  # predictions for edits
    edits_z_pre = unroll(outputs['edits_z'])  # before passing through the rationalizer to mask tokens-to-be-edited
    edits_z_pos = unroll(outputs['edits_z_pos'])  # the z given to the edit by the rationalizer

    return {
        'orig_texts': orig_texts,
        'orig_labels': orig_labels,
        'orig_predictions': orig_predictions,
        'orig_z': orig_z,
        'edits_texts': edits_texts,
        'edits_labels': edits_labels,
        'edits_predictions': edits_predictions,
        'edits_z_pre': edits_z_pre,
        'edits_z_pos': edits_z_pos
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-name", type=str, help="Name used to save edits.", required=True)
    parser.add_argument("--ckpt-path", type=str, help="Path to the editor checkpoint.", required=True)
    parser.add_argument("--ckpt-path-factual", type=str, help="Path to the factual rationalizer ckpt.", default=None)
    parser.add_argument("--dm-name", type=str, help="Name of the data module.", required=True)
    parser.add_argument("--dm-dataloader", type=str, help="Name of the dataloader to use.", default='test')
    parser.add_argument("--batch-size", type=int, help="Batch size.", default=16)
    parser.add_argument("--num-beams", type=int, help="Number of beams to use for beam search.", default=15)
    parser.add_argument("--do-sample", action='store_true', help="Whether to use sampling instead of beam search.")
    parser.add_argument("--sparsemap-budget", type=int, help="Budget for sparsemap.", default=None)
    parser.add_argument("--ignore-neutrals", action='store_true', help="Whether to ignore neutral examples.")
    parser.add_argument("--random-subset-dirpath", type=str, help="Path to the dir of a subset of a dataset.", default=None)
    args = parser.parse_args()

    ckpt_name = args.ckpt_name
    ckpt_path = args.ckpt_path
    factual_path = args.ckpt_path_factual
    dm_name = args.dm_name
    dm_dataloader = args.dm_dataloader
    batch_size = args.batch_size
    num_beams = args.num_beams
    do_sample = args.do_sample
    sparsemap_budget = args.sparsemap_budget
    ignore_neutrals = args.ignore_neutrals
    random_subset_dirpath = args.random_subset_dirpath
    
    dm_args = dict(
        batch_size=batch_size,
        max_seq_len=512,
        num_workers=1,
        vocab_min_occurrences=1,
        is_original=True,
        max_dataset_size=None,
        ignore_neutrals=ignore_neutrals,
        path=random_subset_dirpath,
    )
    cf_generate_kwargs = dict(
        do_sample=do_sample,
        num_beams=num_beams,
        num_beam_groups=1,
        early_stopping=True,
        length_penalty=1.0,
        top_k=50,
        top_p=0.9,
        typical_p=None,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
        min_length=None,
        max_length=512,
    )

    out_dict = get_edits(
        ckpt_path,
        factual_path,
        dm_name,
        dm_dataloader,
        dm_args,
        cf_generate_kwargs,
        sparsemap_budget=sparsemap_budget
    )

    sample_mode = 'sample' if cf_generate_kwargs['do_sample'] else 'beam'
    num_beams = cf_generate_kwargs['num_beams']

    if factual_path is None:
        filename = f'data/edits/{dm_name}_{dm_dataloader}_{sample_mode}_{num_beams}_{ckpt_name}.tsv'
    else:
        filename = f'data/edits/{dm_name}_{dm_dataloader}_{sample_mode}_{num_beams}_{ckpt_name}_factual.tsv'

    save_edits(
        filename,
        out_dict['orig_texts'],
        out_dict['orig_labels'],
        out_dict['orig_predictions'],
        out_dict['orig_z'],
        out_dict['edits_texts'],
        out_dict['edits_labels'],
        out_dict['edits_predictions'],
        out_dict['edits_z_pre'],
        out_dict['edits_z_pos']
    )

    # compute accuracy
    y_pred = np.array(out_dict['orig_predictions'])
    y_gold = np.array(out_dict['orig_labels'])
    y_edit_pred = np.array(out_dict['edits_predictions'])
    y_edit_gold = np.array(out_dict['edits_labels'])
    print('Orig acc:', np.mean(y_pred == y_gold))
    print('Edit acc:', np.mean(y_edit_pred == y_edit_gold))
    print('Cont acc:', np.mean(y_edit_pred != y_gold))
