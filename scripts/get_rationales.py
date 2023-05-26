import argparse
import os
from pprint import pprint

import datasets as hf_datasets
import pandas as pd
import torch
from pytorch_lightning import Trainer

from rationalizers.data_modules import available_data_modules
from rationalizers.utils import unroll
from utils import configure_seed, load, get_args_from_ckpt, tokens_to_text


# import warnings
# warnings.filterwarnings("ignore")


def predict(
    ckpt_path,
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
        save_rationales=False,
        save_edits=False,
        cf_classify_edits=False,
        sparsemap_budget=30,
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

    # load data module
    dm_cls = available_data_modules[dm_name]
    dm = dm_cls(d_params=dm_args, tokenizer=tokenizer)
    dm.prepare_data()
    dm.setup()
    
    # predict
    model.generation_mode = True
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


def save_rationales(fname, tokens, labels, predictions, z):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        print(f'Creating {dirname} directory.')
        os.makedirs(dirname)

    df = pd.DataFrame({
        'texts': tokens,
        'labels': labels,
        'predictions': predictions,
        'z': [z_.detach().cpu().tolist() for z_ in z],
    })
    df.to_csv(fname, sep='\t', index=False)
    print('Saved to:', fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-name", type=str, help="Name used to save edits.", required=True)
    parser.add_argument("--ckpt-path", type=str, help="Path to the editor checkpoint.", required=True)
    parser.add_argument("--ckpt-path-factual", type=str, help="Path to the factual rationalizer ckpt.", default=None)
    parser.add_argument("--dm-name", type=str, help="Name of the data module.", required=True)
    parser.add_argument("--dm-dataloader", type=str, help="Name of the dataloader to use.", default='test')
    parser.add_argument("--batch-size", type=int, help="Batch size.", default=16)
    parser.add_argument("--sparsemap-budget", type=int, help="Budget for sparsemap.", default=None)
    parser.add_argument("--ignore-neutrals", action='store_true', help="Whether to ignore neutral examples.")
    args = parser.parse_args()

    dm_args = dict(
        batch_size=args.batch_size,
        max_seq_len=512,
        num_workers=1,
        vocab_min_occurrences=1,
        is_original=True,
        max_dataset_size=None,
        ignore_neutrals=args.ignore_neutrals,
    )
    
    # load tokenizer, data, and the model, and then get predictions for a specified dataloader ('train', 'val', 'test')
    outputs, tokenizer = predict(
        args.ckpt_path,
        args.dm_name,
        dm_args, 
        dataloader=args.dm_dataloader,
        verbose=True,
        disable_progress_bar=False,
        return_tokenizer=True,
        sparsemap_budget=args.sparsemap_budget
    )

    # get originals
    orig_texts = tokens_to_text(unroll(outputs['texts']))
    orig_labels = unroll(outputs['labels'])
    orig_predictions = torch.cat(outputs['predictions']).argmax(dim=-1).tolist()  # predictions for original inputs
    orig_z = unroll(outputs['z'])  # the z given to the original input by the rationalizer

    # compute accuracy
    y_pred = torch.tensor(orig_predictions)
    y_gold = torch.tensor(orig_labels)
    print('Orig acc:', (y_pred == y_gold).float().mean().item())

    # save everything
    save_rationales(
        f'data/rationales/{args.dm_name}_{args.dm_dataloader}_{args.ckpt_name}.tsv',
        orig_texts,
        orig_labels,
        orig_predictions, 
        orig_z, 
    )
