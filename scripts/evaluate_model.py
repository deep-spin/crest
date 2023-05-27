import os
from argparse import Namespace
from pprint import pprint

import datasets as hf_datasets
import pandas as pd
from pytorch_lightning import Trainer

from rationalizers.data_modules import available_data_modules
from rationalizers.utils import load_ckpt_config
from utils import load


def get_args_from_ckpt(ckpt_args):
    base_args = dict(
        save_rationales=False,
        save_counterfactuals=False,
        load_tokenizer=False,
        load_label_encoder=False,
    )
    old_args = load_ckpt_config(ckpt_args['ckpt'])
    args = Namespace(**{**old_args, **base_args, **ckpt_args})
    return args


def evaluate_on_new_data_module(dm, model, verbose=True, disable_progress_bar=False, accelerator='gpu', devices=1):
    trainer = Trainer(accelerator=accelerator, devices=devices)

    # disable hf_dataset progress bar
    if disable_progress_bar:
        hf_datasets.logging.disable_progress_bar()
    else:
        hf_datasets.logging.enable_progress_bar()
    if verbose:
        hf_datasets.logging.set_verbosity(20)
    else:
        hf_datasets.logging.set_verbosity(50)

    # make sure to perform test on both factuals and counterfactuals
    dm.is_original = None

    # test
    output = trainer.test(model, datamodule=dm, verbose=verbose)
    return output[0]


def evaluate_all_checkpoints(args, dms, **kwargs):
    print('Evaluating model with seed {} from {}'.format(args.seed, args.ckpt))
    print('===' * 10)
    tokenizer, _, model = load(args)
    out_dict = {}
    for dm_name, dm_args in dms.items():
        print('Evaluating on dataset {}'.format(dm_name))
        print('===' * 10)
        dm_cls = available_data_modules[dm_name]
        dm = dm_cls(d_params=dm_args, tokenizer=tokenizer)
        out_dict[dm_name] = evaluate_on_new_data_module(dm, model, **kwargs)
    return out_dict


def tabulate_and_save_results(results, fname):
    d = {}
    for dm_name, dm_stats in results.items():
        if isinstance(dm_stats, dict):
            for k, v in dm_stats.items():
                d[f'{dm_name}_{k}'] = v
        else:
            for dm_stats_ in dm_stats:
                for k, v in dm_stats_.items():
                    d[f'{dm_name}_{k}'] = v

    print('Results:')
    pprint(d)

    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        print(f'Creating {dirname} directory.')
        os.makedirs(dirname)

    print('Saving results to:', fname)
    df = pd.DataFrame([d])
    df.to_csv(fname, index=False)


if __name__ == '__main__':

    eval_on_ood_data = False
    method_name = 'imdb_sparsemap_30p'
    args = get_args_from_ckpt({
        'seed': 0,
        'ckpt': 'experiments/masker_imdb_sparsemap_30p/versionjkexuek0/checkpoints/epoch=1.ckpt',
        'sparsemap_budget': 30
    })

    # in-domain datasets
    dms_id = {
        # IMDB datasets
        "revised_imdb": dict(batch_size=8, is_original=None),
        "contrast_imdb": dict(batch_size=8, is_original=None),
        "imdb": dict(batch_size=8, is_original=None),

        # SNLI datasets
        # "revised_snli":   dict(batch_size=128, is_original=None),
        # "snli":           dict(batch_size=128, is_original=None),
        # "hnli_e":         dict(batch_size=128, is_original=None, difficulty="easy"),
        # "hnli_h":         dict(batch_size=128, is_original=None, difficulty="hard"),
    }

    # out-of-domain datasets
    dms_ood = {
        # IMDB datasets
        "sst2": dict(batch_size=8, is_original=None),
        "rottom": dict(batch_size=8, is_original=None),
        "yelp": dict(batch_size=8, is_original=None),
        "amazon": dict(batch_size=8, is_original=None, max_dataset_size=25000),

        # SNLI datasets
        # "mnli_m":     dict(batch_size=128, is_original=None, domain='matched'),
        # "mnli_mm":    dict(batch_size=128, is_original=None, domain='mismatched'),
        # "anli":       dict(batch_size=128, is_original=None),
        # "wnli":       dict(batch_size=128, is_original=None),
        # "bnli":       dict(batch_size=128, is_original=None),
    }

    results = evaluate_all_checkpoints(
        args,
        dms=dms_id,
        verbose=False,
        disable_progress_bar=True,
        accelerator='gpu',
        devices=1
    )
    tabulate_and_save_results(results, 'results/results_in_domain_{}.csv'.format(method_name))

    if eval_on_ood_data:
        results = evaluate_all_checkpoints(
            args,
            dms=dms_ood,
            verbose=False,
            disable_progress_bar=True,
            accelerator='gpu',
            devices=1
        )
        tabulate_and_save_results(results, 'results/results_out_of_domain_{}.csv'.format(method_name))
