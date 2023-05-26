# Useful scripts

This folder contains the scripts to extract rationales and counterfactuals from the trained models.

Before proceeding, make sure to install the requirements in the root folder of the repository.


## Extracting rationales

If all you want is to extract rationales from a trained model, you can use the `get_rationales.py` script.

For example, for IMDB, run:

```bash
python3 scripts/get_rationales.py \
    --ckpt-name "foo" \
    --ckpt-path "path/to/masker.ckpt" \
    --dm-name "revised_imdb" \
    --dm-dataloader "test"
```

The rationales will be saved in `data/rationales/{dm_name}_{dm_dataloader}_{ckpt_name}.tsv`.


## Extracting counterfactuals

To extract counterfactuals, use the script `get_edits.py`. 

For example, for IMDB, run:

```bash
python3 scripts/get_edits.py \
  --ckpt-name "foo" \
  --ckpt-path "path/to/editor.ckpt" \
  --dm_name "imdb" \
  --dm-dataloader "train" \
  --num-beams 15
```

The counterfactuals will be saved in `data/edits/{dm_name}_{dm_dataloader}_beam_{num_beams}_{ckpt_name}.tsv`

**Note:** For SNLI, you may want to inform the `--ignore-neutrals` flag to ignore neutral examples.

The tsv file will contain the following columns:

| Column             | Description                                                                                     |
|--------------------|-------------------------------------------------------------------------------------------------|
| orig_texts         | The tokenized original input texts                                                              |
| orig_labels        | The original labels (can be empty if the dataset does not have labels)                          |
| orig_predictions   | The predictions made by the original rationalizer                                               |
| orig_z             | The rationales (vector of scores) given by the original rationalizer for the **original** input |
| edits_texts        | The tokenized counterfactual texts                                                              |
| edits_labels       | The counterfactual labels prepended to the input                                                |
| edits_predictions  | The predictions made by the rationalizer on the counterfactuals                                 |
| edits_z_pre        | A binary vector indicating which tokens were marked as `<mask>` to be infilled by the editor    |
| edits_z_pos        | The rationales given by the original rationalizer for the **counterfactual** input              |


## Extracting counterfactuals with a different masker

To use a different masker than the one that was used to train the editor, you can inform its checkpoint via the
`--ckpt-path-factual` flag of the `get_edits.py` script. 

For example, for IMDB, run:

```bash
python3 scripts/get_edits.py \
  --ckpt-name "foo" \
  --ckpt-path "path/to/editor.ckpt" \
  --ckpt-path-factual "path/to/another/masker.ckpt" \
  --dm-name "imdb" \
  --dm-dataloader "train" \
  --num-beams 15
```

The counterfactuals will be saved in `data/edits/{dm_name}_{dm_dataloader}_beam_{num_beams}_{ckpt_name}_factual.tsv`.

For more information, run `python3 scripts/get_edits.py --help`.


## Training students for forward simulation

For IMDB:
```bash
python3 scripts/forward_simulation.py \
    --student-type bow \
    --train-data data/edits/imdb_train_beam_15_t5_small_30p.tsv \
    --test-data data/edits/revised_imdb_test_beam_15_t5_small_30p.tsv \
    --batch-size 16 \
    --epochs 10 \
    --seed 0
```

For SNLI just change the `--student-type` to `bow_nli` and the `--train-data` and `--test-data` to the SNLI files.


