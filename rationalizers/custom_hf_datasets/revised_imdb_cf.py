from __future__ import absolute_import, division, print_function

import os

import datasets
import pandas as pd

from rationalizers.custom_hf_datasets.revised_imdb import _CITATION, _DESCRIPTION, _URL


class CountefactualRevisedIMDBDataset(datasets.GeneratorBasedBuilder):
    """Movie reviews from the IMDB dataset revised by Kaushik et al. (2020)."""
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = datasets.BuilderConfig
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="revised_imdb_cf_dataset",
            description="Movie reviews from the IMDB dataset revised by Kaushik et al. (2020)",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "tokens": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=["Negative", "Positive"]),
                    "batch_id": datasets.Value("int32"),
                    "is_original": datasets.Value("bool"),
                    "cf_tokens": datasets.Value("string"),
                    "cf_label":  datasets.ClassLabel(names=["Negative", "Positive"]),
                    "cf_batch_id":  datasets.Value("int32"),
                    "cf_is_original": datasets.Value("bool"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/acmi-lab/counterfactually-augmented-data",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "imdb")
        filepaths = {
            "train": os.path.join(data_dir, "train.tsv"),
            "dev": os.path.join(data_dir, "dev.tsv"),
            "test": os.path.join(data_dir, "test.tsv"),
        }
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": filepaths["train"], "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": filepaths["dev"], "split": "dev"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": filepaths["test"], "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        df = pd.read_csv(filepath, delimiter='\t')
        for i, (_, g) in enumerate(df.groupby('batch_id')):
            yield i, {
                # the first input is the original review
                "tokens": g['text'].iloc[0],
                "label": g['gold_label'].iloc[0],
                "batch_id": g['batch_id'].iloc[0],
                "is_original": g['is_original'].iloc[0] == 1,
                # the second input is the counterfactual
                "cf_tokens": g['text'].iloc[1],
                "cf_label": g['gold_label'].iloc[1],
                "cf_batch_id": g['batch_id'].iloc[1],
                "cf_is_original": g['is_original'].iloc[1] == 1,
            }
