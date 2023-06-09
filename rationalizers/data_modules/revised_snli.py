from functools import partial
from itertools import chain

import datasets as hf_datasets
import nltk
import numpy as np
import torch
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor
from torchnlp.utils import collate_tensors
from transformers import PreTrainedTokenizerBase

from rationalizers import constants
from rationalizers.data_modules.base import BaseDataModule


class RevisedSNLIDataModule(BaseDataModule):
    """DataModule for the Revised SNLI Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        """
        :param d_params: hyperparams dict. See docs for more info.
        """
        super().__init__(d_params)
        # hard-coded stuff
        self.path = "./rationalizers/custom_hf_datasets/revised_snli.py"
        self.is_multilabel = True
        self.nb_classes = 3  # entailment, neutral, contradiction

        # hyperparams
        self.side = d_params.get("side", "premise")
        self.batch_size = d_params.get("batch_size", 64)
        self.num_workers = d_params.get("num_workers", 0)
        self.vocab_min_occurrences = d_params.get("vocab_min_occurrences", 1)
        self.max_seq_len = d_params.get("max_seq_len", 99999999)
        self.max_dataset_size = d_params.get("max_dataset_size", None)
        self.is_original = d_params.get("is_original", None)
        self.concat_inputs = d_params.get("concat_inputs", True)
        self.swap_pair = d_params.get("swap_pair", False)
        self.filter_neutrals = d_params.get("filter_neutrals", False)
        self.ignore_neutrals = d_params.get("ignore_neutrals", False)
        if self.filter_neutrals:
            self.nb_classes = 2

        # objects
        self.dataset = None
        self.label_encoder = None
        self.tokenizer = tokenizer
        self.tokenizer_cls = partial(
            # WhitespaceEncoder,
            # TreebankEncoder,
            StaticTokenizerEncoder,
            tokenize=nltk.wordpunct_tokenize,
            min_occurrences=self.vocab_min_occurrences,
            reserved_tokens=[
                constants.PAD,
                constants.UNK,
                constants.EOS,
                constants.SOS,
                "<copy>",
            ],
            padding_index=constants.PAD_ID,
            unknown_index=constants.UNK_ID,
            eos_index=constants.EOS_ID,
            sos_index=constants.SOS_ID,
            append_sos=False,
            append_eos=False,
        )
        self.sep_token = self.tokenizer.sep_token or self.tokenizer.eos_token
        self.sep_token_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id

    def _collate_fn(self, samples: list, are_samples_batched: bool = False):
        """
        :param samples: a list of dicts
        :param are_samples_batched: in case a batch/bucket sampler are being used
        :return: dict of features, label (Tensor)
        """
        if are_samples_batched:
            # dataloader batch size is 1 -> the sampler is responsible for batching
            samples = samples[0]

        # convert list of dicts to dict of lists
        collated_samples = collate_tensors(samples, stack_tensors=list)

        # pad and stack input ids
        def pad_and_stack_ids(x, pad_id=constants.PAD_ID):
            x_ids, x_lengths = stack_and_pad_tensors(x, padding_index=pad_id)
            return x_ids, x_lengths

        def stack_labels(y):
            if isinstance(y, list):
                return torch.stack(y, dim=0)
            return y

        if self.concat_inputs:
            input_ids, lengths = pad_and_stack_ids(collated_samples["input_ids"])
            token_type_ids, _ = pad_and_stack_ids(collated_samples["token_type_ids"], pad_id=2)

            # stack labels
            labels = stack_labels(collated_samples["label"])

            # keep tokens in raw format
            prem_tokens = collated_samples["prem"]
            hyp_tokens = collated_samples["hyp"]

            if not self.swap_pair:
                tokens = [p.strip() + ' ' + self.sep_token + ' ' + h.strip() for p, h in zip(prem_tokens, hyp_tokens)]
            else:
                tokens = [p.strip() + ' ' + self.sep_token + ' ' + h.strip() for p, h in zip(hyp_tokens, prem_tokens)]

            # metadata
            batch_id = collated_samples["batch_id"]
            is_original = collated_samples["is_original"]

            batch = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "lengths": lengths,
                "labels": labels,
                "tokens": tokens,
                "batch_id": batch_id,
                "is_original": is_original,
            }

        else:
            prem_ids, prem_lengths = pad_and_stack_ids(collated_samples["prem_ids"])
            hyp_ids, hyp_lengths = pad_and_stack_ids(collated_samples["hyp_ids"])

            # stack labels
            labels = stack_labels(collated_samples["label"])

            # keep tokens in raw format
            prem_tokens = collated_samples["prem"]
            hyp_tokens = collated_samples["hyp"]

            # metadata
            batch_id = collated_samples["batch_id"]
            is_original = collated_samples["is_original"]

            # return batch to the data loader
            batch = {
                "input_ids_1": prem_ids,
                "input_ids_2": hyp_ids,
                "lengths_1": prem_lengths,
                "lengths_2": hyp_lengths,
                "labels": labels,
                "tokens_1": prem_tokens,
                "tokens_2": hyp_tokens,
                "batch_id": batch_id,
                "is_original": is_original,
            }

        return batch

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
            side=self.side,
        )

        # cap dataset size - useful for quick testing
        if self.max_dataset_size is not None:
            self.dataset["train"] = self.dataset["train"].select(range(self.max_dataset_size))
            self.dataset["validation"] = self.dataset["validation"].select(range(self.max_dataset_size))
            self.dataset["test"] = self.dataset["test"].select(range(self.max_dataset_size))

        # build tokenize rand label encoder
        if self.tokenizer is None:
            # build tokenizer info (vocab + special tokens) based on train and validation set
            tok_samples = chain(
                self.dataset["train"]["prem"],
                self.dataset["train"]["hyp"],
                self.dataset["validation"]["prem"],
                self.dataset["validation"]["hyp"],
            )
            self.tokenizer = self.tokenizer_cls(tok_samples)

        # map strings to ids
        def _encode(example: dict):
            if self.concat_inputs:
                if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                    if not self.swap_pair:
                        input_enc = self.tokenizer(
                            example["prem"].strip(),
                            example["hyp"].strip(),
                            padding=False,  # do not pad, padding will be done later
                            truncation=True,  # truncate to max length accepted by the model
                        )
                    else:
                        input_enc = self.tokenizer(
                            example["hyp"].strip(),
                            example["prem"].strip(),
                            padding=False,  # do not pad, padding will be done later
                            truncation=True,  # truncate to max length accepted by the model
                        )
                    example["input_ids"] = input_enc["input_ids"]
                    if 'token_type_ids' in input_enc:
                        example["token_type_ids"] = torch.tensor(input_enc["token_type_ids"])
                    else:
                        example["token_type_ids"] = 1 - torch.cumprod(
                            torch.tensor(example["input_ids"]) != self.sep_token_id, dim=0)
                else:
                    if not self.swap_pair:
                        example["input_ids"] = self.tokenizer.encode(
                            example["prem"].strip() + ' ' + self.sep_token + ' ' + example["hyp"].strip()
                        )
                    else:
                        example["input_ids"] = self.tokenizer.encode(
                            example["hyp"].strip() + ' ' + self.sep_token + ' ' + example["prem"].strip()
                        )
                    example["token_type_ids"] = 1 - torch.cumprod(
                        torch.tensor(example["input_ids"]) != self.sep_token_id, dim=0)
            else:
                if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                    example["prem_ids"] = self.tokenizer(
                        example["prem"].strip(),
                        padding=False,  # do not pad, padding will be done later
                        truncation=True,  # truncate to max length accepted by the model
                    )["input_ids"]
                    example["hyp_ids"] = self.tokenizer(
                        example["hyp"].strip(),
                        padding=False,  # do not pad, padding will be done later
                        truncation=True,  # truncate to max length accepted by the model
                    )["input_ids"]
                else:
                    example["prem_ids"] = self.tokenizer.encode(example["prem"].strip())
                    example["hyp_ids"] = self.tokenizer.encode(example["hyp"].strip())
            return example

        self.dataset = self.dataset.map(_encode)

        if self.concat_inputs:
            self.dataset = self.dataset.filter(lambda example: len(example["input_ids"]) <= self.max_seq_len)
        else:
            self.dataset = self.dataset.filter(lambda example: len(example["prem_ids"]) <= self.max_seq_len)
            self.dataset = self.dataset.filter(lambda example: len(example["hyp_ids"]) <= self.max_seq_len)

        if self.is_original is not None:
            self.dataset = self.dataset.filter(lambda example: example["is_original"] == self.is_original)

        if self.filter_neutrals or self.ignore_neutrals:
            print('Filtering out neutrals')
            self.dataset = self.dataset.filter(lambda ex: ex["label"] != 1)

            if self.filter_neutrals:
                print('Fixing labels to be 0/1')
                self.dataset = self.dataset.map(lambda ex: min(ex['label'], 1))

        def get_dist(y):
            vals, counts = np.unique(y, return_counts=True)
            return dict(zip(vals, counts / counts.sum()))

        print(get_dist(self.dataset["train"]["label"]))
        print(get_dist(self.dataset["validation"]["label"]))
        print(get_dist(self.dataset["test"]["label"]))

        # convert `columns` to pytorch tensors and keep un-formatted columns
        if self.concat_inputs:
            self.dataset.set_format(
                type="torch",
                columns=[
                    "input_ids", "token_type_ids", "label",
                    "batch_id", "is_original"
                ],
                output_all_columns=True,
            )
        else:
            self.dataset.set_format(
                type="torch",
                columns=[
                    "prem_ids", "hyp_ids", "label",
                    "batch_id", "is_original"
                ],
                output_all_columns=True,
            )
