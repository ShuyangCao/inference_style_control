# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import json
import itertools
import logging
import os

import numpy as np
import torch

from fairseq import metrics, options, utils, tasks, search
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    iterators,
    FairseqDataset,
    indexed_dataset,
    VocabPredictionDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from fairseq.tasks import FairseqTask, register_task

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


def load_langpair_dataset(
        data_path, split,
        src, src_dict,
        tgt, tgt_dict,
        combine, dataset_impl,
        left_pad_source, left_pad_target, max_source_positions,
        max_target_positions, prepend_bos=False,
        truncate_source=False
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    # infer langcode
    if split_exists(split, src, tgt, src, data_path):
        prefix = os.path.join(data_path, '{}.{}-{}.'.format(split, src, tgt))
    elif split_exists(split, tgt, src, src, data_path):
        prefix = os.path.join(data_path, '{}.{}-{}.'.format(split, tgt, src))
    else:
        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

    src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
    if truncate_source:
        src_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDataset(src_dataset, src_dict.eos()),
                max_source_positions - 1,
            ),
            src_dict.eos(),
        )
    tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)

    logger.info('{} {} {}-{} {} examples'.format(
        data_path, split, src, tgt, len(src_dataset)
    ))

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    return VocabPredictionDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions
    )


@register_task('hidden_state_vocab_prediction')
class HiddenStateVocabPredictionTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--dynamic-vp', action='store_true')

        # special PP arguments
        parser.add_argument('--base-model', default=None, metavar='BASE MODEL',
                            help='path to the base model')
        parser.add_argument('--roberta-base', action='store_true')

        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, base_model):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.base_model = base_model

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = args.data.split(os.pathsep)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        # load base model
        # assert args.base_model is not None
        # assert os.path.exists(args.base_model)

        if args.base_model is None or args.base_model == '':
            base_model = None
        else:
            use_cuda = torch.cuda.is_available() and not args.cpu

            from fairseq.checkpoint_utils import load_checkpoint_to_cpu
            if args.roberta_base:
                from fairseq.models.roberta import RobertaModel
                base_model = RobertaModel.from_pretrained(args.base_model)
            elif os.path.isdir(args.base_model):
                from fairseq.models.bart import BARTModel
                base_model = BARTModel.from_pretrained(args.base_model)
            else:
                base_model_state = load_checkpoint_to_cpu(args.base_model)
                base_model_args = base_model_state['args']

                base_model_task = tasks.setup_task(base_model_args)
                base_model = base_model_task.build_model(base_model_args)
                base_model.load_state_dict(base_model_state['model'], strict=True)
                base_model.make_generation_fast_()
            if args.fp16:
                base_model.half()
            if use_cuda:
                base_model.cuda()

            base_model.eval()

        return cls(args, src_dict, tgt_dict, base_model)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            self.args.data, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            truncate_source=self.args.truncate_source
        )

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        assert isinstance(dataset, FairseqDataset)
        if not self.args.dynamic_vp:
            if dataset in self.dataset_to_epoch_iter:
                return self.dataset_to_epoch_iter[dataset]
        else:
            dataset.randomize_prefix_length()

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # randomize tgt prefix length


        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices, dataset, max_positions, raise_exception=(not ignore_invalid_inputs),
            )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            drop_last=True
        )
        self.dataset_to_epoch_iter[dataset] = epoch_iter
        return epoch_iter

    def build_model(self, args):
        return super().build_model(args)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        with torch.no_grad():
            if self.args.roberta_base:
                src_tokens = sample['net_input']['src_tokens']
                src_tokens = utils.convert_padding_direction(src_tokens, self.source_dictionary.pad(),
                                                             left_to_right=True)
                if src_tokens.size(1) < 512:
                    encoder_feature = self.base_model.extract_features(src_tokens)
                else:
                    splits = []
                    for i in range(src_tokens.size(1) // 256 - 1):
                        split_i = src_tokens[:, i * 256: (i + 2) * 256]
                        split_i = self.base_model.extract_features(split_i)
                        if i == 0:
                            splits.append(split_i)
                        else:
                            splits.append(split_i[:, 256:])
                    encoder_feature = torch.cat(splits, dim=1)
            else:
                src_tokens = sample['net_input']['src_tokens']
                src_tokens = utils.convert_padding_direction(src_tokens, self.source_dictionary.pad(),
                                                             left_to_right=True)
                encoder_feature = self.base_model.extract_features(src_tokens)
            sample['net_input'] = {
                'features': encoder_feature,
                'lengths': sample['net_input']['src_lengths'].tolist()
            }
        return super().train_step(sample, model, criterion, optimizer, ignore_grad)

    def valid_step(self, sample, model, criterion):
        with torch.no_grad():
            if self.args.roberta_base:
                src_tokens = sample['net_input']['src_tokens']
                src_tokens = utils.convert_padding_direction(src_tokens, self.source_dictionary.pad(),
                                                             left_to_right=True)
                if src_tokens.size(1) < 512:
                    encoder_feature = self.base_model.extract_features(src_tokens)
                else:
                    splits = []
                    for i in range(src_tokens.size(1) // 256 - 1):
                        split_i = src_tokens[:, i * 256: (i + 2) * 256]
                        split_i = self.base_model.extract_features(split_i)
                        if i == 0:
                            splits.append(split_i)
                        else:
                            splits.append(split_i[:, 256:])
                    encoder_feature = torch.cat(splits, dim=1)
            else:
                src_tokens = sample['net_input']['src_tokens']
                src_tokens = utils.convert_padding_direction(src_tokens, self.source_dictionary.pad(),
                                                             left_to_right=True)
                encoder_feature = self.base_model.extract_features(src_tokens)
            sample['net_input'] = {
                'features': encoder_feature,
                'lengths': sample['net_input']['src_lengths'].tolist()
            }
        return super().valid_step(sample, model, criterion)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return 2048

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
