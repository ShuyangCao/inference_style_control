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

from fairseq import metrics, options, utils, tasks, search, hub_utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    InferenceStyleControlDataset,
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
    truncate_source=False, word_unit=False
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

    selected_vocabs = None
    if word_unit:
        with open(prefix + 'entityvocab') as f:
            entity_vocabs = f.readlines()
        entity_vocabs = [[int(x) for x in line.strip().split(' ') if x] for line in entity_vocabs]
        selected_vocabs = entity_vocabs

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    return InferenceStyleControlDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        selected_vocabs=selected_vocabs
    )


@register_task('inference_style_control')
class InferenceStyleControlTask(FairseqTask):
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

        method_group = parser.add_mutually_exclusive_group(required=True)
        method_group.add_argument('--decoder-state', action='store_true', default=False,
                                  help='enable decoder state adjustment')
        method_group.add_argument('--word-unit', action='store_true', default=False,
                                  help='enable word unit prediction')

        parser.add_argument('--style-model', default=None, metavar='STYLE MODEL',
                            help='path to the style model')

        # decoder state adjustment arguments
        ds_group = parser.add_argument_group('Decoder State Adjustment')
        ds_group.add_argument('--token-step', action='store_true')
        ds_group.add_argument('--cclm-token', action='store_true')
        ds_group.add_argument('--step-size', default=0.01, type=float, metavar='N',
                           help='step size for gradient update')
        ds_group.add_argument('--iteration', default=1, type=int, metavar='N',
                           help='number of update iteration')
        ds_group.add_argument('--future-window', default=1, type=int, metavar='N',
                           help='number of future steps to predict')
        ds_group.add_argument('--target-label', default="1", type=str, metavar='L',
                           help='the linguistic style target label')
        ds_group.add_argument('--cclm-target-class')
        ds_group.add_argument('--stop-score', default=1.0, type=float)
        ds_group.add_argument('--head-name', type=str)

        # word unit prediction arguments
        wu_group = parser.add_argument_group('Word Unit Prediction')
        wu_group.add_argument('--wu-base-model')
        wu_group.add_argument('--stride', default=0, type=int)

        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, style_dict, style_model, wu_base_model):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.style_dict = style_dict
        self.style_model = style_model
        self.wu_base_model = wu_base_model

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

        # style_dict = None
        # if args.style_dict is not None:
        #     assert os.path.exists(args.style_dict)
        #     style_dict = cls.load_dictionary(args.style_dict)
        #     logger.info('style dictionary: {} types'.format(len(style_dict)))

        # load style model
        assert args.style_model is not None
        assert os.path.exists(args.style_model)

        use_cuda = torch.cuda.is_available() and not args.cpu

        hub_model = hub_utils.from_pretrained(
            args.style_model,
            checkpoint_file='model.pt',
            data_name_or_path='.',
            load_checkpoint_heads=True
        )

        style_model = hub_model['models'][0]
        try:
            style_dict = hub_model['task'].label_dictionary
        except:
            style_dict = None

        style_model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            style_model.half()
        if use_cuda:
            style_model.cuda()

        style_model.eval()

        for p in style_model.parameters():
            p.requires_grad_(False)

        wu_base_model = None
        if args.word_unit:
            from fairseq.models.roberta import RobertaModel
            wu_base_model = RobertaModel.from_pretrained(args.wu_base_model)
            if args.fp16:
                wu_base_model.half()
            if use_cuda:
                wu_base_model.cuda()

            wu_base_model.eval()

        return cls(args, src_dict, tgt_dict, style_dict, style_model, wu_base_model)

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
            truncate_source=self.args.truncate_source,
            word_unit=self.args.word_unit,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return InferenceStyleControlDataset(src_tokens, src_lengths, self.source_dictionary)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        raise NotImplementedError

    def valid_step(self, sample, model, criterion):
        raise NotImplementedError

    def build_generator(self, args):
        sampling = getattr(args, 'sampling', False)
        sampling_topk = getattr(args, 'sampling_topk', -1)
        sampling_topp = getattr(args, 'sampling_topp', -1.0)
        diverse_beam_groups = getattr(args, 'diverse_beam_groups', -1)
        diverse_beam_strength = getattr(args, 'diverse_beam_strength', 0.5),
        match_source_len = getattr(args, 'match_source_len', False)
        diversity_rate = getattr(args, 'diversity_rate', -1)
        if (
                sum(
                    int(cond)
                    for cond in [
                        sampling,
                        diverse_beam_groups > 0,
                        match_source_len,
                        diversity_rate > 0,
                    ]
                )
                > 1
        ):
            raise ValueError('Provided Search parameters are mutually exclusive.')
        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert sampling_topp < 0 or sampling, '--sampling-topp requires --sampling'

        if sampling:
            search_strategy = search.Sampling(self.target_dictionary, sampling_topk, sampling_topp)
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(self.target_dictionary, diversity_rate)
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        if getattr(args, 'decoder_state', False):
            if args.token_step:
                from fairseq.decoder_state_adjustment_sequence_generator_token import StateAdjustSequenceGenerator
            elif args.cclm_token:
                from fairseq.decoder_state_adjustment_sequence_generator_cclm import StateAdjustSequenceGenerator
            else:
                raise NotImplementedError

            if args.cclm_token:
                target_label = getattr(args, 'cclm_target_class')
                logger.info('[Decoder State Adjustment] [CCLM] target class {}'.format(target_label))
                target_label = self.target_dictionary.index(target_label)
            elif args.token_step:
                target_label = getattr(args, 'target_label', None)
                logger.info('[Decoder State Adjustment] [Scorer] target label {}'.format(target_label))
                if target_label is not None:
                    target_label = self.style_dict.index(target_label) - self.style_dict.nspecial
            else:
                raise NotImplementedError

            return StateAdjustSequenceGenerator(
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                temperature=getattr(args, 'temperature', 1.),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                search_strategy=search_strategy,
                step_size=getattr(args, 'step_size', 0.01),
                iteration=getattr(args, 'iteration', 0),
                future_window=getattr(args, 'future_window', 1),
                target_label=target_label,
                stop_score=getattr(args, 'stop_score', 1.0),
                head_name=getattr(args, 'head_name', None)
            )
        elif getattr(args, 'word_unit', False):
            from fairseq.word_unit_sequence_generator import WordUnitSequenceGenerator
            return WordUnitSequenceGenerator(
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                temperature=getattr(args, 'temperature', 1.),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                search_strategy=search_strategy,
                stride=getattr(args, 'stride', 1)
            )
        else:
            raise NotImplementedError

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        if getattr(self.args, 'decoder_state', False):
            return generator.generate(models, sample, self.style_model, prefix_tokens=prefix_tokens)
        elif getattr(self.args, 'word_unit', False):
            with torch.no_grad():
                return generator.generate(models, sample, self.style_model, self.wu_base_model, prefix_tokens=prefix_tokens)
        else:
            raise NotImplementedError

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
