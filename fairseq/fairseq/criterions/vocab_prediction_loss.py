# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils, metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('vocab_prediction_loss')
class VocabPredictionLossCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])  # bsz x vsz
        logits = net_output
        target = model.get_targets(sample, net_output)  # bsz x len

        def convert_target(target):
            new_target = F.one_hot(target, len(self.task.target_dictionary)).sum(dim=1)  # bsz x vsz
            new_target = (new_target > 0).float()
            new_target = new_target[:, self.task.target_dictionary.nspecial:]
            return new_target

        target = convert_target(target)

        loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none', pos_weight=target.new_zeros(1).fill_(50))

        ntokens = loss.numel()

        if reduce:
            loss = loss.sum()

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': ntokens,
            'nsentences': logits.size(0),
            'sample_size': ntokens,
        }
        return loss, ntokens, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: round(2 ** meters['nll_loss'].avg, 3))
        else:
            metrics.log_derived('ppl', lambda meters: round(2 ** meters['loss'].avg, 3))
