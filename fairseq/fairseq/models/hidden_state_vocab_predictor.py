import torch
import torch.nn as nn

import torch.nn.functional as F

from fairseq.models import (
    BaseFairseqModel,
    register_model_architecture,
    register_model
)


@register_model('hidden_state_vocab_predictor')
class RNNHiddenStateLabeler(BaseFairseqModel):
    def __init__(self, projection, dropout):
        super().__init__()
        self.projection = projection
        self.dropout = dropout

    @staticmethod
    def add_args(parser):
        parser.add_argument('--input-dim', type=int, metavar='N')
        parser.add_argument('--dropout', type=float, metavar='D')

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        target_vocab_size = len(task.target_dictionary) - task.target_dictionary.nspecial

        projection = nn.Sequential(
            ResidualBlock(args.input_dim, args.dropout),
            nn.Linear(args.input_dim, target_vocab_size)
        )

        return cls(projection, args.dropout)

    def forward(self, features, lengths=None):
        bsz, seqlen, _ = features.size()
        x = F.dropout(features, p=self.dropout, training=self.training)
        if lengths is not None:
            x = [torch.mean(x[i, :length], dim=0) for i, length in enumerate(lengths)]
            x = torch.stack(x, dim=0)
        else:
            x = torch.mean(x, dim=1)  # bsz x hidden

        logits = self.projection(x)
        return logits

    def get_targets(self, sample, net_output):
        return sample['target']


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual_x = x
        x = self.bn1(x)
        x = torch.tanh(x)
        x = self.linear1(x)
        x = self.bn2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x + residual_x


@register_model_architecture('hidden_state_vocab_predictor', 'hidden_state_vocab_predictor')
def base_architecture(args):
    args.input_dim = getattr(args, 'input_dim', 1024)
    args.dropout = getattr(args, 'dropout', args.dropout)
