from fairseq.models.roberta import RobertaModel

import torch

import os
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file')
    parser.add_argument('--discriminator-path')
    parser.add_argument('--head-name', choices=['media_head', 'simplicity_head'])
    parser.add_argument('--out-file')
    args = parser.parse_args()

    with open(args.input_file) as f:
        hypos = f.readlines()

    roberta = RobertaModel.from_pretrained(
        os.path.join(args.discriminator_path),
        checkpoint_file='model.pt',
        data_name_or_path='.'
    )

    roberta.eval()
    roberta.cuda()

    label = roberta.task.label_dictionary.index('1') - roberta.task.label_dictionary.nspecial
    scores = []

    with torch.no_grad():
        for hypo in tqdm(hypos):
            tokens = roberta.encode(hypo.strip())
            tokens = tokens.cuda()
            logits = roberta.predict(args.head_name, tokens)
            score = torch.softmax(logits, dim=-1)[0, label].item()
            scores.append(score)

    with open(args.out_file, 'w') as f:
        f.writelines(['{:.5f}\n'.format(score) for score in scores])


if __name__ == '__main__':
    main()