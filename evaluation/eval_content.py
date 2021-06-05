import argparse
import bert_score
import numpy as np

import logging
import transformers

transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)


BERTSCORE_MODEL = 'roberta-large'


def run_bert_score(hyp_file, ref_file):
    with open(hyp_file) as f:
        hyps = f.readlines()
    with open(ref_file) as f:
        refs = f.readlines()

    p, r, f = bert_score.score(hyps, refs, model_type=BERTSCORE_MODEL)
    return np.mean(f.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file')
    parser.add_argument('--ref-file')
    args = parser.parse_args()

    bs_result = run_bert_score(args.input_file, args.ref_file)

    print(str(bs_result.item()))


if __name__ == '__main__':
    main()