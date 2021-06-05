import argparse
import textstat
import numpy as np

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file')
    args = parser.parse_args()

    with open(args.input_file) as f:
        hyps = f.readlines()

    dc_scores = []
    for hyp in tqdm(hyps):
        dc_score = textstat.dale_chall_readability_score(hyp.strip())
        dc_scores.append(dc_score)

    print('Dale Chall', np.mean(dc_scores))


if __name__ == '__main__':
    main()