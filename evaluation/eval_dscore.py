import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score-file')
    args = parser.parse_args()

    with open(args.score_file) as f:
        scores = f.readlines()

    scores = [float(score.strip()) for score in scores]
    print('avg scores: {:.5f}'.format(np.mean(scores)))


if __name__ == '__main__':
    main()