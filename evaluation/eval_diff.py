import argparse
import os

import editdistance
import numpy as np
from nltk import jaccard_distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenized-input1')
    parser.add_argument('--tokenized-input2')
    args = parser.parse_args()

    with open(args.tokenized_input1) as f:
        input1s = f.readlines()

    with open(args.tokenized_input2) as f:
        input2s = f.readlines()

    eds = []
    jds = []
    for input1, input2 in zip(input1s, input2s):
        eds.append(editdistance.eval(input1.strip().split(' '), input2.strip().split(' ')))
        jds.append(jaccard_distance(set(input1.strip().split(' ')), set(input2.strip().split(' '))))

    print('edit distance:', np.mean(eds))
    print('jaccard distance:', np.mean(jds))


if __name__ == '__main__':
    main()