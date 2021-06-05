import argparse
import os


NORMAL_LABEL = '124'
SIMPLE_LABEL = '125'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainpref')
    parser.add_argument('--validpref')
    parser.add_argument('--testpref')
    parser.add_argument('--outdir')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.trainpref + '.source') as f:
        train_source = f.readlines()
    with open(args.trainpref + '.target') as f:
        train_target = f.readlines()
    with open(os.path.join(args.outdir, 'train.bpe.source'), 'w') as f:
        for src, tgt in zip(train_source, train_target):
            f.write(NORMAL_LABEL + ' ' + src)
            f.write(SIMPLE_LABEL + ' ' + tgt)

    with open(args.validpref + '.source') as f:
        valid_source = f.readlines()
    with open(args.validpref + '.target') as f:
        valid_target = f.readlines()
    with open(os.path.join(args.outdir, 'valid.bpe.source'), 'w') as f:
        for src, tgt in zip(valid_source, valid_target):
            f.write(NORMAL_LABEL + ' ' + src)
            f.write(SIMPLE_LABEL + ' ' + tgt)

    with open(args.testpref + '.source') as f:
        test_source = f.readlines()
    with open(args.testpref + '.target') as f:
        test_target = f.readlines()
    with open(os.path.join(args.outdir, 'test.bpe.source'), 'w') as f:
        for src, tgt in zip(test_source, test_target):
            f.write(NORMAL_LABEL + ' ' + src)
            f.write(SIMPLE_LABEL + ' ' + tgt)


if __name__ == '__main__':
    main()