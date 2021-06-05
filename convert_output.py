import argparse
import os
from fairseq.data.encoders.gpt2_bpe import GPT2BPE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-dir', nargs='+')
    args = parser.parse_args()

    bpe = GPT2BPE(None).bpe

    for sp in ['valid', 'test']:
        for generate_dir in args.generate_dir:

            all_samples = []
            sample = []
            if not os.path.exists(os.path.join(generate_dir, 'generate-{}.txt'.format(sp))):
                continue
            with open(os.path.join(generate_dir, 'generate-{}.txt'.format(sp))) as f:
                for line in f:
                    if line[0] == 'S':
                        if sample:
                            all_samples.append((sample_id, sample))
                            sample = []
                        sample_id, sent = line.split('\t')
                        sample_id = sample_id.split('-')[1]
                        sent = bpe.decode([
                            int(tok) if tok not in {'<unk>', '<mask>'} else tok
                            for tok in sent.split()
                        ])
                        sample.append(sent)
                    elif line[0] == 'T':
                        sent = line.split('\t')[1]
                        sent = bpe.decode([
                            int(tok) if tok not in {'<unk>', '<mask>'} else tok
                            for tok in sent.split()
                        ])
                        sample.append(sent)
                    elif line[0] == 'H':
                        sent = line.split('\t')[-1]
                        sent = bpe.decode([
                            int(tok) if tok not in {'<unk>', '<mask>'} else tok
                            for tok in sent.split()
                        ])
                        sample.append(sent)
            if sample:
                all_samples.append((sample_id, sample))
                sample = []

            all_samples = sorted(all_samples, key=lambda x: int(x[0]))
            all_samples = [x[1] for x in all_samples]

            with open(os.path.join(generate_dir, 'formatted-{}.txt'.format(sp)), 'w') as f:
                for sample in all_samples:
                    f.write(sample[2] + '\n')


if __name__ == '__main__':
    main()