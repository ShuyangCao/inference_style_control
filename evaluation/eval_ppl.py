from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch

import argparse
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input--file')
    parser.add_argument('--gpt2-path', default='gpt2')
    args = parser.parse_args()

    with open(args.input_file) as f:
        hyps = f.readlines()
    print('Data loaded.')

    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_path)
    model = GPT2LMHeadModel.from_pretrained(args.gpt2_path)

    model.eval()
    model.cuda()

    print('Model loaded.')

    ppls = []

    with torch.no_grad():
        for hyp in tqdm(hyps):
            tokens = tokenizer.tokenize(hyp.strip())
            tokens = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).cuda()
            model_output = model(tokens, labels=tokens)
            ppl = model_output['loss'].exp().item()
            ppls.append(ppl)

    print(np.mean(ppls))


if __name__ == '__main__':
    main()
