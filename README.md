# Inference Time Style Control for Summarization

Code for NAACL 2021 paper "Inference Time Style Control for Summarization".

## Run our code

The PyTorch version used by our code is `1.4.0`.

Please also install our modified Fairseq library:

```shell
cd fairseq
pip install -e .
```

Our models and binarized data can be downloaded [here](https://drive.google.com/drive/folders/1EVGoFOAvWMDkN8P9JIO7kd2sz8qSoRJb?usp=sharing).

For word unit prediction, please also download the RoBERTa-base model from [here](https://github.com/pytorch/fairseq/tree/master/examples/roberta).

To convert the outputs generated by scripts below to text, please run `convert_output.py --generate-dir $OUTPUT_DIR`.

### Simplicity-controlled summarization on CNN/DailyMail

##### Discriminative scorer

```shell
./script/simplicity/test_decoder_state_adjustment_scorer.sh \
  $DOWNLOAD_PATH/data/binarized_cnndm \
  $DOWNLOAD_PATH/models/summarizers/cnndm_summarizer/model.pt \
  $DOWNLOAD_PATH/models/simplicity_models/discriminator \
  $OUTPUT_DIR
```

##### Class-conditional language model scorer

```shell
./script/simplicity/test_decoder_state_adjustment_cclm.sh \
  $DOWNLOAD_PATH/data/binarized_cnndm \
  $DOWNLOAD_PATH/models/summarizers/cnndm_summarizer/model.pt \
  $DOWNLOAD_PATH/models/simplicity_models/cclm \
  $OUTPUT_DIR
```

##### Word unit prediction

```shell
./script/simplicity/test_word_unit_prediction.sh \
  $DOWNLOAD_PATH/data/binarized_cnndm \
  $DOWNLOAD_PATH/models/summarizers/cnndm_summarizer/model.pt \
  $DOWNLOAD_PATH/models/simplicity_models/wu_predictor \
  $ROBERTA_DIR \
  $OUTPUT_DIR
```

##### Dynamic word unit prediction

```shell
./script/simplicity/test_dynamic_word_unit_prediction.sh \
  $DOWNLOAD_PATH/data/binarized_cnndm \
  $DOWNLOAD_PATH/models/summarizers/cnndm_summarizer/model.pt \
  $DOWNLOAD_PATH/models/simplicity_models/dynamic_wu_predictor \
  $ROBERTA_DIR \
  $OUTPUT_DIR
```

### Ideology-controlled headline generation on SemEval

##### Discriminative scorer

```shell
./script/ideology/test_decoder_state_adjustment_scorer.sh \
  $DOWNLOAD_PATH/data/binarized_semeval \
  $DOWNLOAD_PATH/models/summarizers/semeval_summarizer/model.pt \
  $DOWNLOAD_PATH/models/ideology_models/discriminator/<left/right> \
  $OUTPUT_DIR
```

##### Class-conditional language model scorer

To generate left-leaning headlines, set `TARGET_CLASS=177`;
to generate right-leaning headlines, set `TARGET_CLASS=179`.

```shell
./script/ideology/test_decoder_state_adjustment_cclm.sh \
  $DOWNLOAD_PATH/data/binarized_semeval \
  $DOWNLOAD_PATH/models/summarizers/semeval_summarizer/model.pt \
  $DOWNLOAD_PATH/models/ideology_models/cclm \
  $TARGET_CLASS \
  $OUTPUT_DIR
```

##### Word unit prediction

```shell
./script/ideology/test_word_unit_prediction.sh \
  $DOWNLOAD_PATH/data/binarized_semeval \
  $DOWNLOAD_PATH/models/summarizers/semeval_summarizer/model.pt \
  $DOWNLOAD_PATH/models/ideology_models/wu_predictor \
  $ROBERTA_DIR \
  $OUTPUT_DIR
```

##### Dynamic word unit prediction

```shell
./script/ideology/test_dynamic_word_unit_prediction.sh \
  $DOWNLOAD_PATH/data/binarized_semeval \
  $DOWNLOAD_PATH/models/summarizers/semeval_summarizer/model.pt \
  $DOWNLOAD_PATH/models/ideology_models/dynamic_wu_predictor \
  $ROBERTA_DIR \
  $OUTPUT_DIR
```

-------

## Train and run your/our models

To train your own summarizer, please follow the [instruction](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md) provided by BART.

##### Discriminative scorer

Please follow the [instruction](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.custom_classification.md) provided by RoBERTa to train your discriminative scorer.

After training, organize the directory for storing the model as:

```shell
model.pt
input0/dict.txt
label/dict.txt
```

##### Class-conditional language model scorer

Please follow the [instruction](https://github.com/pytorch/fairseq/tree/master/examples/language_model) provided by Fairseq to train your class-conditional language model.

To construct the data for training the class-conditional language model,
you need to prepend the style label to each training sample after applying BPE preprocessing.
`cclm_data/build_cclm_data_example.py` provides an example for this process.

After training, organize the directory for storing the model as:

```shell
model.pt
dict.txt
```

##### (Dynamic) word unit predictor

To train (dynamic) word unit predictors, you need to prepare `<data_split>.source` and `<data_split>.target`.
Each line in the `source` file is the input for a sample and each line in the `target` file contains the ground-truth predicted words.

Similar to the BPE preprocessing for BART, preprocess the data for word unit prediction with:

```shell
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

for LANG in source target
do
python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$SPLIT.$LANG" \
    --outputs "$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
done
```

Binarize the data with:

```shell
fairseq-preprocess --source-lang "source" --target-lang "target" \
  --trainpref "train.bpe" --validpref "valid.bpe" --destdir "." \
  --workers 60 --srcdict dict.txt --tgtdict dict.txt
```

Then train the predictor with:

```shell
CUDA_VISIBLE_DEVICES=0 ./scripts/train_wu_predictors/train_word_unit_predictors.sh \
  $DATA_DIR \
  $ROBERTA_BASE_DIR \
  $MODEL_SAVE_DIR
```
