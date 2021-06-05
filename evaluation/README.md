# Evaluation

Codes for evaluating the generated summaries and headlines.

Requirements:

```
textstat==0.7.0
nltk==3.5
transformers==4.4.2
bert-score==0.3.8
```

##### Discriminative style score

```shell
python discriminative_score.py \
  --input-file <generated_text_file> \
  --discriminator-path <discriminator_path> \
  --head-name <head_name> \
  --out-file <evaluated_score_file>
```

```
[Simplicity]
<discriminator_path>: $DOWNLOAD_PATH/models/simplicity_models/discriminator
<head_name>: simplicity_head

[Ideology]
<discriminator_path>: $DOWNLOAD_PATH/models/ideology_models/discriminator/<left/right>
<head_name>: media_head
```

To get the average score:

```shell
python eval_dscore.py --score-file <score_file>
```

##### Readability

```shell
python eval_readability.py --input-file <generated_text_file>
```

##### Perplexity

```shell
python eval_ppl.py --input-file <generated_text_file>
```

##### Edit distance

We use `Stanford CoreNLP 3.9.2` to tokenize the generated summaries and headlines.

```shell
python eval_diff.py \
  --tokenized-input1 <tokenized_text_file1> \
  --tokenized-input2 <tokenized_text_file2>
```

##### BERT score

```shell
python eval_content.py \
  --input-file <generated_text_file> \
  --ref-file <reference_text_file>
```
