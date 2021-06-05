BEAM_SIZE=1
MAX_LEN_B=75
MIN_LEN=3
LEN_PEN=2.0
MAX_SENTENCES=64

STEP_SIZE=1.0
ITERATION=30
TARGET_LABEL=1
STOP_SCORE=1.0
FUTURE_WINDOW=2

HEAD_NAME="media_head"

DATA_PATH=$1
BASE_MODEL=$2
STYLE_MODEL=$3
RESULT_PATH=$4

fairseq-generate $DATA_PATH \
    --path $BASE_MODEL --results-path $RESULT_PATH --head-name $HEAD_NAME \
    --target-label 1 --style-dict $LABEL_DICT \
    --source-lang source --target-lang target \
    --task inference_style_control \
    --style-model $STYLE_MODEL \
    --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN \
    --no-repeat-ngram-size 3 \
    --max-sentences $MAX_SENTENCES \
    --required-batch-size-multiple 1 \
    --truncate-source \
    --decoder-state --token-step --step-size $STEP_SIZE --iteration $ITERATION --future-window $FUTURE_WINDOW \
    --stop-score $STOP_SCORE;
