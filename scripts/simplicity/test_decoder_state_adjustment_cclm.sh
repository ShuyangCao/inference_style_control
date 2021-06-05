BEAM_SIZE=1
MAX_LEN_B=140
MIN_LEN=55
LEN_PEN=2.0
MAX_SENTENCES=64

STEP_SIZE=0.1
ITERATION=10
CCLM_TARGET_CLASS=125
FUTURE_WINDOW=2

DATA_PATH=$1
BASE_MODEL=$2
STYLE_MODEL=$3
RESULT_PATH=$4

fairseq-generate $DATA_PATH \
    --path $BASE_MODEL --results-path $RESULT_PATH \
    --source-lang source --target-lang target \
    --task inference_style_control \
    --style-model $STYLE_MODEL \
    --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN \
    --no-repeat-ngram-size 3 \
    --max-sentences $MAX_SENTENCES \
    --required-batch-size-multiple 1 \
    --truncate-source \
    --decoder-state --cclm-token --step-size $STEP_SIZE --iteration $ITERATION --future-window $FUTURE_WINDOW \
    --cclm-target-class $CCLM_TARGET_CLASS --ori-task;
