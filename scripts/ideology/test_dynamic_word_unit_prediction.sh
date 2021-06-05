BEAM_SIZE=5
MAX_LEN_B=75
MIN_LEN=3
LEN_PEN=2.0
MAX_SENTENCES=64

DATA_PATH=$1
BASE_MODEL=$2
STYLE_MODEL=$3
WU_BASE_MODEL=$4
RESULT_PATH=$5


fairseq-generate $DATA_PATH \
    --results-path $RESULT_PATH \
    --source-lang source --target-lang target \
    --task inference_style_control \
    --style-model $STYLE_MODEL \
    --path $BASE_MODEL \
    --wu-base-model $WU_BASE_MODEL \
    --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN \
    --no-repeat-ngram-size 3 \
    --max-sentences $MAX_SENTENCES \
    --required-batch-size-multiple 1 \
    --truncate-source \
    --word-unit --stride 3;
