WARMUP_UPDATES=500
LR=1e-4
MAX_SENTENCES=32
UPDATE_FREQ=1

DATA_PATH=$1
WU_BASE_MODEL=$2
SAVE_PATH=$3

fairseq-train $DATA_PATH \
    --base-model $BASE_MODEL --roberta-base \
    --max-sentences $MAX_SENTENCES \
    --task hidden_state_vocab_prediction \
    --source-lang source --target-lang target \
    --truncate-source \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch hidden_state_vocab_predictor \
    --dropout 0.1 --input-dim 768 \
    --criterion vocab_prediction_loss \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 --fp16 \
    --lr-scheduler polynomial_decay --lr $LR --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --validate-interval 1 \
    --num-workers 0 \
    --save-dir $SAVE_PATH \
    --no-epoch-checkpoints --no-last-checkpoints \
    --no-save-optimizer-state \
    --patience 2;