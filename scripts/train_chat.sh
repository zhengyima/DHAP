DEVICES='0'
DATA_DIR='./data'
EMB_FILE='/home/zhengyi_ma/sgns.weibo.bigram-char-withheader'
RESULT_FILE='./res.txt'
# Start training
python runModel.py \
        --device $DEVICES \
        --bidirectional \
        --use_attn \
        --random_seed 2808 \
        --src_vocab_size 40000 \
        --tgt_vocab_size 40000 \
        --data_path $DATA_DIR \
        --word2vec_path $EMB_FILE \
        --result_path $RESULT_FILE
