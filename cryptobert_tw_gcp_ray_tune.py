import cryptobert_tw_gcp_weighting_search.py
import os

PATH_MODEL="vinai/bertweet-base"
MODEL_NAME="Bertweet_fine-tuned"
RUN_NAME="weights_3_08"
PATH_OUTPUT="/home/bijlesjvl/model/Bertweet_fine-tuned/" + RUN_NAME
PATH_DATA="/home/bijlesjvl/data/finetuning/StockTwits_test/"
os.mkdir(PATH_OUTPUT)

os.system("cryptobert_tw_gcp_weighting \
    --model_name=MODEL_NAME \
    --path_output=PATH_OUTPUT\
    --path_data=PATH_DATA\
    --path_model=PATH_MODEL \
    --epochs='3'\
    --train_batch_size='128' \
    --eval_batch_size='128' \
    --learning_rate='5e-5' \
    --weights_1='3' \
    --weights_2='2' \
    --warmup_steps='100'")

