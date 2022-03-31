import os

os.environ["PATH_MODEL"]="vinai/bertweet-base"
os.environ["MODEL_NAME"]="Bertweet_fine-tuned"
os.environ["RUN_NAME"]="weights_3_08"
os.environ["PATH_OUTPUT"]="/home/bijlesjvl/model/Bertweet_fine-tuned/" + "weights_3_08"
os.environ["PATH_DATA"]="/home/bijlesjvl/data/finetuning/StockTwits_test/"
os.environ["LD_LIBRARY_PATH"]="/usr/local/lib"
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir("/home/bijlesjvl/model/Bertweet_fine-tuned/")
ensure_dir("/home/bijlesjvl/model/Bertweet_fine-tuned/" + "weights_3_08")

os.system("cryptobert_tw_gcp_weighting \
    --model_name=$MODEL_NAME \
    --path_output=$PATH_OUTPUT \
    --path_data=$PATH_DATA \
    --path_model=$PATH_MODEL \
    --weights_1='3' \
    --weights_2='2' \
    --epochs='3' \
    --train_batch_size='128' \
    --eval_batch_size='128' \
    --learning_rate='5e-5' \
    --warmup_steps='100'")


# os.system("cryptobert_tw_gcp_weighting \
#     --model_name='test' \
#     --path_output='test' \
#     --path_data='test' \
#     --path_model='test' \
#     --weights_1='3' \
#     --weights_2='2' \
#     --epochs='3' \
#     --train_batch_size='128' \
#     --eval_batch_size='128' \
#     --learning_rate='5e-5' \
#     --warmup_steps='100'")

