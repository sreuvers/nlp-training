import os
from sklearn.model_selection import ParameterGrid

param_grid = {"weights_1" : [8,7,6,5,4.5,4,3.5,3,2.5,2,1.5,1],
            "weights_2" : [1],
            "learning_rate": [3e-5],
            "num_epochs": [2]}

configs = list(ParameterGrid(param_grid))
os.environ["PATH_MODEL"] = "vinai/bertweet-base"
os.environ["MODEL_NAME"] = "Bertweet_fine-tuned"
os.environ["PATH_DATA"] = "/home/bijlesjvl/data/finetuning/StockTwits_test/"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib"
os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
PATH_OUTPUT = "/home/bijlesjvl/model/Bertweet_fine-tuned/"

for config in configs:

    RUN_NAME = ""
    for key, value in config.items():
        if key == list(config.keys())[-1]:
            RUN_NAME = RUN_NAME + f"{key}_{value}"
        else:
            RUN_NAME = RUN_NAME + f"{key}_{value}_"
    print("START NEW RUN")
    print(f"RUN NAME: {RUN_NAME}")

    os.environ["RUN_NAME"] = RUN_NAME
    os.environ["PATH_OUTPUT"] = PATH_OUTPUT + RUN_NAME

    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    ensure_dir(PATH_OUTPUT)
    ensure_dir(PATH_OUTPUT + RUN_NAME + "/")

    os.system("python3 cryptobert_tw_gcp_weighting_search.py \
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
    print("FINISHED RUN, CONTINUE........")



