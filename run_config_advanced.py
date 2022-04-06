import json
import os
import json
import os
from sklearn.model_selection import ParameterGrid
import sys
import subprocess
import shlex
import time


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def initialize_grid_search():
    param_grid = {"weights_1": [1, 1.5, 2, 4, 6],
                  "weights_2": [1],
                  "learning_rate": [3e-5],
                  "num_epochs": [3]}

    configs = list(ParameterGrid(param_grid))
    log = {'current_config': 0}
    ensure_dir("/home/bijlesjvl/settings/")

    with open('/home/bijlesjvl/settings/configs.json', 'w') as outfile:
        json.dump(configs, outfile)

    with open('/home/bijlesjvl/settings/log.json', 'w') as outfile:
        json.dump(log, outfile)
    return configs, log


def get_search():
    settings_folder = '/home/bijlesjvl/settings/'
    if os.path.exists(settings_folder + 'configs.json') and os.path.exists(settings_folder + 'log.json'):
        print("EXISTING SEARCH EXISTS, LOAD AND CONTINUE...")

        with open(settings_folder + 'configs.json') as json_file:
            configs = json.load(json_file)

        with open(settings_folder + 'log.json') as json_file:
            log = json.load(json_file)
    else:
        print("CREATE NEW GRID SEARCH...")
        [configs, log] = initialize_grid_search()

    return configs, log


def initialize_data(test):
    if test.lower() == "true" and not os.path.exists('/home/bijlesjvl/data/finetuning/StockTwits_test/'):
        print("GET TEST DATA")
        ensure_dir("/home/bijlesjvl/data/finetuning/")
        os.system("gsutil -m cp -r gs://thesis-tpu/data/StockTwits_test /home/bijlesjvl/data/finetuning/")
    elif os.path.exists('/home/bijlesjvl/data/finetuning/StockTwits/'):
        print("DATA ALREADY EXITS")
    else:
        ensure_dir("/home/bijlesjvl/data/finetuning/")
        os.system("gsutil -m cp -r gs://thesis-tpu/data/StockTwits /home/bijlesjvl/data/finetuning/")


def initialize_model(mode):
    if os.path.exists(f"/home/bijlesjvl/model/CryptoBERT_{mode}/pretrained"):
        print("MODEL ALREADY EXITS")
    else:
        ensure_dir(f"/home/bijlesjvl/model/CryptoBERT_{mode}/pretrained/")
        if mode == "FIN":
            os.system("gsutil -m cp \
                        gs://thesis-tpu/model/CryptoBERT_FIN/CryptoBERT_FIN/* \
                        /home/bijlesjvl/model/CryptoBERT_FIN/pretrained/")
        elif mode == "TW":
            os.system("gsutil -m cp \
                        gs://thesis-tpu/model/CryptoBERT_TW_pretrained_2/CryptoBERT_TW/* \
                        /home/bijlesjvl/model/CryptoBERT_TW/pretrained/")


def initialize_scripts():
    ensure_dir("/home/bijlesjvl/scripts/")
    if os.path.exists("/home/bijlesjvl/scripts/cryptobert_tw_gcp_weighting_search.py"):
        os.system("rm /home/bijlesjvl/scripts/cryptobert_tw_gcp_weighting_search.py")
    os.system(
        "wget https://raw.githubusercontent.com/sreuvers/nlp-training/main/cryptobert_tw_gcp_weighting_search.py -P /home/bijlesjvl/scripts/ -q")
    os.system(
        "wget https://raw.githubusercontent.com/sreuvers/nlp-training/main/get_command.py -P /home/bijlesjvl/settings/ -q")


def run_command(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc


if __name__ == "__main__":
    [configs, log] = get_search()

    print("INITIALIZE DATA: ")

    mode = sys.argv[1]
    if len(sys.argv) > 2:
        test = sys.argv[2]
    else:
        test = 'false'
    initialize_data(test)
    if mode == "FIN":
        print(f"SELECTED MODE IS: {mode}")
        os.environ["PATH_MODEL"] = "/home/bijlesjvl/model/CryptoBERT_FIN/pretrained/"
        os.environ["MODEL_NAME"] = "CryptoBERT_FIN_fine-tuned"
        if test.lower() == 'true':
            print("TEST MODE ON")
            os.environ["PATH_DATA"] = "/home/bijlesjvl/data/finetuning/StockTwits_test/"
        else:
            os.environ["PATH_DATA"] = "/home/bijlesjvl/data/finetuning/StockTwits/"
        os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib"
        os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
        PATH_OUTPUT = "/home/bijlesjvl/model/CryptoBERT_FIN_fine-tuned/"
    elif mode == "Bertweet":
        print(f"SELECTED MODE IS: {mode}")
        os.environ["PATH_MODEL"] = "vinai/bertweet-base"
        os.environ["MODEL_NAME"] = "Bertweet"
        os.environ["PATH_DATA"] = "/home/bijlesjvl/data/finetuning/StockTwits/"
        os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib"
        os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
        PATH_OUTPUT = "/home/bijlesjvl/model/Bertweet_fine-tuned/"
    elif mode == "TW":
        print(f"SELECTED MODE IS: {mode}")
        os.environ["PATH_MODEL"] = "/home/bijlesjvl/model/CryptoBERT_TW/pretrained/"
        os.environ["MODEL_NAME"] = "CryptoBERT_TW_fine-tuned"
        os.environ["PATH_DATA"] = "/home/bijlesjvl/data/finetuning/StockTwits/"
        os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib"
        os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
        PATH_OUTPUT = "/home/bijlesjvl/model/CryptoBERT_TW_fine-tuned/"
    else:
        print(f"SELECTED MODE: {mode} IS NOT RECOGNIZED")
        raise SystemExit()

    print("INITIALIZE MODEL: ")
    initialize_model(mode)
    print(f"STARTING CONFIGS FROM {log['current_config']}")

    print("INITIALIZE SCRIPTS: ")
    initialize_scripts()
    commands = []
    for config in configs:
        RUN_NAME = ""
        for key, value in config.items():
            if key == list(config.keys())[-1]:
                RUN_NAME = RUN_NAME + f"{key}_{value}"
            else:
                RUN_NAME = RUN_NAME + f"{key}_{value}_"

        os.environ["RUN_NAME"] = RUN_NAME
        os.environ["PATH_OUTPUT"] = PATH_OUTPUT + RUN_NAME

        ensure_dir(PATH_OUTPUT)
        ensure_dir(PATH_OUTPUT + RUN_NAME + "/")


        command = f"python3 scripts/cryptobert_tw_gcp_weighting_search.py \
                    --model_name={os.environ['MODEL_NAME']} \
                    --path_output={os.environ['PATH_OUTPUT']} \
                    --path_data={os.environ['PATH_DATA']} \
                    --path_model={os.environ['PATH_MODEL']} \
                    --weights_1=%s \
                    --weights_2=%s \
                    --epochs='3' \
                    --train_batch_size='128' \
                    --eval_batch_size='128' \
                    --learning_rate='5e-5' \
                    --warmup_steps='300'" % (config['weights_1'], config['weights_2'])
        commands.append(command)
    with open('/home/bijlesjvl/settings/commands.json', 'w') as outfile:
        json.dump(commands, outfile)