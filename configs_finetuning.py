import json
import os
from sklearn.model_selection import ParameterGrid

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

param_grid = {"weights_1": [4],
              "weights_2": [1],
              "learning_rate": [3e-5],
              "num_epochs": [3]}

configs = list(ParameterGrid(param_grid))
jsonString = json.dumps(configs)
ensure_dir("/home/bijlesjvl/settings/")

with open('/home/bijlesjvl/settings/configs.json', 'w') as outfile:
    json.dump(configs, outfile)

log = {'current_config' : 0}

with open('/home/bijlesjvl/settings/log.json', 'w') as outfile:
    json.dump(log, outfile)
