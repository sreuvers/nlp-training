import json
import os
import sys

settings_folder = '/home/bijlesjvl/settings/'

with open(settings_folder + 'commands.json') as json_file:
    commands = json.load(json_file)

with open(settings_folder + 'log.json') as json_file:
    log = json.load(json_file)

log['current_config'] = int(sys.argv[1]) + 1
with open(settings_folder + 'log.json', 'w') as outfile:
    json.dump(log, outfile)
print(commands[int(sys.argv[1])])
