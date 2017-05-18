import json

def create_json(settings_file, json_dict):
    with open(settings_file, 'w') as json_file:
        json.dump(json_dict, json_file, sort_keys=True, indent=4)