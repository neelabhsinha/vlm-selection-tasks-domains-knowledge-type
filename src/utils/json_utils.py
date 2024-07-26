import json


def save_dict_to_json(file_path, dictionary):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4)
    except Exception as e:
        print(f"An error occurred while saving the dictionary to JSON: {e}")


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
