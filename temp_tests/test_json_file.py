import os
from datasets import load_dataset
from datasets import Dataset

import json
import pandas as pd

# File I/O utilities
def load_jsonl_to_list(jsonl_file_path):
    data_list = []
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            data_list.append(json_obj)
    return data_list

# Load dataset
def load_dataset_from_file(filename):
    #if the file is json
    if filename.endswith('.json'):
        with open(filename, 'r') as file:
            return json.load(file)
    elif filename.endswith('.jsonl'):
        return load_jsonl_to_list(filename)
    else:
        raise ValueError("Invalid file format. Please provide a .json or .jsonl file.")

dataset_path="input/file/path.jsonl" # MUST UPDATE
jsonlist = load_dataset_from_file(dataset_path)
print(f"Values in json: {len(jsonlist)}")

# Load the dataset
updated_dataset = Dataset.from_list(jsonlist)
print("=schema=")
print(updated_dataset.features) 