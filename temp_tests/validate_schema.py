import os
from datasets import load_dataset

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

def save_dataset(data, filename, convert_to_jsonl=False):
    if convert_to_jsonl:
        with open(filename, 'w') as file:
            for obj in data:
                file.write(json.dumps(obj) + '\n')
    else:
        with open(filename, 'w') as file:
            json.dump(data, file, indent=2)
            
def is_ascii(s):
    return all(ord(c) < 128 for c in s)

dir = "/input/file/dir/path" # <Update file directory here>
dataset_path = os.path.join(dir,"file_name.json") # <Update file name here>
dataset = load_dataset("json", data_files=dataset_path, num_proc=os.cpu_count())
print(dataset)
print(dataset["train"].features)