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

# Save dataset
def save_dataset(data, filename, convert_to_jsonl=False):
    if convert_to_jsonl:
        with open(filename, 'w') as file:
            for obj in data:
                file.write(json.dumps(obj) + '\n')
    else:
        with open(filename, 'w') as file:
            json.dump(data, file, indent=2)

dir = os.environ["DATA_MGT"] # MAKE SURE TO SET THIS ENVIRONMENT VARIABLE
dataset="input_file_name.jsonl" # UPDATE INPUT FILE NAME
new_dataset = "input_file_name_fix.jsonl" # UPDATE OUTPUT FILE NAME
dataset_path = os.path.join(dir,dataset)
output_path= os.path.join(dir, new_dataset)

jsonlist = load_dataset_from_file(dataset_path)
print(f"Values in json: {len(jsonlist)}")

def get_key_info(idx, key, keytype):
    correct_type = isinstance(item[key], keytype)
    if item[key] is not None and not correct_type:
        print(f"idx: {idx}")
        print(key)
        print(f"type: {type(item[key])}")
        print(f"value: {item[key]}")
        
def change_empty_lst_to_none(idx, item, key):
    change_to_null = False
    if isinstance(item[key], list):
        if len(item[key]) <=0 or isinstance(item[key][0], dict):
            change_to_null = not item[key][0] # If dict is empty -> [{}]
            if change_to_null:
                print(f"must update {idx}: {key}") #TEMP
                print(item[key])
    return change_to_null

def get_item(idx, item, correct_type):
    correct_type = isinstance(item, correct_type)
    if not correct_type:
        print(f"idx: {idx}")
        print(f"type: {type(item)}")
        print(f"value: {item}")
    return correct_type

print("=Type errors at level 1=")
corrections_l1 = 0
for idx, item in zip(range(len(jsonlist)), jsonlist):
    # Level 1 - check level 1 keys datatype
    corrected = False
    # get_key_info(idx, 'intent', list)
    # get_key_info(idx, 'knowledge', list)
    # get_key_info(idx, 'difficulty', list)
    # get_key_info(idx, 'input_quality', list)
    # get_key_info(idx, 'input_quality_explanation', list)
    # get_key_info(idx, 'task_category', list)
    # get_key_info(idx, 'judge_quality_score', list)
    # get_key_info(idx, 'judge_quality_explanation', list)
    
    if change_empty_lst_to_none(idx, item, 'intent') or change_empty_lst_to_none(idx, item, 'knowledge') or change_empty_lst_to_none(idx, item, 'difficulty'):
        item['intent'] = None
        item['knowledge'] = None
        item['difficulty'] = None
        item['metadata']['label_model'] = None
        corrected = True        
    if change_empty_lst_to_none(idx, item, 'input_quality') or change_empty_lst_to_none(idx, item, 'input_quality_explanation'):
        item['input_quality'] = None
        item['input_quality_explanation'] = None
        item['metadata']['label_model'] = None
        corrected = True    
    if change_empty_lst_to_none(idx, item, 'task_category'):
        item['task_category'] = None
        item['metadata']['label_model'] = None
        corrected = True    
    if change_empty_lst_to_none(idx, item, 'judge_quality_score') or change_empty_lst_to_none(idx, item, 'judge_quality_explanation'):
        item['judge_quality_score'] = None
        item['judge_quality_explanation'] = None
        item['metadata']['label_model'] = None
        corrected = True
    
    if corrected: 
        corrections_l1 += 1
print(f"\nCorrections at level 1 made to {corrections_l1} instances.")

        
print("\n=Type errors at level 2=")
corrections_l2 = 0
for idx, item in zip(range(len(jsonlist)), jsonlist):
    # Level 2 - check score keys, datatype must be str or None
    corrected = False
    if item['difficulty'] is not None:
        correct_type = get_item(idx, item['difficulty'][0]['mistralai/Mistral-7B-Instruct-v0.3'], str)
        if not correct_type: 
            item['intent'] = None
            item['knowledge'] = None
            item['difficulty'] = None
            item['metadata']['label_model'] = None
            corrected = True
    if item['input_quality'] is not None:
        correct_type = get_item(idx, item['input_quality'][0]['mistralai/Mistral-7B-Instruct-v0.3'], str)
        if not correct_type:
            item['input_quality'] = None
            item['input_quality_explanation'] = None
            item['metadata']['label_model'] = None
            corrected = True
    if item['task_category'] is not None:
        correct_type = get_item(idx, item['task_category'][0]['mistralai/Mistral-7B-Instruct-v0.3'], str)
        if not correct_type:
            item['task_category'] = None
            item['metadata']['label_model'] = None
            corrected = True
    if item['judge_quality_score'] is not None:
        correct_type = get_item(idx, item['judge_quality_score'][0]['mistralai/Mistral-7B-Instruct-v0.3'], str)
        if not correct_type: 
            item['judge_quality_score'] = None
            item['judge_quality_explanation'] = None
            item['metadata']['label_model'] = None
            corrected = True

    if corrected: 
        corrections_l2 += 1

print(f"\nCorrections at level 2 made to {corrections_l2} instances.")

# Loading dataset test
updated_dataset = Dataset.from_list(jsonlist)
print(f"\n=Datase loaded=")
print(updated_dataset.features)

# Save fixed dataset
save_dataset(updated_dataset, output_path, convert_to_jsonl=True)
print(f"File saved at: {output_path}")