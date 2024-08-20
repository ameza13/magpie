import json
import os
import argparse
from datasets import load_dataset

"""
python /$WORKSPACE/magpie/exp/filter.py --input_dir /input/dir/path/ --output_dir /output/dir/path --dataset_name dataset_file_name.jsonl

"""
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
            
def filter_out_invalid_samples(example): 
    return (
        # Validate input and output
        example['input'] is not None
        and example['output'] is not None
        and len(example['input']) > 0
        and len(example['output']) > 0
        # Remove instances with null tags
        and example['input_quality'] is not None
        and example['judge_quality_score'] is not None
        and example['difficulty'] is not None
        and example['task_category'] is not None
        # and example['reward_quality_score'] is not None 
        )

# unpack labels assigned by one model ( mistralai/Mistral-7B-Instruct-v0.3) for simplicity
def unpack_scores(example):
    example["input_quality"] = example['input_quality'][0]['mistralai/Mistral-7B-Instruct-v0.3']
    example["input_quality_explanation"] = example['input_quality_explanation'][0]['mistralai/Mistral-7B-Instruct-v0.3']
    example["difficulty"] = example['difficulty'][0]['mistralai/Mistral-7B-Instruct-v0.3']
    example["judge_quality_score"] = example['judge_quality_score'][0]['mistralai/Mistral-7B-Instruct-v0.3']
    example["judge_quality_explanation"] = example['judge_quality_explanation'][0]['mistralai/Mistral-7B-Instruct-v0.3']
    example["task_category"] = example['task_category'][0]['mistralai/Mistral-7B-Instruct-v0.3']
    # example["reward_quality_score"] = example['reward_quality_score'][0]['blue/reward-model']
    return example

def validate_scores_format(example):
    return (
        example["input_quality"] in ['very poor','poor','average','good','excellent']
        and example["difficulty"] in ['very easy','easy','medium','hard','very hard']
        and example["judge_quality_score"] in ['1','2','3','4','5']
        and len(example["task_category"]) > 0
    )

def high_quality_filter(example):
   return (
       example['input_quality'] in ['good','excellent']
       and example['judge_quality_score'] in ['5']
       # and example['reward_quality_score'] > -10 # This is the way MagPie is using the reward model score
       and (
           example['min_similar_uuid'] is None
           or example['uuid'] == example['min_similar_uuid']
       )
   )
   
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Filtering")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory with input file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output files")
    parser.add_argument("--dataset_name", type=str, default=None, help="File name of dataset including extension")

    # System Settings
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save_faiss_index", type=bool, default=True, help="Save the Faiss index.")
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = get_args()
    
    # Input dirs
    dir = args.input_dir

    # Output dirs
    output_dir = args.output_dir
   
    # Input dataset
    dataset_name = args.dataset_name
    ds_number = dataset_name.partition('_')[0]
    
    print(f"input dir: {dir}")
    print(f"output dir: {output_dir}")
    print(f"dataset name: {dataset_name}")
    print(f"dataset number: {ds_number}")

    # Load dataset
    dataset_path = os.path.join(dir, dataset_name)
    dataset = load_dataset("json", 
                           data_files=dataset_path, 
                           num_proc=os.cpu_count())

    # print(dataset)
    # print(dataset["train"].features)
    # print(dataset["train"][0])
    samples_labeled = len(dataset["train"])
    print(f"# of samples in original dataset: {samples_labeled}")
    
    filtered_invalids = dataset['train'].filter(filter_out_invalid_samples, 
                                                num_proc=os.cpu_count())
    print(f"# of samples after removing null or empty data: {len(filtered_invalids)}")
    
    ds_unpacked = filtered_invalids.map(unpack_scores, num_proc=os.cpu_count())
    # print(ds_unpacked)
    # print(ds_unpacked.features)
    
    ds_valid_format = ds_unpacked.filter(validate_scores_format, 
                                         num_proc=os.cpu_count())
    print(f"# of samples with valid format: {len(ds_valid_format)}")

    # high_quality_filter is the method that should be updated for each dataset.
    filtered_dataset = ds_valid_format.filter(high_quality_filter)
    print(f"# of samples after applying high quality filter: {len(filtered_dataset)}")
    
    ## UPDATE DS NUMBER
    output_datast_name = f"{dataset_path[dataset_path.rfind(ds_number):dataset_path.rfind('difficulty')]}filtered.jsonl"
    output_file_path = os.path.join(output_dir,output_datast_name)
    save_dataset(filtered_dataset, output_file_path, convert_to_jsonl=True)
    print(f"Filtered dataset saved at: {output_file_path}")
    
    
    
    