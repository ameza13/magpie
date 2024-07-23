import torch
import os
import sys
import argparse
import json
import requests
import concurrent.futures
from time import sleep
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import load_dataset_from_file, save_dataset, make_api_request_with_retry, get_conversation_template
from vllm import LLM, SamplingParams

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Response Generation Manager.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="We will support more models in the future.")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of samples per batch")
    parser.add_argument("--checkpoint_every", type=int, default=20, help="Save checkpoint every n batches")
    parser.add_argument("--api", type=bool, default=False, help="Use API to generate responses")
    parser.add_argument("--api_url", type=str, default="https://api.together.xyz/v1/chat/completions", help="API URL")
    parser.add_argument("--api_key", type=str, default=None, help="Together API Key")
    parser.add_argument("--offline", action="store_false", dest="api", help="Use local vllm engine")

    # Generation Parameters
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism. Only used for Llama 70B models.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--tokenizer_template", type=bool, default=False, help="Use tokenizer template for generating the response.")
    parser.add_argument("--use_tokenizer_template", action="store_true", dest="tokenizer_template")

    return parser.parse_args()

args = get_args()
print(f"Response Generation Manager. Arguments: {args}") # For logging

if args.input_file is None:
    raise ValueError("Please specify the input file path.")

# Constants for the local vllm engine
MODEL_NAME = args.model_path
INPUT_FILE_NAME = args.input_file 
BATCH_SIZE = args.batch_size
CHECKPOINT_FILE = f"{INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]}_checkpoint.json"
CHECKPOINT_EVERY = args.checkpoint_every
SAVED_FILE = f"{INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]}_res.json"

# Obtain config from configs/model_configs.json
CONFIG_FILE_PATH = os.path.join(os.environ['HOME'],"/workspace/magpie/configs/model_configs.json")
with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
    model_configs = json.load(f)
    model_config = model_configs[args.model_path]
    stop_tokens = model_config["stop_tokens"]
    stop_token_ids = model_config["stop_token_ids"]

# API Setups
if args.api:
    # Change name for API (Together Naming Convention)
    if MODEL_NAME == "meta-llama/Meta-Llama-3-8B-Instruct":
        api_model_name = "meta-llama/Llama-3-8b-chat-hf"
    elif MODEL_NAME == "meta-llama/Meta-Llama-3-70B-Instruct":
        api_model_name = "meta-llama/Llama-3-70b-chat-hf"
    else:
        api_model_name = MODEL_NAME

    # Constants for the API
    API_ENDPOINT = args.api_url
    API_HEADERS = {
        "Authorization": args.api_key,
    }
    API_PARAMS = {
        "model": api_model_name,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "stop": stop_tokens
    }

# Process a batch of data using the API
def process_batch_with_api(batch):
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_item = {
            executor.submit(
                make_api_request_with_retry, 
                {'content': item['instruction'], 'role': 'user'},
                API_PARAMS,
                API_ENDPOINT,
                API_HEADERS,
            ): item 
            for item in batch
        }

        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                api_response = future.result()
                item['response'] = api_response.strip()
                item['gen_response_configs'] = {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "max_tokens": args.max_tokens,
                    "stop_tokens": stop_tokens,
                    "output_generator": MODEL_NAME,
                }
            except Exception as e:
                print(f"Failed to process item: {item} with error: {str(e)}")
                item['response'] = ""
                
    return batch

# Process a batch of data using local vllm engine
def process_batch(batch, llm, params, tokenizer=None):
    user_instructions = [item['instruction'] for item in batch]
    prompts = []
    for instruction in user_instructions:
        if not args.tokenizer_template:
            conv = get_conversation_template(MODEL_NAME)
            conv.append_message(conv.roles[0], instruction)
            conv.append_message(conv.roles[1], None)
            template = conv.get_prompt()
        else:
            chat = [{"role": "user", "content": instruction}]
            template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            # template = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(chat, 
            #                                                                   tokenize=False, 
            #                                                                   add_generation_prompt=True)
        prompts.append(template)
    # print("= TEST =") # TEST
    # print(f"Total prompts in batch: {len(prompts)}") # TEST
    # print(f"First sample type: {type(prompts[0])}")
    # print(f"First sample content: {prompts[0]}") # TEST

    outputs = llm.generate(prompts, params)

    # print(len(outputs)) # TEST
    # print(outputs[0]) # TEST

    for i, item in enumerate(batch):
        item['response'] = outputs[i].outputs[0].text.strip()
        item['gen_response_configs'] = {
            "prompt": prompts[i],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_tokens": args.max_tokens,
            "stop_tokens": stop_tokens,
            "output_generator": MODEL_NAME,
        }
    return batch

# Generate outputs, update dataset in batches, and overwrite checkpoint
def generate_and_update(dataset, llm=None, params=None, api=False, tokenizer=None):

    # Intialize the dataset with the checkpoint file (if it exists)
    if os.path.exists(CHECKPOINT_FILE):
        last_checkpoint_idx = len(load_dataset_from_file(CHECKPOINT_FILE))
        print(f"Checkpoint file found. Resuming from last checkpoint with index {last_checkpoint_idx}.")
        dataset[:last_checkpoint_idx] = load_dataset_from_file(CHECKPOINT_FILE)
        # Calculate total number of batches
        num_batches = (len(dataset) - last_checkpoint_idx + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"Remaining number of batches: {num_batches}")
    else:
        last_checkpoint_idx = 0
        # Calculate total number of batches
        num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Total number of batches: {num_batches}")

    for i in tqdm(range(num_batches)):
        start_idx = i * BATCH_SIZE + last_checkpoint_idx
        end_idx = min((i + 1) * BATCH_SIZE + last_checkpoint_idx, len(dataset))
        batch = dataset[start_idx:end_idx]
        if api:
            batch = process_batch_with_api(batch)
        else:
            # print(f"Batch: {i}") # TEST
            batch = process_batch(batch, llm, params, tokenizer)
        
        dataset[start_idx:end_idx] = batch
        # Overwrite the same checkpoint file after serveral batches
        if i % CHECKPOINT_EVERY == 0:
            save_dataset(dataset[:end_idx], CHECKPOINT_FILE)
            print(f"Dataset checkpoint saved after batch {i + 1}.")

    return dataset

# Main function to control workflow
def main():
    # Load instructions from the input file
    dataset = load_dataset_from_file(INPUT_FILE_NAME)
    
    if args.api:
        print("Start together API engine...")
        llm = None
        params = None
    else:
        # Set the device
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print("Start Local vllm engine...")

        # TEST
        # print("= llm params =")
        # print(f"model: {MODEL_NAME}")
        # print(f"dtype: {args.dtype}")
        # print(f"gpu_memory_utilization: {args.gpu_memory_utilization}")
        # print(f"max_model_len: {args.max_model_len}")
        # # print(f"swap_space: {args.swap_space}")
        # print(f"tensor_parallel_size: {args.tensor_parallel_size}")
        # # print(f"seed: {args.seed if args.seed is not None else args.timestamp}")

        llm =  LLM(model=MODEL_NAME, 
            dtype=args.dtype,
            trust_remote_code=True,
            gpu_memory_utilization = args.gpu_memory_utilization,
            max_model_len = args.max_model_len, # limited by kv-cache 
            tensor_parallel_size = args.tensor_parallel_size,
            )

        # TEST
        # print("= sampling params =")
        # # print(f"n: {args.n}")   
        # print(f"temperature: {args.temperature}")
        # print(f"top_p: {args.top_p}")
        # print(f"max_tokens: {args.max_tokens}")       
        # # print(f"skip_special_tokens: {args.skip_special_tokens}")
        # # print(f"stop: {stop_tokens}")
        # print(f"repetition_penalty: {args.repetition_penalty}")
        # print(f"stop_token_ids: {stop_token_ids}")     

        params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=stop_token_ids,
            )

    updated_dataset = generate_and_update(dataset, llm, params, api=args.api, tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME))

    # TEST
    # batch = dataset[:100] # Take first 100 samples only
    # updated_dataset = process_batch(batch=batch, llm=llm, params=params)


    # Save final dataset
    save_dataset(updated_dataset, SAVED_FILE)

    # Optionally remove the checkpoint file after completion
    os.remove(CHECKPOINT_FILE)
    print(f"Final dataset saved at: {SAVED_FILE}.\nCheckpoint removed.")

# Run the main function
if __name__ == "__main__":
    main()