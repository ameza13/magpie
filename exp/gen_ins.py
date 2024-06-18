import torch
import os
import sys
import argparse
import json
import time
import random
import numpy as np
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vllm import LLM, SamplingParams

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Instruction Generation Manager.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="We will support more models in the future.")
    # Generation Parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=200, help="Number of samples to generate for one time.")
    parser.add_argument("--repeat", type=int, default=None, help="Number of times to repeat the instruction generation. Only available when total prompts is not specified.")
    parser.add_argument("--total_prompts", type=int, default=1000, help="Total number of prompts to generate. If specified, repeat will be ignored.")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--early_stopping", type=bool, default=True, help="Stop generation when the \n is generated.")
    parser.add_argument("--disable_early_stopping", action="store_false", dest="early_stopping", help="Disable early stopping.")
    parser.add_argument("--have_system_prompt", type=bool, default=False, help="Use system prompt for extracting the input. Recommend disabling it.")
    parser.add_argument("--enable_system_prompt", action="store_true", dest="have_system_prompt", help="Enable system prompt for extracting the input.")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle the outputs generated by vllm.")
    parser.add_argument("--skip_special_tokens", type=bool, default=True)
    parser.add_argument("--checkpoint_every", type=int, default=100, help="Save checkpoint every n repeats.")

    # System Settings
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism. Only used for Llama 70B models.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--swap_space", type=float, default=2.0)
    parser.add_argument("--output_folder", type=str, default="../data")
    parser.add_argument("--job_name", type=str, default=None, help="Job Name. Get from the script.")
    parser.add_argument("--timestamp", type=int, default=int(time.time()), help="Timestamp for the job.")
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--verbose_on", action="store_true", dest="verbose", help="Enable verbose.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    return parser.parse_args()

args = get_args()
print(f"Instruction Generation Manager. Arguments: {args}") # For logging

if args.total_prompts is None:
    if args.repeat is None:
        raise ValueError("Either total prompts or repeat should be specified.")
    args.total_prompts = args.repeat * args.n
else:
    # If total prompts is specified, repeat will be ignored
    args.repeat = int(np.ceil(args.total_prompts / args.n))

# Set the random seed for NumPy
if args.seed is not None:
    np.random.seed(args.seed)
    # Set the random seed for PyTorch
    torch.manual_seed(args.seed)
    # If you are using CUDA (i.e., a GPU), also set the seed for it
    torch.cuda.manual_seed_all(args.seed)

# Create output file / folder
output_filename = f"Magpie_{args.model_path.split('/')[-1]}_{args.total_prompts}_{args.timestamp}_ins.json"
if not args.job_name:
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    output_dir = f"{args.output_folder}/{output_filename}"
else:
    output_dir = f"{args.output_folder}/{args.job_name}/{output_filename}"

# Set the device
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

# Create vllm instance  
llm = LLM(model=args.model_path, 
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed if args.seed is not None else args.timestamp)

# Obtain config from configs/model_configs.json
with open("../configs/model_configs.json", "r") as f:
    model_configs = json.load(f)
    model_config = model_configs[args.model_path]
    if args.have_system_prompt:
        extract_input = model_config["extract_input_with_system_prompt"]
    else:
        extract_input = model_config["extract_input"]
    stop_tokens = model_config["stop_tokens"]
    stop_tokens_assistant = model_config["stop_tokens_assistant"]
    stop_tokens += stop_tokens_assistant
    stop_token_ids = model_config["stop_token_ids"]

if args.early_stopping:
    stop_tokens.append("\n")

# Define sampling parameters
sampling_params = SamplingParams(
    n=args.n,
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
    skip_special_tokens=args.skip_special_tokens,
    stop=stop_tokens,
    stop_token_ids=stop_token_ids,
)

################
# Generate outputs
################
results = []
for rounds in tqdm(range(args.repeat)):
    # Generate outputs
    output = llm.generate(extract_input, sampling_params)
    if args.shuffle:
        random.shuffle(output[0].outputs)

    # Save outputs
    for i, completion in enumerate(output[0].outputs):
        result = {
            "id": rounds * args.n + i,
            "extract_input": f"{extract_input}",
            "instruction": completion.text.strip(),
            "response": None,
            "created": int(time.time()),
            "gen_input_configs": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "input_generator": f"{args.model_path}",
                "seed": args.seed,
            },
            "gen_response_configs": None,
        }
        results.append(result)

    # Save the checkpoints every args.checkpoint_every rounds
    if rounds % args.checkpoint_every == 0:
        with open(output_dir, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Checkpoint saved. Total prompts: {len(results)}")

# Save the final results
with open(output_dir, "w") as f:
    json.dump(results, f, indent=2)

print(f"Instruction generated from {args.model_path}. Total prompts: {len(results)}")