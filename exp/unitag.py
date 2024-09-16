import os
import json
import warnings
from tqdm import tqdm
import argparse
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
from utils import load_dataset_from_file, save_dataset
from str_utils import input_difficulty_rating, input_classification, input_quality_rating, input_safety_rating,sample_quality_rating, input_quality_rating_mt, sample_quality_rating_mt, conversations_quality_rating_mt
from lingua import Language, LanguageDetectorBuilder
import time

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Unified Tagging Manager.")
    parser.add_argument("--tag_mission", type=str, default="quality", help="The tagging mission.", choices=["difficulty", "quality", "classification", "safety", "reward", "language", "sample_quality","conversation_quality"])
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Tag Model.")
    parser.add_argument("--guard_model_path", type=str, default="allenai/wildguard", help="Guard Model.")
    parser.add_argument("--reward_model_path", type=str, default="sfairXC/FsfairX-LLaMA3-RM-v0.1", help="Reward Model.")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--output_dir", type=str, default="", help="Directory path to save output files")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of samples per batch. Online <100, Offline <200.")
    parser.add_argument("--checkpoint_every", type=int, default=40, help="Save checkpoint every n batches")
    # parser.add_argument("--api", type=bool, default=False, help="Use API to generate responses")
    parser.add_argument("--offline", action="store_false", dest="api", help="Use local vllm engine")
    parser.add_argument("--online", action="store_true", dest="api", help="Use Together API engine")
    # parser.add_argument("--api_url", type=str, default="https://api.together.xyz/v1/chat/completions", help="API URL")
    # parser.add_argument("--api_key", type=str, default=None, help="Together API Key")
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode. Only process the first 100 samples.")
    parser.add_argument("--save_as", type=str, default="jsonl", choices=["json", "jsonl"], help="Save the generated responses as a what kind of file")

    # vllm Configs
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--quantization", type=str, default="fp8", choices=["fp8", "awq", "gptq", None])
    parser.add_argument("--kv_cache_dtype", type=str, default="auto", choices=["auto", "fp8"])
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    
    # Tagging Generation Configs
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    return parser.parse_args()

args = get_args()
print(f"[unitag.py] Unified Tagging Manager. Arguments: {args}") # For logging

MODEL_NAME = args.model_path
# R_MODEL_NAME = args.reward_model_path
R_MODEL_NAME="blue/reward-model" # TEMP: We loaded the reward model weights from a local path, as a local path is sensitive info we temporarily identify this reward mdoel as blue/reward-model
checkpoint_every = args.checkpoint_every if args.tag_mission != "reward" else args.checkpoint_every*100
batch_size = args.batch_size
mission = args.tag_mission

def template_generator_mt(conversations, mission):
    if mission == "quality":
        return input_quality_rating_mt(conversations=conversations)
    elif mission == "sample_quality":
        return sample_quality_rating_mt(conversations=conversations)
    elif mission == "conversation_quality":
        return conversations_quality_rating_mt(conversations=conversations)
    else:
        raise ValueError("Invalid mission. Available missions: quality, sample_quality, conversation_quality")
    
def template_generator(input, mission, output=""):
    if mission == "difficulty":
        return input_difficulty_rating(input)
    elif mission == "quality":
        return input_quality_rating(input)
    elif mission == "classification":
        return input_classification(input)
    elif mission == "sample_quality":
        return sample_quality_rating(input,output)
    else:
        raise ValueError("Invalid mission. Available missions: difficulty, quality, classification, sample_quality")

# Process safety item
def convert_safety_response_to_json(resp, c):
    lines = resp.split(c)
    output = {}
    try:
        for line in lines:
            parts = line.split(":")
            output[f'{parts[0]}'] = f'{parts[1]}'.strip()   
    except:
        print("Error: cannot parse output to dict")
    output['source'] = resp # We save the source in case the conversion to json fails
    return output

def process_safety_responses(response, item):
    if 'safety_output' in item:
        safety_scores = item['safety_output']
        safety_scores.append({f'{args.guard_model_path}':response})
    else:
        safety_scores = [{f'{args.guard_model_path}':response}]
    item['safety_output'] = safety_scores

    if 'safety_model' in item['metadata']:
        safety_models = item['metadata']['safety_model']
        safety_models.append(args.guard_model_path)
    else:
        safety_models = [args.guard_model_path]
    item['metadata']['safety_model'] = safety_models
    return item

def get_item(item, correct_type):
    correct_type = isinstance(item, correct_type)
    if not correct_type:
        print(f"[unitag.py] Failed to process item with error: incorrect type in assesment response")
        print(f"\ttype: {type(item)}")
        print(f"\tvalue: {item}")
    return correct_type

def set_empty_values(item, mission):
    if mission == "difficulty":
        item['intent'] = None
        item['knowledge'] = None
        item['difficulty'] = None
        item['metadata']['label_model'] = None
    elif mission == "quality":
        item['input_quality'] = None
        item['input_quality_explanation'] = None
        item['metadata']['label_model'] = None
    elif mission == "classification":
        item['task_category'] = None
        item['metadata']['label_model'] = None
    elif mission == "sample_quality":
        item['judge_quality_score'] = None
        item['judge_quality_explanation'] = None
        item['metadata']['label_model'] = None 
    elif mission == "conversation_quality":
        item['conversation_score'] = None
        item['conversation_explanation'] = None
        item['metadata']['label_model'] = None 
    return item

def clean_engine_response(response):
    # occurrences = [i for i in range(len(response)) if response.startswith("```", i)] 
    # if len(occurrences) == 2: # There are 2 ```
    #     pass # Remove text outside ``` ``` (if any)
    
    if response.startswith("```") and response.endswith("```"):
        # print(f"==Response to be loaded==\n{response[3:-3].strip()}") # TEMP
        return response[3:-3].strip()
    if response.startswith("```json") and response.endswith("```"):
        # print(f"==Response to be loaded==\n{response[7:-3].strip()}") # TEMP
        return response[7:-3].strip()
    return response

def process_engine_responses(response, item, mission):
    try:
        # Attempt to clean response
        response = clean_engine_response(response)
        # Attempt to load response as json 
        tags_json = json.loads(response)    
        if mission == "difficulty":
            # Check difficulty format
            correct_type = get_item(tags_json['difficulty'], str)
            if not correct_type:
                item = set_empty_values(item, mission)
                return item
            
            # Only saves the LLM response if json format is valid and the score has correct format
            intent = {}
            intent[f'{MODEL_NAME}'] = tags_json['intent']
            knowledge = {}
            knowledge[f'{MODEL_NAME}'] = tags_json['knowledge']
            difficulty = {}
            difficulty[f'{MODEL_NAME}'] = tags_json['difficulty']
            
            item['intent'] = [intent]
            item['knowledge'] = [knowledge]
            item['difficulty'] = [difficulty]

            item['metadata']['label_model'] = [MODEL_NAME]
            
        elif mission == "quality":
            # Check input_quality format
            correct_type = get_item(tags_json['input_quality'], str)
            if not correct_type:
                item = set_empty_values(item, mission)
                return item
            
            input_quality = {}
            input_quality[f'{MODEL_NAME}'] = tags_json['input_quality']
            input_quality_explanation = {}
            input_quality_explanation[f'{MODEL_NAME}'] = tags_json['explanation'] # Does it crash because of breaklines?

            item['input_quality'] = [input_quality]
            item['input_quality_explanation'] = [input_quality_explanation]

            item['metadata']['label_model'] = [MODEL_NAME]
        elif mission == "classification":
            # Check input_quality format
            correct_type = get_item(tags_json['primary_tag'], str)
            if not correct_type:
                item = set_empty_values(item, mission)
                return item
            
            task_category = {}
            task_category[f'{MODEL_NAME}'] = tags_json["primary_tag"]
            item['task_category'] = [task_category]
            
            item['metadata']['label_model'] = [MODEL_NAME]
        elif mission == "sample_quality":
            # Check input_quality format
            correct_type = get_item(tags_json['score'], str)
            if not correct_type:
                item = set_empty_values(item, mission)
                return item
            
            sample_quality_score = {}
            sample_quality_score[f'{MODEL_NAME}'] = tags_json['score']
            sample_quality_explanation = {}
            sample_quality_explanation[f'{MODEL_NAME}'] = tags_json['explanation'] # Does it crash because of breaklines?

            item['judge_quality_score'] = [sample_quality_score]
            item['judge_quality_explanation'] = [sample_quality_explanation]

            item['metadata']['label_model'] = [MODEL_NAME]      
        elif mission == "conversation_quality":
            # Check input_quality format
            correct_type = get_item(tags_json['score'], str)
            if not correct_type:
                item = set_empty_values(item, mission)
                return item
            
            conversation_quality_score = {}
            conversation_quality_score[f'{MODEL_NAME}'] = tags_json['score']
            conversation_quality_explanation = {}
            conversation_quality_explanation[f'{MODEL_NAME}'] = tags_json['explanation'] # Does it crash because of breaklines?

            item['conversation_score'] = [conversation_quality_score]
            item['conversation_explanation'] = [conversation_quality_explanation]

            item['metadata']['label_model'] = [MODEL_NAME]     
    except Exception as e:
        print(f"[unitag.py] Failed to process item with error: {str(e)}")
        print(f"[unitag.py] Raw response from LLM tagger:\n {response}")
        item = set_empty_values(item, mission)
    return item

# Process a batch of data for: difficulty, quality, and classification missions
def process_batch(batch, llm, params, mission, tokenizer=None):
    
    prompts = []
    for i, item in enumerate(batch):
        if len(item["conversations"]) > 2 and mission in ["quality","sample_quality", "conversation_quality"]: # Multi-turn
            chat = [{"role": "user", "content": template_generator_mt(conversations=item["conversations"],
                                                                      mission=mission)}]
        else: # Single-turn: conversations length = 2
            chat = [{"role": "user", "content": template_generator(item[INSTRUCTION], mission, item[RESPONSE])}]
            # chat = [{"role": "user", "content": template_generator(item[INSTRUCTION], mission)}] # Instruction only

        template = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        template += "{" # Do we need it?, Yes
        prompts.append(template)

    # TEST
    print("=TEST=")
    print(f"Mission: {mission}")
    print(f"Prompt Example:\n{prompts[0]}")

    outputs = llm.generate(prompts, params)

    for i, item in enumerate(batch):
        model_response = "{\n" + outputs[i].outputs[0].text.strip()
        # Remove additional information at the end of the response
        model_response = model_response[:model_response.rfind("}")+1]

        # print("==Model Response==") # TEST
        # print(model_response) # TEST
        
        item = process_engine_responses(model_response, item, mission)
    
    return batch

# BATCH
# def process_batch_with_reward_model(batch, rm_model, rm_tokenizer):
#     prompts = []
#     for i, item in enumerate(batch):
#         input = item[INSTRUCTION]
#         output = item[RESPONSE]
#         template = f'<s> [INST]{input}[/INST]{output}</s>'
#         prompts.append(template)
        
#     # print(prompts) # TEST
#     outputs = rm_tokenizer(prompts, return_tensors='pt').to(device)
    
#     # str_max_length = max(prompts, key = len)
#     # outputs = rm_tokenizer(prompts, 
#     #           return_tensors='pt',
#     #           truncation=True,
#     #           padding='max_length', 
#     #           max_length=len(str_max_length)).to(device)

#     outputs = rm_tokenizer(prompts, return_tensors='pt').to(device)
#     scores = rm_model(**outputs)[0]
#     rm_model.eval().requires_grad_(False) # To save memory during inference
        
#     for i, item in enumerate(batch):
#         try:
#             task_reward = {}
#             task_reward[f'{R_MODEL_NAME}'] = scores[i].item()
#             item['reward_quality_score'] = [task_reward]
#             item['metadata']['reward_model'] = [R_MODEL_NAME]
#         except Exception as e:
#             print(f"Failed to process item: {item} with error: {str(e)}")
#             item['reward_quality_score'] = None
#             item['metadata']['reward_model'] = None 
#     return batch

# TEMPORAL: 1 by 1
def process_batch_with_reward_model(batch, rm_model, rm_tokenizer):
    scores = []
    for i, item in enumerate(batch):
        input = item[INSTRUCTION]
        output = item[RESPONSE]
        template = f'<s> [INST]{input}[/INST]{output}</s>'
        try:
            output = rm_tokenizer(template, return_tensors='pt').to(device)
            scores.append(rm_model(**output)[0])
            rm_model.eval().requires_grad_(False) # To save memory during inference
        except Exception as e:
            print(f"Error:{item['uuid']}, {len(item['input'])}, {len(item['output'])}")
            scores.append(-1000000.0) # - 1M to use as filter due to input/output lengths that the RM cannot process 
    
    for i, item in enumerate(batch):
        try:
            task_reward = {}
            task_reward[f'{R_MODEL_NAME}'] = scores[i].item()
            item['reward_quality_score'] = [task_reward]
            item['metadata']['reward_model'] = [R_MODEL_NAME]
        except Exception as e:
            print(f"Failed to process item: {item} with error: {str(e)}")
            item['reward_quality_score'] = None # For -1000000.0 values it will really assign None.
            item['metadata']['reward_model'] = None 
    return batch

# wildguard
def process_batch_with_guard_model(batch, s_model, sm_kwargs):
    inputs = []
    for instance in batch:
        model_input = input_safety_rating(prompt=instance[INSTRUCTION], response=instance[RESPONSE])
        inputs.append(model_input)

    str_max_length = max(inputs, key = len)
    print(f"Max Length in batch: {len(str_max_length)}") # TEST

    tokenized_inputs = sm_tokenizer(inputs, 
                                    return_tensors='pt',
                                    add_special_tokens=False,
                                    truncation=True,
                                    padding='max_length',
                                    max_length=len(str_max_length))
    results = s_model.generate(**tokenized_inputs, max_new_tokens=sm_kwargs["max_new_tokens"])

    inputs_decoded = sm_tokenizer.batch_decode(tokenized_inputs['input_ids'], skip_special_tokens=True)
    results_decoded = sm_tokenizer.batch_decode(results, skip_special_tokens=True)

    for item, result_decoded,input_decoded in zip(batch,results_decoded,inputs_decoded):
        # print(f"=Result decoded=\n{len(result_decoded)}")
        # print(f"Input len: {len(input_decoded)}")
        # print(f"Result: {result_decoded}")

        resp = result_decoded[len(input_decoded):]
        wildguard_json = convert_safety_response_to_json(resp,'\n')

        # print(f"=Response=\n{resp}") # TEST
        # print(f"=Response json=\n{wildguard_json}") # TEST

        # Update output
        process_safety_responses(wildguard_json, item)
    return batch

# Generate outputs, update dataset in batches, and overwrite checkpoint
def generate_and_update(dataset, mission, llm, params, rm_model, rm_tokenizer, s_model,sm_kwargs, batch_size, checkpoint_file, checkpoint_every = 20):
    if os.path.exists(checkpoint_file):
        last_checkpoint_idx = len(load_dataset_from_file(checkpoint_file))
        print(f"[unitag.py] Checkpoint file found. Resuming from last checkpoint with index {last_checkpoint_idx}.")
        dataset[:last_checkpoint_idx] = load_dataset_from_file(checkpoint_file)
        num_batches = (len(dataset) - last_checkpoint_idx + batch_size - 1) // batch_size
        print(f"[unitag.py] Remaining number of batches: {num_batches}")
    else:
        last_checkpoint_idx = 0
        num_batches = (len(dataset) + batch_size - 1) // batch_size  # Calculate total number of batches
        print(f"[unitag.py] Total number of batches: {num_batches}")

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size + last_checkpoint_idx
        end_idx = min((i + 1) * batch_size + last_checkpoint_idx, len(dataset))
        batch = dataset[start_idx:end_idx]

        print(f'Batch #: {i}/{num_batches}')
        print(f'Batch size: {len(batch)}')
        
        
        if mission == "reward":
            batch= process_batch_with_reward_model(batch, rm_model, rm_tokenizer)
        elif mission == "safety":
            process_batch_with_guard_model(batch,s_model,sm_kwargs)
        else:
            batch = process_batch(batch, llm, params, mission)

        dataset[start_idx:end_idx] = batch
        # Overwrite the same checkpoint file every checkpoint_every batches
        if (i + 1) % checkpoint_every == 0:
            save_dataset(dataset[:end_idx], checkpoint_file)
            print(f"[unitag.py] Dataset checkpoint {checkpoint_file} saved after batch {i + 1}.")

    return dataset

if __name__ == "__main__":
    start = time.time()

    INSTRUCTION = "input"
    RESPONSE = "output"
    input_file = args.input_file
    output_dir = args.output_dir

    # Mission Settings
    if mission == "difficulty":
        output_file = f"{input_file[:input_file.rfind('.')]}_difficulty.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_difficulty_checkpoint.json"
    elif mission == "quality":
        output_file = f"{input_file[:input_file.rfind('.')]}_quality.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_quality_checkpoint.json"
    elif mission == "classification":
        output_file = f"{input_file[:input_file.rfind('.')]}_category.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_category_checkpoint.json"
    elif mission == "safety":
        # sm_tokenizer = AutoTokenizer.from_pretrained(args.guard_model_path) # Create it at the same time than model
        output_file = f"{input_file[:input_file.rfind('.')]}_safety.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_safety_checkpoint.json"
    elif mission == "reward":
        # rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path) # Create it at the same time than model
        output_file = f"{input_file[:input_file.rfind('.')]}_reward.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_reward_checkpoint.json"
    elif mission == "language":
        output_file = f"{input_file[:input_file.rfind('.')]}_language.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_language_checkpoint.json"
    elif mission == "sample_quality":
        output_file = f"{input_file[:input_file.rfind('.')]}_sample-quality.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_sample-quality_checkpoint.json"
    elif mission == "conversation_quality":
        output_file = f"{input_file[:input_file.rfind('.')]}_conversation-quality.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}__conversation-quality_checkpoint.json"
    else:
        raise ValueError("Invalid mission. Available missions: difficulty, quality, classification, safety, reward, language, sample_quality, conversation_quality")
    # Change jsonl to json if args.save_as is json
    if args.save_as == "json":
        output_file = f"{output_file[:output_file.rfind('.')]}.json"

    # Load dataset
    if not args.debug:
        dataset = load_dataset_from_file(input_file)
    else:
        warnings.warn("Debug mode enabled. Only processing the first 100 samples.")
        dataset = load_dataset_from_file(input_file)[:100]
        # checkpoint_file = f"{output_file[:output_file.rfind('.')]}_debug.jsonl"
        # checkpoint_file = f"{output_file[:output_file.rfind('.')]}_debug_checkpoint.json"

    print(f"Processing {len(dataset)} samples")
    
    if mission != "language":
        if args.tag_mission == "reward":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            
            rm_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path, 
                                                                          torch_dtype=torch.float16).to(device)
            rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)
            
            s_model = None
            sm_kwargs = None
            llm = None
            params = None
        elif args.tag_mission == "safety":
            # Currently vllm do not support safety model inference, use transformer inference instead
            sm_tokenizer = AutoTokenizer.from_pretrained(args.guard_model_path)
            s_model = AutoModelForCausalLM.from_pretrained(args.guard_model_path)
    
            sm_kwargs = {
                "max_new_tokens" : 32
            }
            # rm_pipe = None 
            # rm_pipe_kwargs = None
            llm = None
            params = None
            rm_model= None
            rm_tokenizer=None
        else:
            print("[unitag.py] Start Local vllm engine...")
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

            llm = LLM(model=MODEL_NAME,
                        dtype=args.dtype,
                        quantization=args.quantization,
                        kv_cache_dtype=args.kv_cache_dtype,
                        max_model_len=args.max_model_len,
                        tensor_parallel_size=args.tensor_parallel_size,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        trust_remote_code=True,
                        enable_prefix_caching=True,)
            
            params = SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        repetition_penalty=args.repetition_penalty,
                        stop=["}"], # Dow we need this one?
                        stop_token_ids=[
                        "<s>",
                        "</s>",
                        "[INST]",
                        "[/INST]"
                        ],
                        include_stop_str_in_output=True,
                        )
    
            s_model = None
            sm_kwargs = None
            # rm_pipe = None
            # rm_pipe_kwargs = None
            rm_model= None
            rm_tokenizer=None

        updated_dataset = generate_and_update(dataset, mission, llm, params, rm_model, rm_tokenizer, s_model,sm_kwargs, batch_size, checkpoint_file, checkpoint_every)
    else:
        print("[unitag.py] Start language detection engine...")
        detector = LanguageDetectorBuilder.from_all_languages().build()
        for item in tqdm(dataset):
            if item[INSTRUCTION] != "":
                try:
                    item['language_identify'] = detector.detect_language_of(item[INSTRUCTION]).iso_code_639_1.name
                except Exception as e:
                    print(f"Failed to process item with error: {str(e)}")
                    item['language_identify'] = None
            else:
                item['language_identify'] = None
        updated_dataset = dataset

    if args.save_as == "json":
        save_dataset(updated_dataset, os.path.join(output_dir,output_file), convert_to_jsonl=False)
    else:
        save_dataset(updated_dataset, os.path.join(output_dir,output_file), convert_to_jsonl=True)
    
    # TEST - save to csv
    # import pandas as pd
    # output_file_csv = f"{output_file[:output_file.rfind('.')]}.csv"
    # df = pd.read_json(output_file)
    # df.to_csv(output_file_csv, index=False)

    # Remove the checkpoint file after completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("[unitag.py] Final dataset saved. Checkpoint removed.")
    end = time.time()

    print(f"Total computation takes: {end-start} s")