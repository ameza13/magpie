import os
import torch
import json
from vllm import LLM, SamplingParams

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
    

def sample_quality_rating(input,output):
    user_message = f'''
## Instruction 

We would like to request your feedback on the performance of AI assistant in regard to the instruction and response displayed following.

Instruction: 
```
{input}
```
Response:
```
{output}
```

Please rate according to the accuracy, instruction following, and presentation of the response to the instruction. In particular, when judging, consider:

- Instruction Following: Does the answer directly address the question?
- Accuracy: Is the information provided in the response correct?
- Presentation: Is the answer logically structured and easy to understand?

You should provide each assistant a score on a scale from 0 to 5, where a higher score indicates a higher overall quality. 
You should also provide a comprehensive explanation of your assesment.

## Output Format
Please provide your response in the following format, by filling in the placeholders in [...]:

```
{{   
    "score": "[1,2,3,4,5]",
    "explanation": "[...]"
}}
```
'''
    return user_message

if __name__ == "__main__":
    input_file= os.path.join("/input/file/path") # MUST UPDATE
    dataset = load_dataset_from_file(input_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3", tensor_parallel_size=8)

    params = SamplingParams(
                temperature=0,
                max_tokens=1024,
                repetition_penalty=1.0,
                stop=["}"], # Dow we need this one?
                stop_token_ids=[
                "<s>",
                "</s>",
                "[INST]",
                "[/INST]"
                ],
                include_stop_str_in_output=True,
                )
    prompts = []
    for instance in dataset:
        msg = sample_quality_rating(instance["input"], instance["output"])
        chat = [{"role": "user", "content":  msg.strip()}]
        template = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        template += "{" # Do we need it?, Yes
        prompts.append(template)

    print("=First Prompt=")
    print(prompts[0])

    outputs = llm.generate(prompts, params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = "{\n" + output.outputs[0].text.strip()
        # Remove additional information at the end of the response
        generated_text = generated_text[:generated_text.rfind("}")+1]
        print(f"Prompt: {prompt!r}\nGenerated Instruction (chat template):\n{generated_text!r}")

