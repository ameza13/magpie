import os
import torch
from vllm import LLM, SamplingParams
import json

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    device="0,1,2,3,4,5,6,7"

    os.environ["CUDA_VISIBLE_DEVICES"] = device
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3",
    llm = LLM(model="mistralai/Mistral-Nemo-Instruct-2407",
             dtype="bfloat16",
             trust_remote_code=True,
             gpu_memory_utilization=0.95,
             swap_space=2.0,
             max_model_len=4096,
             tensor_parallel_size=1)
    
    # llm = LLM(model="mistralai/Mistral-Nemo-Instruct-2407",
    #         dtype="bfloat16",
    #         trust_remote_code=True,
    #         gpu_memory_utilization=0.95,
    #         swap_space=2.0,
    #         max_model_len=4096,
    #         tensor_parallel_size=8) # tensor parallel not working with mistral 12B

    params = SamplingParams(temperature=0.3, 
                                    top_p=1.0, 
                                    max_tokens=4096,
                                    repetition_penalty=1.0,
                                    stop_token_ids=[1,2,3,4])

    print("model loaded")

    # messages = [
    #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    #     {"role": "user", "content": "Who are you?"},
    # ]

    messages = [
        {"role": "user", "content": ""},
    ]
   
    prompts = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(messages, 
                                                                    tokenize=False, 
                                                                    add_generation_template=True)
    outputs = llm.generate(prompts, params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated Response: {generated_text!r}")