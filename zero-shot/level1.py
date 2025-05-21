from openai import AzureOpenAI
import pandas as pd
from PIL import Image
import sys
import json
import time
from llava_icl import inference as llava_inference
from llama_32_zeroshot import inference as llama_inference
from LLaVA_OV import inference as inference_onevision
from qwen_zeroshot import inference as inference_qwen
from internvl2_zeroshot import inference as inference_intern
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, MllamaForConditionalGeneration, AutoProcessor, LlavaOnevisionForConditionalGeneration, Qwen2VLForConditionalGeneration
import torch
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
import math
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
import sys
import random
import string

def load_blank_image(input_size=448):
    image = Image.new('RGB', (input_size, input_size), (255, 255, 255))
    return image

#filename = sys.argv[1]
model_name = sys.argv[1]

if('gpt' in model_name):
    deployment = model_name

    client = AzureOpenAI(
        api_key="",
        api_version='',
        azure_endpoint='',
        azure_deployment=deployment
    )

scenarios = pd.read_excel('../path-generation/Scenarios.xlsx')

output_file = f"output_{model_name.replace('/','-')}_level1_zero_shot.json"
try:
    outputs = json.load(open(output_file))
except:
    outputs = []


for _ in range(5):
    for i in range(scenarios.shape[0]):
        correct = []
        
        for met in scenarios.columns[1:]:
            if(scenarios.iloc[i][met] != 0):
                correct.append(met)

        prompt = scenarios.iloc[i][0] + f''' The following descriptors are available:
1. Minimum Clearance: The minimum distance from the obstacles.
2. Maximum Clearance: The maximum distance from the obstacles.
3. Average Clearance: The average distance from the obstacles.
3. Smoothness: The sum of absolute angles between path segments. Smoother paths have a lower smoothness value.
4. Number of sharp turns: The number of turns that are >90 degrees.
5. Maximum turn angle: The sharpest turn angle in the path.
6. Path length: The sum of Euclidean distances between points in the path.

Which ones are the most important for the specified scenario?
Your answer should follow this format:
Answer: list of required descriptors separated by ";"
Explanation: why these descriptors are important
'''

        if('gpt' in model_name):
            messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a knowledgeable assistant in the field of path planning and navigation."
                            }
                        ]
                    },{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
        
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0,
                    max_tokens=1000,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )    
            except:
                time.sleep(30)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0,
                    max_tokens=1000,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                
            answer = response.choices[0].message.content
            
        elif 'llava' in model_name and '1.6' in model_name:
            model = LlavaNextForConditionalGeneration.from_pretrained(model_name, cache_dir="new_cache_dir/", torch_dtype=torch.float16, low_cpu_mem_usage=True).to('cuda')
            processor = LlavaNextProcessor.from_pretrained(model_name, cache_dir="new_cache_dir/")
            
            prompt = f"<image>\nUSER:{prompt}\nASSISTANT:"
            answer = llava_inference('', prompt=prompt, method='level1', model=model, processor=processor)[0].split('ASSISTANT:')[1]

        elif 'Llama' in model_name:
            model = MllamaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir='./new_cache_dir'
            )

            processor = AutoProcessor.from_pretrained(model_name)
            prompt = f"<image>\nUSER:{prompt}\nASSISTANT:"
            answer = llama_inference('', prompt=prompt, method='level1', model=model, processor=processor)  
        elif 'onevision' in model_name:
            processor = AutoProcessor.from_pretrained(model_name)
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir='./new_cache_dir'
            )
            
            prompt = f'USER: {prompt}\nASSISTANT:'
            answer = inference_onevision('', prompt, method='level1',model=model, processor=processor)
        elif 'Qwen' in model_name:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                cache_dir='./new_cache_dir'  # Specify the custom cache directory
            )
            processor = AutoProcessor.from_pretrained(model_name, cache_dir='./new_cache_dir')
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "blank_image.png",
                        },
                        {"type": "text", "text":prompt},
                    ],
                }
            ]
            
            answer = inference_qwen(messages, model=model, processor=processor)
       
        elif 'Intern' in model_name:
            
            def split_model(model_name):
                device_map = {}
                world_size = torch.cuda.device_count()
                num_layers = {
                    'OpenGVLab/InternVL2-1B': 24, 'OpenGVLab/InternVL2-2B': 24, 'OpenGVLab/InternVL2-4B': 32, 'OpenGVLab/InternVL2-8B': 32,
                    'OpenGVLab/InternVL2-26B': 48, 'OpenGVLab/InternVL2-40B': 60, 'OpenGVLab/InternVL2-Llama3-76B': 80}[model_name]
                # Since the first GPU will be used for ViT, treat it as half a GPU.
                num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
                num_layers_per_gpu = [num_layers_per_gpu] * world_size
                num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
                layer_cnt = 0
                for i, num_layer in enumerate(num_layers_per_gpu):
                    for j in range(num_layer):
                        device_map[f'language_model.model.layers.{layer_cnt}'] = i
                        layer_cnt += 1
                device_map['vision_model'] = 0
                device_map['mlp1'] = 0
                device_map['language_model.model.tok_embeddings'] = 0
                device_map['language_model.model.embed_tokens'] = 0
                device_map['language_model.output'] = 0
                device_map['language_model.model.norm'] = 0
                device_map['language_model.lm_head'] = 0
                device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
                return device_map
            
            device_map = split_model(model_name)

            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True,
                device_map=device_map,
                return_dict=True,
                cache_dir='./new_cache_dir').eval()
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, cache_dir='./new_cache_dir')
            
            answer = inference_intern('', prompt=prompt, model=model, tokenizer=tokenizer, method='level1')
 
        print(answer) 
        outputs.append({
            'prompt': prompt,
            'output': answer,
            'metrics': correct
        })
        
        print(len(outputs))
        with open(output_file, 'w') as f:
            obj = json.dumps(outputs, indent = 4)
            f.write(obj)
