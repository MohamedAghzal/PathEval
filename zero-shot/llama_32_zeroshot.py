import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import requests
import os
import json
import torch
import torch.nn.functional as F
import random
import string
import sys

def generate_random_string(length=4):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def load_blank_image(input_size=448):
    image = Image.new('RGB', (input_size, input_size), (255, 255, 255))
    return image


token = ''
login(token = token)

def inference(url, prompt, model, processor, method=''):
    if (method != 'no_image' and method != 'level1'):
        image = Image.open(url)
    else:
        image = load_blank_image()

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    
    output = model.generate(**inputs, max_new_tokens=2000)
    
    return processor.decode(output[0])

outputs = []

acc = 0


def parse(answer):
    assistant = answer.split('ASSISTANT:')[1]
    answer = assistant
    if('Answer:' in assistant):
        answer = assistant.split('Answer:')[1].split('Explanation:')[0]

    return answer


def main():
    model_id = sys.argv[1] #"meta-llama/Llama-3.2-11B-Vision"
    test_set = sys.argv[2]

    method = sys.argv[3]
    dim = sys.argv[4]
    
    test_data = json.load(open(test_set)) 

    print(len(test_data))   

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir='./new_cache_dir'
    )

    processor = AutoProcessor.from_pretrained(model_id)

    for instance in test_data:
        if('../' not in instance['image']):
            img_url = f"../{instance['image']}"
        else:
            img_url = f"{instance['image']}" 
        
        print(img_url) 
        scenario = instance['conversations'][0]['value']
        if(method == 'descriptors'):
            prompt = f'''USER: {scenario} Path 1 is on the left side and Path 2 is on the right side.
The following path descriptor values are computed for each path:
Minimum Clearance: The minimum distance from the obstacles. 
Maximum Clearance: The maximum distance from the obstacles. 
Smoothness: the sum of absolute angles between path segments. Smoother paths have a lower smoothness value.
Number of sharp turns: number of turns that are >90 degrees. 
Maximum turn angle: The sharpest turn angle in the path 
Path length: The sum of Euclidean distances between points in the path  
Here are path descriptor values for path 1: {instance['metrics']['Path 1']}
Here are path descriptor values for path 2: {instance['metrics']['Path 2']}
Your answer should follow the format below:
Answer: Path 1 or Path 2.
Explanation: Why you chose the path (1 or 2). 
\nASSISTANT:
    '''
        elif (method == 'perception'):
            prompt = f'''USER: Path 1 is on the left side and Path 2 is on the right side.
    You are asked to compare the two paths in terms of the following metrics:
    1. Minimum Clearance: The minimum distance from the obstacles.
    2. Maximum Clearance: The maximum distance from the obstacles.
    3. Average Clearance: The average distance from the obstacles. 
    4. Smoothness: The sum of absolute angles between path segments. Smoother paths have a lower smoothness value.
    5. Number of sharp turns: The number of turns that are >90 degrees.
    6. Maximum turn angle: The sharpest turn angle in the path.
    7. Path length: The sum of Euclidean distances between points in the path.

    Your answer should follow the format below:
    Metric: Name of the metric\tAnswer: Path 1 or Path 2 (which one has a lower value on the metric)\tExplanation:why?.\nASSISTANT:
    ''' 
        elif (method == 'none' or method == 'flipped'):
            prompt = f'''USER: {scenario} Path 1 is on the left side and Path 2 is on the right side.
    Your answer should follow the format below:
    Answer: Path 1 or Path 2.
    Explanation: Why you chose the path (1 or 2). 
    \nASSISTANT:
    '''
        elif (method == 'random'):
            path1_id = generate_random_string(4)
            path2_id = generate_random_string(4)
            
            prompt = f'''USER:{scenario} Path {path1_id} is on the left side and {path2_id} is on the right side.
    Your answer should follow the format below:
    Answer: Either Path {path1_id} or Path {path2_id}.
    Explanation: Why you chose the path ({path1_id} or {path2_id}). 
    \nASSISTANT:
    '''  
        elif (method == 'no_image'):
            prompt = f'''USER: {scenario} Path 1 is on the left side and Path 2 is on the right side.
    The following path descriptor values are computer for each path:
    Minimum Clearance: The minimum distance from the obstacles. 
    Maximum Clearance: The maximum distance from the obstacles. 
    Smoothness: the sum of absolute angles between path segments. Smoother paths have a lower smoothness value.
    Number of sharp turns: number of turns that are >90 degrees. 
    Maximum turn angle: The sharpest turn angle in the path 
    Path length: The sum of Euclidean distances between points in the path  
    Here are path descriptor values for path 1: {instance['metrics']['Path 1']}
    Here are path descriptor values for path 2: {instance['metrics']['Path 2']}
    Your answer should follow the format below:
    Answer: Path 1 or Path 2.
    Explanation: Why you chose the path (1 or 2). 
    \nASSISTANT:
    ''' 

        print(prompt)
        answer = inference(img_url, prompt, method, model, processor)
        
        if(method == "perception"):
            outputs.append({
                'file': img_url,
                'id': instance['id'],
                'llava_output': answer,
                'ground_truth': instance['conversations'][1]['value'],
                "metrics": instance['metrics']
            }) 
        elif(method != 'random'):
            outputs.append({
                'file': img_url,
                'id': instance['id'],
                'llava_output': answer,
                'llava_choice': parse(answer),
                'ground_truth': instance['conversations'][1]['value']
            })
        elif (method == 'random'):
            choice = parse(answer)
            if(path1_id in choice):
                c = 'Path 1'
            elif (path2_id in choice):
                c = 'Path 2'
            
        outputs.append({
                'file': img_url,
                'id': instance['id'],
                'prompt': prompt,
                'llava_output': answer,
                'llava_choice_raw': parse(answer),
                'llava_choice': c,
                'ground_truth': instance['conversations'][1]['value']
            }) 

        with open(f'output_{model_id.replace("/","-")}_{dim}_{method}.json', 'w') as f:
            obj = json.dumps(outputs, indent=3)
            f.write(obj)
            
if __name__ == '__main__':
    main()