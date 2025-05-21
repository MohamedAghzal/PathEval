from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import sys
import json
import random
import string
from PIL import Image

def inference(messages, model, processor, method=''):
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
            
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=2056)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

def generate_random_string(length=4):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def load_blank_image(input_size=448):
    image = Image.new('RGB', (input_size, input_size), (255, 255, 255))
    return image

def parse(answer):
    #assistant = answer.split('ASSISTANT:')[1]
    answer = answer[0]
    try:
        if('Answer:' in answer):
            answer = answer.split('Answer:')[1]
        if('Explanation:' in answer):
            answer = answer.split('Explanation:')[0]
    except:
        pass
    
    return answer

cache_dir = "./new_cache_dir"

def main():
    model_name = sys.argv[1] #"Qwen/Qwen2-VL-72B-Instruct"
    test_set = sys.argv[2]

    method = sys.argv[3]
    dim = sys.argv[4]
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        cache_dir=cache_dir  # Specify the custom cache directory
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    test_data = json.load(open(test_set)) 

    outputs = []
    for instance in test_data:
        if('../' not in instance['image']):
            img_url = f"../{instance['image']}"
        else:
            img_url = f"{instance['image']}" 
        scenario = instance['conversations'][0]['value']

        if(method == 'descriptors'):
            prompt = f'''{scenario} Path 1 is on the left side and Path 2 is on the right side.
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
    Answer: Path 1 or Path 2.\n
    Explanation: Why you chose the path (1 or 2). 
    '''
        elif(method == 'none' or method == 'flipped'):
            prompt = f'''{scenario} Path 1 is on the left side and Path 2 is on the right side.
    Your answer should follow the format below:
    Answer: Path 1 or Path 2.\n
    Explanation: Why you chose the path (1 or 2). 
        '''
        
        elif (method == 'random'):
            path1_id = generate_random_string(4)
            path2_id = generate_random_string(4)
            
            prompt = f'''{scenario} Path {path1_id} is on the left side and {path2_id} is on the right side.
    Your answer should follow the format below:
    Answer: Either Path {path1_id} or Path {path2_id}.
    Explanation: Why you chose the path ({path1_id} or {path2_id}). 
    '''  

        elif (method == 'perception'):
            prompt = f'''Path 1 is on the left side and Path 2 is on the right side.
    You are asked to compare the two paths in terms of the following metrics:
    1. Minimum Clearance: The minimum distance from the obstacles.
    2. Maximum Clearance: The maximum distance from the obstacles.
    3. Average Clearance: The average distance from the obstacles. 
    4. Smoothness: The sum of absolute angles between path segments. Smoother paths have a lower smoothness value.
    5. Number of sharp turns: The number of turns that are >90 degrees.
    6. Maximum turn angle: The sharpest turn angle in the path.
    7. Path length: The sum of Euclidean distances between points in the path.

    Please choose the better path for each metric.
    Your answer should follow the format below:
    Metric: Name of the metric\tAnswer: Path 1 or Path 2 (which one has a lower value)\tExplanation:.
    ''' 
        elif (method == "no_image"):
            prompt = f'''{scenario} Path 1 is on the left side and Path 2 is on the right side.
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
    Explanation: Why you chose the path (1 or 2).'''
            img_url = 'blank_image.png'

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_url,
                    },
                    {"type": "text", "text":prompt},
                ],
            }
        ]
        answer = inference(messages, model, processor)

        print(parse(answer), instance['conversations'][1]['value'].replace(' ', ''))
        
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
                'ground_truth': instance['conversations'][1]['value'],
            })
        else:
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

        with open(f'{model_name.replace("/","-")}_{dim}_{method}.json', 'w') as f:
            obj = json.dumps(outputs, indent=3)
            f.write(obj)