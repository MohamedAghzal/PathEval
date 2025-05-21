import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
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
import math


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def generate_random_string(length=4):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB').resize((256, 256))
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_blank_image(input_size=448):
    # Create a blank white image
    image = Image.new('RGB', (input_size, input_size), (255, 255, 255))
    transform = build_transform(input_size)
    pixel_values = transform(image)
    return torch.stack([pixel_values])

def inference(url, prompt,tokenizer, model, input_size=448, max_num=1000, method=''):
    if method != "no_images" and method != 'level1':
        pixel_values = load_image(url, input_size=input_size, max_num=max_num).to(torch.bfloat16).cuda()
        question = f'<image>\n{prompt}'
    else:
        pixel_values = load_image('blank_image.png', input_size=input_size, max_num=max_num).to(torch.bfloat16).cuda()
        question = prompt  # For text-only processing, we skip image loading.

    generation_config = dict(max_new_tokens=4000, do_sample=True)
    if method != "no_images" and method != 'level1':
        response = model.chat(tokenizer, pixel_values, question, generation_config)
    else:
        response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response

def parse(answer):    
    try:
        if('Answer:' in answer):
            answer = answer.split('Answer:')[1]
        if('Explanation:' in answer):
            answer = answer.split('Explanation:')[0]
    except:
        pass
    
    return answer

def main():
    model_name = sys.argv[1] 
    test_set = sys.argv[2]

    method = sys.argv[3]
    dim = sys.argv[4]

    path = model_name

    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        cache_dir='./new_cache_dir'
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    test_data = json.load(open(test_set)) 


    outputs = []
    for instance in test_data:
        img_url = f"{instance['image']}"
        scenario = instance['conversations'][0]['value']

        if(method == 'descriptors'):
            prompt = f'''<image>\n{scenario} Path 1 is on the left side and Path 2 is on the right side.
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
            prompt = f'''<image>\n{scenario} Path 1 is on the left side and Path 2 is on the right side.
    Your answer should follow the format below:
    Answer: Path 1 or Path 2.\n
    Explanation: Why you chose the path (1 or 2). 
        '''
        
        elif (method == 'random'):
            path1_id = generate_random_string(4)
            path2_id = generate_random_string(4)
            
            prompt = f'''<image>\n{scenario} Path {path1_id} is on the left side and {path2_id} is on the right side.
    Your answer should follow the format below:
    Answer: Either Path {path1_id} or Path {path2_id}.
    Explanation: Why you chose the path ({path1_id} or {path2_id}). 
    '''  

        elif (method == 'perception'):
            prompt = f'''<image>\nPath 1 is on the left side and Path 2 is on the right side.
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
        elif (method == "no_images"):
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

        answer = inference(img_url, prompt=prompt, model=model, tokenizer=tokenizer, method=method)

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
    
if __name__ == "__main__":
    main()