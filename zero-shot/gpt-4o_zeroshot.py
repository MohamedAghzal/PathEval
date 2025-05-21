from openai import AzureOpenAI
import base64
import time
import json
import sys
import random
import string


def generate_random_string(length=4):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


client = AzureOpenAI(
    api_key="",
    api_version='',
    azure_endpoint='',
    azure_deployment=''
)

filename = sys.argv[1]
method = sys.argv[2]
dim = sys.argv[3]
data = json.load(open(filename))

output_file = f'output_{method}_zero_shot_{dim}.json'
try:
    outputs = json.load(open(output_file))
except:
    outputs = []

idx = len(outputs)

for instance in data[idx:]:
    scenario = instance['conversations'][0]['value']
    
    if('../' not in instance['image']):
        IMAGE_PATH = f"../{instance['image']}"
    else:
        IMAGE_PATH = f"{instance['image']}" 
    encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
    if(method == 'descriptors'):
        prompt = f'''{scenario} Path 1 is on the left side and Path 2 is on the right side.
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
'''
    elif (method == 'none' or method == 'flipped'):
       prompt = f'''{scenario} Path 1 is on the left side and Path 2 is on the right side.
Your answer should follow the format below:
Answer: Path 1 or Path 2.
Explanation: Why you chose the path (1 or 2). 
'''
    elif ('point' in method):
        prompt = f'''{scenario} Point 1 is on the left side and Point 2 is on the right side.
Your answer should follow the format below:
Answer: Point 1 or Point 2.
Explanation: Why you chose the point (1 or 2). 
'''  
    elif(method == 'random'):
        path1_id = generate_random_string(4)
        path2_id = generate_random_string(4)
        
        prompt = f'''{scenario} Path {path1_id} is on the left side and {path2_id} is on the right side.
Your answer should follow the format below:
Answer: Path {path1_id} or Path {path2_id}.
Explanation: Why you chose the path ({path1_id} or {path2_id}). 
''' 
    elif ('explanation' in method):
        prompt = f'''{scenario} Path 1 is on the left side and Path 2 is on the right side.
The correct answer is {instance['conversations'][1]['value']}. Can you explain why?
'''
    elif method == "perception":
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
Metric: Name of the metric\tAnswer: Path 1 or Path 2 (which one has a lower value)\tExplanation:'''
    if (method == "no_images"):
        prompt = f'''Path 1 is on the left side and Path 2 is on the right side.
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
Question
Answer: Path 1 or Path 2.
Explanation: Why you chose the path (1 or 2). '''
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
    else:
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
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        },
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
    if("perception" in method or 'explanation' in method):
        outputs.append({
            'file': IMAGE_PATH,
            'id': instance['id'],
            'prompt': prompt,
            'gpt4o_output': answer,
            'metrics': instance["metrics"],
            'ground_truth': instance['conversations'][1]['value']
        })
    elif(method != 'random'):
        try:
            ans = answer.split('Answer: ')[1].split('Explanation:')[0]
        except:
            ans = answer
        outputs.append({
            'file': IMAGE_PATH,
            'id': instance['id'],
            'prompt': prompt,
            'gpt4o_output': answer,
            'gpt4o_choice': ans,
            'ground_truth': instance['conversations'][1]['value']
        })
    else:
       choice = answer.split('Answer: ')[1].split('Explanation:')[0] 
       if(path1_id in choice):
           c = 'Path 1'
       elif (path2_id in choice):
           c = 'Path 2'
           
       outputs.append({
           'file': IMAGE_PATH,
           'id': instance['id'],
           'prompt': prompt,
           'gpt4o_output': answer,
           'gpt4o_choice_raw': answer.split('Answer: ')[1].split('Explanation:')[0],
           'gpt4o_choice': c,
           'ground_truth': instance['conversations'][1]['value']
        }) 
    
     
    print(len(outputs))
    with open(output_file, 'w') as f:
        obj = json.dumps(outputs, indent = 4)
        f.write(obj)