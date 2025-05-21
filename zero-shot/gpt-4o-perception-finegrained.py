from openai import AzureOpenAI
import base64
import time
import json
import sys
import random
import string

client = AzureOpenAI(
    api_key="",
    api_version='',
    azure_endpoint='',
    azure_deployment=''
)

dataset = sys.argv[1]
dim = sys.argv[2]

data = json.load(open(f"{dataset}"))

print(dataset.split("/")[-1])
output_file = f'output_gpt4o_{dataset.split("/")[-1]}_zero_shot_{dim}.json'

try:
    outputs = json.load(open(output_file))
except:
    outputs = []

idx = len(outputs)

for instance in data[idx:]:
    scenario = instance['conversations'][0]['value']
    
    IMAGE_PATH = f"{instance['image'].replace('.png', f'_{dim}.png')}"
    encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
    if "Minimum clearance" in scenario:
        add = "A smaller minimum clearance value means that the path gets closer to the obstacles."
    elif "Maximum clearance" in scenario:
        add = "A smaller maximum clearance value means that the path gets less far away to the obstacles." 
    elif "Average clearance" in scenario:
        add = "A smaller average clearance value means that the path stays closer to the obstacles on average."
    elif "Smoothness" in scenario:
        add = "A smaller smoothness value means that the path has fewer abrupt turns and is smoother overall."
    elif "Maximum angle" in scenario:
        add = "A smaller maximum turn angle means that the sharpest angle on the path is smaller"
    elif "Path length" in scenario:
        add = "A smaller path length means the path is shorter in terms of total distance traveled."
    elif "Sharp turns" in scenario:
        add = "A smaller number of sharp turns means the path contains fewer turns that are greater than 90 degrees."
 
    prompt = f'''{scenario.split('..')[0]+'.'} 
Path 1 is on the left side and Path 2 is on the right side.
The task is to determine which path results in a numerically smaller number. Compare Path 1 and Path 2, and choose the path with the smaller value. {add}
Your answer should follow this exact format:
Answer: Path 1 or Path 2.
Explanation: Briefly explain why you chose the path (e.g., Path 1 has a smaller value for the given metric).
''' 
    print(prompt)
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are useful assistant in the field of path planning and navigation."
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
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )    
    except Exception as e:
        print(f"An error occurred: {e}")
        
        time.sleep(30)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    
    answer = response.choices[0].message.content
    outputs.append({
            'file': IMAGE_PATH,
            'id': instance['id'],
            'prompt': prompt,
            'gpt4o_output': answer,
            'metrics': instance["metrics"],
            'ground_truth': instance['conversations'][1]['value']
    })
        
    print(len(outputs))
    with open(output_file, 'w') as f:
        obj = json.dumps(outputs, indent = 4)
        f.write(obj)