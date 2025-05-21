import json
import sys

"""
argv: 
    outputs file
    model
"""

file = sys.argv[1]
model = sys.argv[2]

data = json.load(open(file))

def parse_outputs(output):
    output = output.split('ASSISTANT:')[1]
    output = output.lower().replace(' ','').replace('*','').replace('.','')
    if('\n' in output):
        lines = output.split('\n')
        for line in lines:
            if line == 'path1':
                return 'Path 1'
            elif line == 'path2':
                return 'Path 2'
            
    if('answer:' in output):
        
        answer = output.split('answer:')[1]
        if answer == 'path1':
            return 'Path 1'
        elif answer == 'path2':
            return 'Path 2'


correct = 0
for i in data:

    if(i[f'{model}_choice'].split('Explanation:')[0]
       .replace(' ','')
       .replace('*','')
       .replace('.','').replace('\n', '') == i['ground_truth'].replace(' ','')):
        correct += 1

print('Accuracy:', correct / len(data))