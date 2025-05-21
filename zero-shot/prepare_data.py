import json
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import random
import sys

dataset_folder = sys.argv[1]
def combine_images_side_by_side(image_path1, image_path2):
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    
    if image1.height != image2.height:
        new_height = min(image1.height, image2.height)
        image1 = image1.resize((int(image1.width * new_height / image1.height), new_height), Image.ANTIALIAS)
        image2 = image2.resize((int(image2.width * new_height / image2.height), new_height), Image.ANTIALIAS)
    
    total_width = image1.width + image2.width
    max_height = max(image1.height, image2.height)
    combined_image = Image.new('RGB', (total_width, max_height))
    
    combined_image.paste(image1, (0, 0))
    
    combined_image.paste(image2, (image1.width, 0))
    
    return combined_image

def format_dataset(path, img_root='../images', dim='2D'):
    data = json.load(open(path))

    formatted_data = []

    images = os.listdir(img_root)

    for sample in data:
        if(sample['annotation'] == 'Unresolved'):
            continue

        img1 = img_root + '/' + sample['Path 1']['file']
        img1_tokens = img1.split('_')[:-2]
        img1 = '_'.join(img1_tokens) + f"_{dim}_{sample['Path 1']['id']}_{img1.split('_')[-1]}"

        img2 = img_root + '/' + sample['Path 2']['file']
        img2_tokens = img2.split('_')[:-2]
        img2 = '_'.join(img2_tokens) + f"_{dim}_{sample['Path 2']['id']}_{img2.split('_')[-1]}"

        print(img1, img2)
        if(img1.replace(f"{img_root}/", '') not in images or img2.replace(f"{img_root}/", '') not in images):
            continue
        

        formatted_data.append(
            (
                img1,
                img2,
                sample['Scenario'],
                int(sample['annotation'].replace('Path ', '')) - 1,
                sample['Path 1']['metrics'],
                sample['Path 2']['metrics']  
            )
        )
    
    return formatted_data

def format_llava_data(data, dim='2D'):
    data_llava = []

    for i in range(len(data)):
        data_llava.append({
            "id": i,
            "image": f'combined_images/{i}_{dim}.png',
            "conversations": [
                {
                    "from": "human",
                    "value": f"{data[i][2]}. Which path better achieves the task?"
                },
                {
                    "from": "gpt",
                    "value": f"Path {data[i][3] + 1}"
                }
            ], 
            "metrics":  {
                "Path 1": data[i][4],
                "Path 2": data[i][5]
            }
        })
    
    return data_llava

_data = format_dataset(dataset_folder, f'{dataset_folder}/images', dim='2D')

cnt_sc = {}

for inst in _data:
    if(inst[2] in cnt_sc):
        cnt_sc[inst[2]].append(inst)
    else:
        cnt_sc[inst[2]] = [inst] 

data__sampled = []

counts = []

for sc in cnt_sc.keys():
    counts.append(len(cnt_sc[sc])) 

counts = sorted(counts)[-5:]
      
for sc in cnt_sc.keys():
    n = 33 if len(cnt_sc[sc]) not in counts else 34

    sampled = random.sample(cnt_sc[sc], n)
    for s in sampled:
        data__sampled.append(s)

i = 0
for instance in data__sampled:
    combined_image = combine_images_side_by_side(instance[0], instance[1])
    combined_image.save(f"combined_images/{i}_2D.png")
    combined_image = combine_images_side_by_side(instance[0].replace('2D', '3D'), instance[1].replace('2D', '3D'))
    combined_image.save(f"combined_images/{i}_3D.png")

    i += 1

llava_data = format_llava_data(data__sampled, dim='2D')

with open(f'{dataset_folder}_dataset_2D_formatted.json', 'w') as f:
    obj = json.dumps(llava_data, indent=4)
    f.write(obj)

llava_data_3D = format_llava_data(data__sampled, dim='3D')

with open(f'{dataset_folder}_auto_dataset_3D_formatted.json', 'w') as f:
    obj = json.dumps(llava_data_3D, indent=4)
    f.write(obj)