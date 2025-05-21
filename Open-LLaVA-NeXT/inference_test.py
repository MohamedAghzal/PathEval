from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import json
import sys
import random
fine_tuned_model_path = sys.argv[1] 
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=fine_tuned_model_path,
    model_base=None,  
    model_name=get_model_name_from_path(fine_tuned_model_path),
    mm_vision_tower='openai/clip-vit-large-patch14-336'
)

dataset = sys.argv[2]

eval_dataset = json.load(open(dataset))

random.shuffle(eval_dataset)
outputs = []

correct = 0
for i in range(len(eval_dataset)):
    prompt = eval_dataset[i]['conversations'][0]['value']
    image_file = f"../{eval_dataset[i]['image']}"
    ground_truth = eval_dataset[i]['conversations'][1]['value']

    args = type('Args', (), {
        "model_path": fine_tuned_model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(fine_tuned_model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": "\n",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
        "mm_vision_tower":'openai/clip-vit-large-patch14-336'
    })()
    
    prediction = eval_model(args)
    print(prediction)
    outputs.append({
        'id': eval_dataset[i]['id'],
        'prompt': prompt,
        'image': image_file,
        'ground_truth': ground_truth,
        'prediction': prediction
    })
    
    correct += (str(prediction).lower().replace(' ', '') == ground_truth.lower().replace(' ', ''))

    with open(sys.argv[3], 'w') as f:
        obj = json.dumps(outputs, indent = 3)
        f.write(obj)
        
print('Accuracy: ', correct / len(eval_dataset))