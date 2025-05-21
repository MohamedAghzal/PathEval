# Running the Zero-Shot Experiments

This directory contains code to reproduce the zero-shot experiments.

## Formatting the data

Upon generating the data (see [path-generation](../path-generation/)), run the script ``python prepare_data.py`` to format the dataset for the experiments. Each instance should consist of the following attributes:

```
{
    "id": id of the path pair,
    "image": relative path to the image containing the two paths side by side,
    "conversations": {
        {
                "from": "human",
                "value": Scenario/Task Description
        },
        {
                "from": "gpt",
                "value": Ground truth label
        }
    }, 
    "metrics": {
            "Path 1": {
                descriptor values for path 1
            },
            "Path 2": {
                descriptor values for path 2
            }
    }
}
```

## Running the zero-shot experiments

Each model we consider in our paper is in a python file with its name. Each python script takes the following arguments:

- **model_name**: The exact name/configuration of the model you would like to run
- **test_set**: The ``json`` file you would like to run the model on
- **method**: The method that you would like to use as input to the model. We support the methods used in our paper
    
    * 'none': Prompting the model using only the images and the task description, and expecting the model to directly generate an answer
    * 'flipped': Testing on a flipped dataset (i.e. swapping image 1 and image 2)
    * 'descriptors': providing the descriptors as part of the input
    * 'random': assigning random IDs to the paths 
    * 'no_images': prompting using only the descriptors, without the image input
    * 'perception': prompting the model to provide answers based on each descriptor. 