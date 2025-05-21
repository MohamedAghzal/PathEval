import pandas as pd
import random
import numpy as np
from shapely.geometry import Point, Polygon
import os
import json
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
import sys
from helpers import *

def read_data(filename):
    paths_str = open(filename).read().split('@')
    
    goal = (36.0982, 36.363) 
    paths = []
    for path in paths_str:
        p = path.split('\n')
        
        pp = []
        for p_ in p:
            a = p_.split(' ')
            if(len(a) < 2 or 'Robot' in a[0]):
                continue
            l = a[0]
            r = a[1]
            pp.append((
                float(l),
                float(r)
            ))
        
        valid = False
        if(len(pp)):
            point = pp[-1]
            if(abs(point[0] - goal[0]) < 1.5 and abs(point[1] - goal[1]) < 1.5):
                valid = True
        if(valid):
            paths.append(pp)
    
    print(f"Number of Valid Paths in {filename}: ", len(paths))
    return paths    

def build_pairs(root, file, num_pairs):
    full_path = os.path.join(root, file)

    num_paths = 2 * num_pairs
    paths = read_data(full_path)  

    metrics_arr = []
    metrics_dict = []
    for path in paths:
        mt = calculate_metrics(path, parse_obstacles(f'../data/{file}'))
        metrics_dict.append(mt)
        metrics_arr.append([mt[k] for k in mt.keys()])
        
        
    if len(metrics_arr) < 2:
        print(f'{file} has no valid pairs.')
        return None 

    data = np.array(metrics_arr)
    
    if len(data) < 2:
        print(f'Not enough valid files to compute distances after filtering {file}.')
        return []

    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    pairwise_distances = squareform(pdist(data_normalized, metric='euclidean'))
    sorted_indices = np.argsort(-pairwise_distances, axis=None)

    row_indices, col_indices = np.unravel_index(sorted_indices, pairwise_distances.shape)

    selected_indices = []
    selected_paths = set()

    for r, c in zip(row_indices, col_indices):
        if r not in selected_paths and c not in selected_paths:
            selected_indices.extend([r, c])
            selected_paths.update([r, c])
        if len(selected_indices) >= num_paths:
            break

    dataset = []
    for i in range(0, len(selected_indices), 2):
        dataset.append(
            {
                "Path 1": {
                    "file": file,
                    "id": int(selected_indices[i]),
                    "metrics": metrics_dict[selected_indices[i]],
                    "path": paths[selected_indices[i]]
                },
                "Path 2": {
                    "file": file,
                    "id": int(selected_indices[i+1]),
                    "metrics": metrics_dict[selected_indices[i+1]],
                    "path": paths[selected_indices[i+1]] 
                }
            }
        )

    return dataset

scenarios_df = pd.read_excel('Scenarios.xlsx')

paths_source = 'paths-rrt/' 
paths_dest = sys.argv[1]
paths_with_metrics = {}

total = 0
for file in os.listdir(f'{paths_source}'):
    paths_with_metrics[file] = build_pairs(paths_source, file, num_pairs=6)
    if(paths_with_metrics[file] == None):
        continue
    
    cc = 0
    for pair in paths_with_metrics[file]:
        p1 = pair['Path 1']
        p2 = pair['Path 2']
        
        annotated_dataset = []
        for i in range(scenarios_df.shape[0]):
            scenario = scenarios_df.iloc[i][0]
            sc_metrics = scenarios_df.iloc[i][1:]
                
            p1_win = []
            p2_win = []
            sample = {}
            sample['Path 1'] = p1
            sample['Path 2'] = p2
            sample['Scenario'] = scenario
            sample['file'] = file
            
            for metric in sc_metrics.keys():
                if sc_metrics[metric] == 0:
                    continue
                
                thresholds_file = open(f"{paths_dest}/params.txt").read().split("\n")
                
                ms = [
                    "Minimum clearance",
                    "Maximum clearance",
                    "Average clearance",
                    "Path length",
                    "Smoothness",
                    "Sharp turns",
                    "Maximum angle"   
                ]
                
                thresh = {}
                for line in thresholds_file:
                    for m in ms:
                        if m in line:
                            value = float(line.split(': ')[1])
                            thresh[m] = value
                
                cc += 1
                
                if(abs(p1['metrics'][metric] - p2['metrics'][metric]) < thresh[metric]):
                    continue
                
                if sc_metrics[metric] == -1:
                    if(p1['metrics'][metric] < p2['metrics'][metric]):
                        p1_win.append(metric)
                    elif (p1['metrics'][metric] > p2['metrics'][metric]):
                        p2_win.append(metric) 
                elif sc_metrics[metric] == 1:
                    if(p1['metrics'][metric] > p2['metrics'][metric]):
                        p1_win.append(metric)
                    elif (p1['metrics'][metric] < p2['metrics'][metric]):
                        p2_win.append(metric) 
        
            if(len(p2_win) == 0 and len(p1_win) != 0):
                sample['annotation'] = 'Path 1'
            elif (len(p1_win) == 0 and len(p2_win) != 0 ):
                sample['annotation'] = 'Path 2'
            else:
                sample['annotation'] = 'Unresolved'
                sample['Reason'] = {
                    'P1': p1_win,
                    'P2': p2_win
                }
                continue
                
            annotated_dataset.append(sample)
    
    total += len(annotated_dataset)
    print(f'{len(annotated_dataset)} samples in {file}')
    if(len(annotated_dataset) > 0):
        with open(f"{paths_dest}/{file.replace('.txt', '.json')}", 'w') as f:
            f.write(json.dumps(annotated_dataset, indent=4))


print("Total Number of Samples:", total)