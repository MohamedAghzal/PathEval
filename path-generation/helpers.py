import pandas as pd
import random
import numpy as np
from shapely.geometry import Point, Polygon
import os
import json
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
import sys

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def point_to_polygon_distance(point, polygon):
    point_geom = Point(point)
    polygon_geom = Polygon(polygon)
    return point_geom.distance(polygon_geom)

def parse_obstacles(obstacle_file):
    obstacle_data = open(obstacle_file).read()
    lines = obstacle_data.strip().split('\n')
    n_polygons = int(lines[0])
    polygon_data = lines[1].split()
    
    polygons = []
    index = 0
    
    for _ in range(n_polygons):
        n_points = int(polygon_data[index])
        index += 1
        points = [(float(polygon_data[index + i * 2]), float(polygon_data[index + i * 2 + 1])) for i in range(n_points)]
        polygons.append(points)
        index += 2 * n_points

    return polygons

def calculate_clearance_metrics(path, polygons):
    min_clearance = 100000
    max_clearance = 0
    total_clearance = 0

    for point in path:
        min_distance_to_polygon = 100000
        for polygon in polygons:
            distance = point_to_polygon_distance(point, polygon)
            if distance < min_distance_to_polygon:
                min_distance_to_polygon = distance

        min_clearance = min(min_clearance, min_distance_to_polygon)
        max_clearance = max(max_clearance, min_distance_to_polygon)
        total_clearance += min_distance_to_polygon

    average_clearance = total_clearance / len(path) if path else 0

    return min_clearance, max_clearance, average_clearance

def calculate_metrics(path, polygons):
    if len(path) < 2:
        return 0, 0, 0, 0

    total_length = 0
    smoothness = 0
    sharp_turns = 0
    max_turn_angle = 0

    for i in range(1, len(path)):
        segment_length = euclidean_distance(path[i-1], path[i])
        total_length += segment_length

        if i > 1:
            v1 = np.array(path[i-1]) - np.array(path[i-2])
            v2 = np.array(path[i]) - np.array(path[i-1])
            
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            angle_deg = np.degrees(angle)

            smoothness += angle_deg

            if angle_deg > 90:
                sharp_turns += 1

            max_turn_angle = max(max_turn_angle, angle_deg)

    min_clearance, max_clearance, avg_clearance = calculate_clearance_metrics(path=path, polygons=polygons)
    
    return {
        'Minimum clearance': min_clearance,
        'Maximum clearance': max_clearance,
        'Average clearance': avg_clearance,
        'Path length': total_length, 
        'Smoothness': smoothness, 
        'Sharp turns': sharp_turns, 
        'Maximum angle': max_turn_angle,
    }