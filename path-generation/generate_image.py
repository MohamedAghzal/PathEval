import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
import math
import matplotlib.pyplot as plt
import sys
import os
import json

def load_polygon_data(file_path):
    with open(file_path, 'r') as f:
        data = f.read().split()
    
    num_polygons = int(data[0])
    polygons = []
    index = 1

    for _ in range(num_polygons):
        num_points = int(data[index])
        index += 1
        points = []
        for _ in range(num_points):
            x = float(data[index])
            y = float(data[index + 1])
            index += 2
            points.append((x, y))
        polygons.append(points)
    
    return polygons

# Function to initialize OpenGL
def init_gl(width, height, near_plane, far_plane):
    glViewport(0, 0, width, height)
    glClearColor(1.0, 1.0, 1.0, 1.0)  # White background
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glShadeModel(GL_SMOOTH)

    # Set the polygon mode to fill
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width / float(height), near_plane, far_plane)  # FOV, Aspect Ratio, Near and Far Planes
    glMatrixMode(GL_MODELVIEW)

# Function to draw the floor as a gray plane
def draw_floor(min_x, max_x, min_y, max_y, color):
    glColor3f(color[0], color[1], color[2])  
    glBegin(GL_QUADS)
    glVertex3f(min_x, min_y, 0.0)
    glVertex3f(max_x, min_y, 0.0)
    glVertex3f(max_x, max_y, 0.0)
    glVertex3f(min_x, max_y, 0.0)
    glEnd()

# Function to draw the extruded polygons with a texture
def draw_extruded_polygons(polygons, height, texture_id):

    glEnable(GL_TEXTURE_2D)  # Enable 2D texturing for obstacles
    glBindTexture(GL_TEXTURE_2D, texture_id)  # Bind the texture

    glColor3f(1.0, 1.0, 1.0)  # White color to use the texture's color

    for polygon in polygons:
        num_vertices = len(polygon)

        # Draw the side walls with texture
        glBegin(GL_QUAD_STRIP)
        for (x, y) in polygon:
            glTexCoord2f(x / 10.0, 0.0)
            glVertex3f(x, y, 0.0)  # Bottom vertex
            glTexCoord2f(x / 10.0, 1.0)
            glVertex3f(x, y, height)  # Top vertex
        # Close the strip by connecting the last point to the first
        (x, y) = polygon[0]
        glTexCoord2f(x / 10.0, 0.0)
        glVertex3f(x, y, 0.0)  # Close strip bottom
        glTexCoord2f(x / 10.0, 1.0)
        glVertex3f(x, y, height)  # Close strip top
        glEnd()

        # Use GLU tessellation to fill the enclosed area properly
        tess = gluNewTess()

        def tess_vertex(vertex):
            glVertex3f(vertex[0], vertex[1], height)

        gluTessCallback(tess, GLU_TESS_VERTEX, tess_vertex)
        gluTessCallback(tess, GLU_TESS_BEGIN, glBegin)
        gluTessCallback(tess, GLU_TESS_END, glEnd)
        gluTessCallback(tess, GLU_TESS_ERROR, lambda error: None)

        gluTessBeginPolygon(tess, None)
        gluTessBeginContour(tess)

        for (x, y) in polygon:
            gluTessVertex(tess, (x, y, height), (x, y, height))

        gluTessEndContour(tess)
        gluTessEndPolygon(tess)

        gluDeleteTess(tess)

    glDisable(GL_TEXTURE_2D)

# Function to draw the path with elevation
def draw_path(path_coords, path_elevation):
    glColor3f(1.0, 0.0, 0.0)  # Red color for the path
    glLineWidth(3.5)  # Increase path thickness
    glBegin(GL_LINE_STRIP)
    for (x, y) in path_coords:
        glVertex3f(x, y, path_elevation)  # Use consistent elevation for path
    glEnd()
    glLineWidth(1.0)  # Reset line width to default

def draw_maze_boundary(min_x, max_x, min_y, max_y, height, texture_id):
    # Define the boundary polygons as four separate walls
    boundary_polygons = [
        # Bottom boundary
        [(min_x, min_y), (max_x, min_y), (max_x, min_y + 0.5), (min_x, min_y + 0.5)],
        # Top boundary
        [(min_x, max_y - 0.5), (max_x, max_y - 0.5), (max_x, max_y), (min_x, max_y)],
        # Left boundary
        [(min_x, min_y), (min_x, max_y), (min_x + 0.5, max_y), (min_x + 0.5, min_y)],
        # Right boundary
        [(max_x - 0.5, min_y), (max_x - 0.5, max_y), (max_x, max_y), (max_x, min_y)],
    ]

    glEnable(GL_TEXTURE_2D)  # Enable 2D texturing
    glBindTexture(GL_TEXTURE_2D, texture_id)  # Bind the texture for the obstacles

    glColor3f(1.0, 1.0, 1.0)  # Set color to white to use the texture's color

    for polygon in boundary_polygons:
        # Draw the top face of the boundary polygon
        glBegin(GL_POLYGON)
        for (x, y) in polygon:
            glTexCoord2f(x / 10.0, y / 10.0)  # Map texture coordinates
            glVertex3f(x, y, height)  # Top face at given height
        glEnd()

        # Draw the bottom face of the boundary polygon
        glBegin(GL_POLYGON)
        for (x, y) in polygon:
            glTexCoord2f(x / 10.0, y / 10.0)  # Map texture coordinates
            glVertex3f(x, y, 0.0)  # Bottom face at z = 0
        glEnd()

        # Draw the side faces to elevate the edges of the boundary
        glBegin(GL_QUAD_STRIP)
        for (x, y) in polygon:
            glTexCoord2f(x / 10.0, 0.0)
            glVertex3f(x, y, 0.0)  # Bottom vertex at z = 0
            glTexCoord2f(x / 10.0, 1.0)
            glVertex3f(x, y, height)  # Top vertex at the specified height
        # Close the strip by connecting the last point to the first
        (x, y) = polygon[0]
        glTexCoord2f(x / 10.0, 0.0)
        glVertex3f(x, y, 0.0)  # Close strip bottom
        glTexCoord2f(x / 10.0, 1.0)
        glVertex3f(x, y, height)  # Close strip top
        glEnd()

    glDisable(GL_TEXTURE_2D)  # Disable texturing after drawing the boundaries

# Function to draw start and goal points
def draw_start_goal(start, goal, radius=1.5, num_segments=100):
    glBegin(GL_POLYGON)
    glColor3f(1.0, 0.0, 0.0)  # Red color for the start point
    for i in range(num_segments):
        theta = 2.0 * math.pi * i / num_segments  # Angle for this segment
        x = radius * math.cos(theta) + start[0]
        y = radius * math.sin(theta) + start[1]
        glVertex3f(x, y, start[2])  # Draw vertex on the start point plane
    glEnd()

    glBegin(GL_POLYGON)
    glColor3f(0.0, 0.0, 1.0)  # Blue color for the goal point
    for i in range(num_segments):
        theta = 2.0 * math.pi * i / num_segments  # Angle for this segment
        x = radius * math.cos(theta) + goal[0]
        y = radius * math.sin(theta) + goal[1]
        glVertex3f(x, y, goal[2])  # Draw vertex on the goal point plane
    glEnd()

# Function to save the rendered image to a file
def save_image(filename, width, height):
    # Read pixels from the frame buffer
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    
    # Convert to an image
    image = Image.frombytes("RGB", (width, height), pixels)
    
    # Flip the image vertically (OpenGL origin is at the bottom left)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Save the image
    image.save(filename)

def load_texture(file_path):
    texture_image = Image.open(file_path)
    texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip image vertically for OpenGL

    texture_data = texture_image.convert("RGBA").tobytes()  # Convert to RGBA format
    width, height = texture_image.size

    texture_id = glGenTextures(1)  # Generate texture ID
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)  # Wrap texture on S axis
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)  # Wrap texture on T axis
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)  # Linear filtering for minification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)  # Linear filtering for magnification

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

    return texture_id

def generate_2D_image(data_file, example_id, path_coordinates, output_dir):
    # Clear the current figure to avoid overlaying paths when the function is called multiple times
    plt.clf()

    data = open(data_file).read()
    data_list = list(map(float, data.split()))

    num_polygons = int(data_list[0])

    index = 1
    

    start_point = (-36.6031, -36.4332)
    goal_point = (36.0982, 36.363)


    plt.scatter(*start_point, color='red', s=100, label='Start Point')  
    plt.scatter(*goal_point, color='blue', s=100, label='Goal Point')  

    for _ in range(num_polygons):
        num_points = int(data_list[index])
        index += 1

        x_coords = data_list[index:index + num_points * 2:2]
        y_coords = data_list[index + 1:index + num_points * 2:2]
        index += num_points * 2
        
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        
        plt.fill(x_coords, y_coords, color='black')

    path_x, path_y = zip(*path_coordinates)
    plt.plot(path_x, path_y, color='red', linewidth=2, linestyle='-', label='Path')

    min_x, max_x = -40, 40  
    min_y, max_y = -40, 40  
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    plt.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], color='black')

    plt.gca().set_aspect('equal')
    plt.gca().set_facecolor('white')
    plt.axis('off')

    plt.gca().margins(0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    img_path = data_file.split('/')[-1].replace('.txt', '')
    plt.savefig(f'{output_dir}/images/{img_path}_2D_{example_id}.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_3D_image(data_file, example_id, path_coordinates, output_dir, params=None):
    if(params == None):
        params = {
            'GraphicsResolutionX': 1280,
            'GraphicsResolutionY': 720,
            'GraphicsPerspectiveNearPlane': 1,
            'GraphicsPerspectiveFarPlane': 10000,
            'GraphicsCameraEyeX': 0.0,
            'GraphicsCameraEyeY': -70.0,  # Adjust for zoom in
            'GraphicsCameraEyeZ': 55.0,  # Adjust Z to change vertical perspective
            'GraphicsCameraCenterX': 0.0,
            'GraphicsCameraCenterY': 0.0,
            'GraphicsCameraCenterZ': -10.0,
            'SceneObstaclesHeight': 2,  # Increased obstacle height
            'PathElevation': 0.03,  # Consistent elevation for path
            'ObstacleTextureFile': 'textures/terrain5.ppm',  # Replace with your actual obstacle texture file path
            'FloorTexture': (0.7, 0.7, 0.7),
            'SceneGridMinX': -40.0,
            'SceneGridMaxX': 40.0,
            'SceneGridMinY': -40.0,
            'SceneGridMaxY': 40.0,
            'StartPoint': (-36.6031, -36.4332, 0.03),  # Start location
            'GoalPoint': (36.0982, 36.363, 0.03)  # Goal location
        }
    
    pygame.init()
    display = (params['GraphicsResolutionX'], params['GraphicsResolutionY'])
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    init_gl(display[0], display[1], params['GraphicsPerspectiveNearPlane'], params['GraphicsPerspectiveFarPlane'])

    polygons = load_polygon_data(data_file)

    obstacle_texture_id = load_texture(params['ObstacleTextureFile'])

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    gluLookAt(params['GraphicsCameraEyeX'], params['GraphicsCameraEyeY'], params['GraphicsCameraEyeZ'],
              params['GraphicsCameraCenterX'], params['GraphicsCameraCenterY'], params['GraphicsCameraCenterZ'],
              0, 1, 0)  # Up vector

    draw_floor(params['SceneGridMinX'], params['SceneGridMaxX'], params['SceneGridMinY'], params['SceneGridMaxY'], color=params['FloorTexture'])

    draw_extruded_polygons(polygons, params['SceneObstaclesHeight'], obstacle_texture_id)
    draw_maze_boundary(params['SceneGridMinX'], params['SceneGridMaxX'], params['SceneGridMinY'], params['SceneGridMaxY'],
                       params['SceneObstaclesHeight'], texture_id=obstacle_texture_id)

    draw_path(path_coordinates, params['PathElevation'])  # Draw path with consistent elevation

    draw_start_goal(params['StartPoint'], params['GoalPoint'])

    img_path = data_file.split('/')[-1].replace('.txt','')
    save_image(f'{output_dir}/images/{img_path}_3D_{example_id}.png', display[0], display[1])

    pygame.quit()

def main():
    data_folder = sys.argv[1]

    for file in os.listdir(data_folder):
        if not 'json' in file:
            continue
        env_data = json.load(open(f"{data_folder}/{file}"))
        for dp in env_data:
            generate_2D_image(
                path_coordinates=dp['Path 1']['path'],
                example_id=dp['Path 1']['id'],
                output_dir=data_folder,
                data_file=f"../data/{dp['Path 1']['file']}"
            )
            
            generate_2D_image(
                path_coordinates=dp['Path 2']['path'],
                example_id=dp['Path 2']['id'],
                output_dir=data_folder,
                data_file=f"../data/{dp['Path 2']['file']}"
            )
            
            generate_3D_image(
                path_coordinates=dp['Path 1']['path'],
                example_id=dp['Path 1']['id'],
                output_dir=data_folder,
                data_file=f"../data/{dp['Path 1']['file']}"
            )
            
            generate_3D_image(
                path_coordinates=dp['Path 2']['path'],
                example_id=dp['Path 2']['id'],
                output_dir=data_folder,
                data_file=f"../data/{dp['Path 2']['file']}"
            )
        
if __name__ == "__main__":
    main()