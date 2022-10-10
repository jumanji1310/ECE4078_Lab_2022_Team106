# estimate the pose of a target object detected
import numpy as np
import json
import os
from pathlib import Path
import ast
# import cv2
import math
from machinevisiontoolbox import Image

import matplotlib.pyplot as plt
import PIL

# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(base_dir, file_path, image_poses):
    # there are at most five types of targets in each image
    target_lst_box = [[], [], [], [], []]
    target_lst_pose = [[], [], [], [], []]
    completed_img_dict = {}

    # add the bounding box info of each target in each image
    # target labels: 1 = apple, 2 = lemon, 3 = pear, 4 = orange, 5 = strawberry, 0 = not_a_target
    text_file_path = file_path.split('.')[0] + '.txt' #replacing with .txt

    #reading json data
    with open(text_file_path,"r") as f:
        data = json.load(f)
        for fruit in data:
            xmin = fruit['xmin']
            ymin = fruit['ymin']
            xmax = fruit['xmax']
            ymax = fruit['ymax']
            x = (xmin + xmax)/2
            y = (ymin + ymax)/2
            width = xmax - xmin
            height = ymax - ymin

            box = [x, y, width, height]
            pose = image_poses[file_path] #[x, y, theta]
            target_num = fruit['class']
            target_lst_box[target_num].append(box)
            target_lst_pose[target_num].append(np.array(pose).reshape(3,)) # robot pose

    # if there are more than one objects of the same type, combine them
    for i in range(5):
        if len(target_lst_box[i])>0:
            box = np.stack(target_lst_box[i], axis=1)
            pose = np.stack(target_lst_pose[i], axis=1)
            completed_img_dict[i] = {'target': box, 'robot': pose}
    return completed_img_dict

# estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(base_dir, camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    # actual sizes of targets [For the simulation models]
    # You need to replace these values for the real world objects
    target_dimensions = []
    apple_dimensions = [0.075448, 0.074871, 0.071889]
    apple_dimensions = [0, 0, 0.075] #7.5cm
    target_dimensions.append(apple_dimensions)
    lemon_dimensions = [0.060588, 0.059299, 0.053017]
    lemon_dimensions = [0, 0, 0.06] #6cm
    target_dimensions.append(lemon_dimensions)
    orange_dimensions = [0.0721, 0.0771, 0.0739]
    orange_dimensions = [0, 0, 0.08] #8cm
    target_dimensions.append(orange_dimensions)
    pear_dimensions = [0.0946, 0.0948, 0.135]
    pear_dimensions = [0, 0, 0.11] #11cm
    target_dimensions.append(pear_dimensions)
    strawberry_dimensions = [0.052, 0.0346, 0.0376]
    strawberry_dimensions = [0, 0, 0.045] #4.5cm
    target_dimensions.append(strawberry_dimensions)

    target_list = ['apple', 'lemon', 'pear', 'orange', 'strawberry']
    target_list = ['apple','lemon','orange','pear','strawberry'] #0, 1, 2, 3, 4
    target_pose_dict = {}

    # for each target in each detection output, estimate its pose
    for target_num in completed_img_dict.keys():
        for i in range(len(completed_img_dict[target_num]['target'][0])):
            box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
            robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
            true_height = target_dimensions[target_num][2]

            ######### Replace with your codes #########
            # TODO: compute pose of the target based on bounding box info and robot's pose
            target_pose = {'y': 0.0, 'x': 0.0}

            # On a 640x480 camera

            b = box[3][i] #height of object
            B = true_height #true height
            a = focal_length #focal length Fx, top left of camera matrix

            A = a*B/b #depth of object

            theta = robot_pose[2][i] #pose of robot w.r.t World Frame
            robot_x = robot_pose[0][i] #x of robot w.r.t World Frame
            robot_y = robot_pose[1][i] #y of robot w.r.t World Frame

            y = A * np.sin(theta) #y of object w.r.t to Robot frame
            x = A * np.cos(theta) #x of object w.r.t to Robot frame

            object_x = box[0][i] #x position of object in camera plane
            x_from_centre = 160 - object_x# 320/2 = 160 to get the x distance from centreline
            camera_theta = np.arctan(x_from_centre/a) #calculate angle from centreline

            total_theta = theta + camera_theta #angle of object w.r.t to Robot frame

            object_y = A * np.sin(total_theta) #object y w.r.t to Robot Frame
            object_x = A * np.cos(total_theta) #object x w.r.t to Robot Frame


            object_y_world = robot_y + object_y #object y w.r.t to World Frame
            object_x_world = robot_x + object_x #object x w.r.t to World Frame

            target_pose = {'y':object_y_world, 'x':object_x_world}

            target_pose_dict[f'{target_list[target_num]}_{i}'] = target_pose
            ###########################################

    return target_pose_dict

# Custom function to average the location of fruits
def calc_distance(x1, x2, y1, y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def average_fruit_location(fruit_est):
    while len(fruit_est) > 2:
        min_dist = 9999
        #find two points close to each other
        for i, fruit1 in enumerate(fruit_est):
            for j, fruit2 in enumerate(fruit_est):
                if (fruit1[0] != fruit2[0]) or (fruit1[1] != fruit2[1]): #if not same fruit
                    distance = calc_distance(fruit1[1],fruit2[1],fruit1[0],fruit2[0])
                    if distance < min_dist:
                        min_dist = distance
                        min1 = i
                        min2 = j
        #merge two points by averaging
        x_avg = (fruit_est[min1][1] + fruit_est[min2][1])/2 #averaging x
        y_avg = (fruit_est[min1][0] + fruit_est[min2][0])/2 #averaging y
        fruit_est = np.delete(fruit_est,(min1, min2), axis=0)
        fruit_est = np.vstack((fruit_est, [y_avg,x_avg]))
    return fruit_est

# merge the estimations of the targets so that there are at most 3 estimations of each target type
def merge_estimations(target_pose_dict):
    target_pose_dict = target_pose_dict
    apple_est, lemon_est, pear_est, orange_est, strawberry_est = [], [], [], [], []
    target_est = {}

    # combine the estimations from multiple detector outputs
    for f in target_map:
        for key in target_map[f]:
            if key.startswith('apple'):
                apple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('lemon'):
                lemon_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('pear'):
                pear_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('orange'):
                orange_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('strawberry'):
                strawberry_est.append(np.array(list(target_map[f][key].values()), dtype=float))

    ######### Replace with your codes #########
    # TODO: the operation below takes the first two estimations of each target type, replace it with a better merge solution
    if len(apple_est) > 2:
        apple_est = average_fruit_location(apple_est)
    if len(lemon_est) > 2:
        lemon_est = average_fruit_location(lemon_est)
    if len(pear_est) > 2:
        pear_est = average_fruit_location(pear_est)
    if len(orange_est) > 2:
        orange_est = average_fruit_location(orange_est)
    if len(strawberry_est) > 2:
        strawberry_est = average_fruit_location(strawberry_est)

    for i in range(2):
        try:
            target_est['apple_'+str(i)] = {'y':apple_est[i][0], 'x':apple_est[i][1]}
        except:
            pass
        try:
            target_est['lemon_'+str(i)] = {'y':lemon_est[i][0], 'x':lemon_est[i][1]}
        except:
            pass
        try:
            target_est['pear_'+str(i)] = {'y':pear_est[i][0], 'x':pear_est[i][1]}
        except:
            pass
        try:
            target_est['orange_'+str(i)] = {'y':orange_est[i][0], 'x':orange_est[i][1]}
        except:
            pass
        try:
            target_est['strawberry_'+str(i)] = {'y':strawberry_est[i][0], 'x':strawberry_est[i][1]}
        except:
            pass
    ###########################################

    return target_est


if __name__ == "__main__":
    # camera_matrix = np.ones((3,3))/2
    fileK = "{}intrinsic.txt".format('./calibration/param/physical/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')


    # a dictionary of all the saved detector outputs
    image_poses = {}
    with open(base_dir/'lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose'] # output as {'filename': [[x],[y],[theta]],'filename': [[x],[y],[theta]],...}

    # estimate pose of targets in each detector output
    target_map = {}
    for file_path in image_poses.keys():
        completed_img_dict = get_image_info(base_dir, file_path, image_poses)
        target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict)

    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    target_est = merge_estimations(target_map)

    # save target pose estimations
    with open(base_dir/'lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo)

    print('Estimations saved!')





