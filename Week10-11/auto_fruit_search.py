# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
from path_planning.RRT import *
# import SLAM components
# sys.path.insert(0, "{}/slam".format(os.getcwd()))
# from slam.ekf import EKF
# from slam.robot import Robot
# import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure


def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['apple', 'pear', 'lemon']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id-1][0] = x
                    aruco_true_pos[marker_id-1][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints

def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters
    fileS = "calibration/param/sim/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',') # meters/tick
    fileB = "calibration/param/sim/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',') # meters

    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    waypoint_x = waypoint[0]
    waypoint_y = waypoint[1]
    robot_x = robot_pose[0]
    robot_y = robot_pose[1]
    robot_theta = robot_pose[2]
    waypoint_angle = np.arctan2((waypoint_y-robot_y),(waypoint_x-robot_x))

    angle = waypoint_angle - robot_theta

    distance = np.sqrt((waypoint_x-robot_x)**2 + (waypoint_y-robot_y)**2) #calculates distance between robot and object

    print(f'Driving from {robot_x},{robot_y} to {waypoint_x},{waypoint_y}')
    print(f'Turn {angle} and drive {distance}')

    wheel_vel = 30 #ticks
    # Convert to m/s
    left_speed_m = wheel_vel * scale
    right_speed_m = wheel_vel * scale

    # Compute the linear and angular velocity
    linear_velocity = (left_speed_m + right_speed_m) / 2.0

    # Convert to m/s
    left_speed_m = -wheel_vel * scale
    right_speed_m = wheel_vel * scale

    angular_velocity = (right_speed_m - left_speed_m) / baseline

    # turn towards the waypoint
    turn_time = abs(angle/angular_velocity)

    print("Turning for {:.2f} seconds".format(turn_time))
    if angle >= 0:
        # lv1, rv1 = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
        command1 = [0, 1]
        time1 = turn_time
    else:
        # lv1, rv1 = ppi.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
        command1 = [0, -1]
        time1 = turn_time
    # after turning, drive straight to the waypoint
    drive_time = distance/linear_velocity
    print("Driving for {:.2f} seconds".format(drive_time))
    # lv2, rv2 = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    command2 = [1,0]
    time2 = drive_time
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

    # new_robot_pose = [waypoint_x, waypoint_y, waypoint_angle]
    return [[command1,time1],[command2,time2]]

def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    robot_pose = [0,0,0]
    ####################################################

    return robot_pose

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    # LEVEL 2 CODE
    #starting robot pose
    start = np.array([0,0]) + 1.5
    robot_pose = [0,0,0]

    print(fruits_list)
    print(fruits_true_pos)
    print(search_list)
    #adding obstacles
    obstacles = []
    for x,y in aruco_true_pos:
        obstacles.append([x + 1.5, y + 1.5])


    print(obstacles)
    all_obstacles = generate_path_obstacles(obstacles)
    img = cv2.imread("path_planning/grid.png")

    paths = []
    for location in fruits_true_pos:
        #Stop in front of fruit
        if location[0] > 0 and location[1] > 0:
            location -= [0.25, 0.25]
        elif location[0] > 0 and location[1] < 0:
            location -= [0.25, -0.25]
        elif location[0] < 0 and location[1] > 0:
            location -= [-0.25, 0.25]
        else:
            location += [0.25, 0.25]

        goal = np.array(location) + 1.5


        rrtc = RRT(start=start, goal=goal, width=3, height=3, obstacle_list=all_obstacles,
                expand_dis=1, path_resolution=0.25)
        path = rrtc.planning()[::-1] #reverse path


        img = draw_obstacles(img, obstacles)
        img = draw_path(img, path)


        #printing path
        for i in range(len(path)):
            x, y = path[i]
            path[i] = [x - 1.5, y - 1.5]
        print(f'The path is {path}')

        #adding paths
        paths.append(path)

        start = np.array(goal)
    cv2.imshow('image',img)
    cv2.waitKey(0)

    # for path in paths:
    #     #driving based on path
    #     for waypoint in path[1:]:
    #         print(f'Driving to waypoint {waypoint}')
    #         robot_pose = drive_to_point(waypoint, robot_pose, ppi)
    #         print(f'Finished driving to waypoint {waypoint}')
        # time.sleep(3) #stop for 3 seconds

    #     # start = np.array(robot_pose[:2]) + 1.5 #update starting location based on robot pose




