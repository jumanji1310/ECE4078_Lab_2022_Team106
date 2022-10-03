import numpy as np

fileK = "{}intrinsic.txt".format('./calibration/param/sim/')
camera_matrix = np.loadtxt(fileK, delimiter=',')

def estimate_fruit_pose(bounding_boxes, pose):
    target_lst_box = [[], [], [], [], []]
    target_lst_pose = [[], [], [], [], []]
    completed_img_dict = {}

    for fruit in bounding_boxes:
        xmin = fruit['xmin']
        ymin = fruit['ymin']
        xmax = fruit['xmax']
        ymax = fruit['ymax']
        x = (xmin + xmax)/2
        y = (ymin + ymax)/2
        width = xmax - xmin
        height = ymax - ymin

        box = [x, y, width, height]
        pose = pose #[x, y, theta]
        target_num = fruit['class']
        target_lst_box[target_num].append(box)
        target_lst_pose[target_num].append(np.array(pose).reshape(3,)) # robot pose
        completed_img_dict[target_num] = {'target': box, 'robot': pose}

    focal_length = camera_matrix[0][0]
    # actual sizes of targets [For the simulation models]
    # You need to replace these values for the real world objects
    target_dimensions = []
    apple_dimensions = [0.075448, 0.074871, 0.071889]
    target_dimensions.append(apple_dimensions)
    lemon_dimensions = [0.060588, 0.059299, 0.053017]
    target_dimensions.append(lemon_dimensions)
    orange_dimensions = [0.0721, 0.0771, 0.0739]
    target_dimensions.append(orange_dimensions)
    pear_dimensions = [0.0946, 0.0948, 0.135]
    target_dimensions.append(pear_dimensions)
    strawberry_dimensions = [0.052, 0.0346, 0.0376]
    target_dimensions.append(strawberry_dimensions)

    target_list = ['apple', 'lemon', 'pear', 'orange', 'strawberry']
    target_list = ['apple','lemon','orange','pear','strawberry'] #0, 1, 2, 3, 4
    target_pose_dict = {}

    # for each target in each detection output, estimate its pose
    for target_num in completed_img_dict.keys():
        box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
        robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
        true_height = target_dimensions[target_num][2]

        ######### Replace with your codes #########
        # TODO: compute pose of the target based on bounding box info and robot's pose
        target_pose = {'y': 0.0, 'x': 0.0}

        # On a 640x480 camera

        b = box[3] #height of object
        B = true_height #true height
        a = focal_length #focal length Fx, top left of camera matrix

        A = a*B/b #depth of object

        theta = robot_pose[2] #pose of robot w.r.t World Frame
        robot_x = robot_pose[0] #x of robot w.r.t World Frame
        robot_y = robot_pose[1] #y of robot w.r.t World Frame

        y = A * np.sin(theta) #y of object w.r.t to Robot frame
        x = A * np.cos(theta) #x of object w.r.t to Robot frame

        object_x = box[0] #x position of object in camera plane
        x_from_centre = 320 - object_x# 640/2 = 320 to get the x distance from centreline
        camera_theta = np.arctan(x_from_centre/a) #calculate angle from centreline

        total_theta = theta + camera_theta #angle of object w.r.t to Robot frame

        object_y = A * np.sin(total_theta) #object y w.r.t to Robot Frame
        object_x = A * np.cos(total_theta) #object x w.r.t to Robot Frame


        object_y_world = robot_y + object_y #object y w.r.t to World Frame
        object_x_world = robot_x + object_x #object x w.r.t to World Frame

        target_pose = {'y':object_y_world, 'x':object_x_world}

        target_pose_dict[f'{target_list[target_num]}'] = target_pose
        ###########################################
    return target_pose_dict