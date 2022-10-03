python3 operate3.py --true_map "M4_true_map_5fruits.txt"

source ~/catkin_ws/devel/setup.bash
rosrun penguinpi_gazebo scene_manager.py -l M4_true_map_5fruits.txt

#LEVEl 1
cd "ECE4078_Lab_2022_Team106/Week08-09"
python3 operate_level1.py --true_map "M4_true_map_5fruits.txt"

#LEVEL 2
cd "ECE4078_Lab_2022_Team106/Week08-09"
python3 operate_level2.py --true_map "M4_true_map_3fruits.txt"


#DELETE OBJECTS
rosrun penguinpi_gazebo scene_manager.py -d M4_true_map_5fruits.txt