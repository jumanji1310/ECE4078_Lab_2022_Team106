#Put maps inside catkin src and Week08-09 folder
M4_true_map_3fruits.txt
M4_true_map_5fruits.txt
search_list.txt

#Replace calibration files in "calibration/param"

### Simulation ###
source ~/LiveDemo/catkin_ws/devel/setup.bash
roslaunch penguinpi_gazebo ECE4078.launch

#Level 1
source ~/LiveDemo/catkin_ws/devel/setup.bash
rosrun penguinpi_gazebo scene_manager.py -l "M4_true_map_5fruits.txt"

cd "ECE4078_Lab_2022_Team106/Week08-09"
python3 operate_level1.py --true_map "M4_true_map_5fruits.txt"

#Delete first map
rosrun penguinpi_gazebo scene_manager.py -d "M4_true_map_5fruits.txt"

#Level 2
source ~/LiveDemo/catkin_ws/devel/setup.bash
rosrun penguinpi_gazebo scene_manager.py -l "M4_true_map_3fruits.txt"

cd "ECE4078_Lab_2022_Team106/Week08-09"
python3 operate_level2.py --true_map "M4_true_map_3fruits.txt"

Click "A" to begin auto navigation