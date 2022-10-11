search_list.txt in main folder
"marking_map.txt" in main folder and catkin source

source ~/LiveDemo/catkin_ws/devel/setup.bash
source ~/catkin_ws/devel/setup.bash
roslaunch penguinpi_gazebo ECE4078.launch

source ~/catkin_ws/devel/setup.bash
rosrun penguinpi_gazebo scene_manager.py -l "M4_true_map_7fruits.txt"


######## Simulation ######

### SLAM ###

(SLAM eval creates "slam_map.txt" for CV)
Run slam with enter and click "s" to save map

cd ECE4078_Lab_2022_Team106/Week10-11/
python3 slam_operate.py

python3 SLAM_eval.py <groundtruth> <estimate>
python3 SLAM_eval.py "M4_true_map_7fruits.txt" "lab_output/slam.txt"



### CV ###
cd ECE4078_Lab_2022_Team106/Week10-11/
python3 cv_operate.py --ckpt yolo-sim.pt 

cd "M3_106/ECE4078_Lab_2022_Team106/Week10-11/"
python3 TargetPoseEstSim_final.py

python3 CV_eval.py --truth M4_true_map_7fruits.txt --est lab_output/targets.txt

### Navigation ###
cd ECE4078_Lab_2022_Team106/Week10-11/
python3 operate_level1.py (--true_map "combined_map.txt")
python3 operate_level2.py (--true_map "combined_map.txt") (Z to generate new map, A to start route)

### Physical ###
copy physical param to main param folder

### SLAM ###
(SLAM eval creates "slam_map.txt" for CV)

cd ECE4078_Lab_2022_Team106/Week10-11/
python3 slam_operate.py --ip 192.168.50.1 --port 8080

python3 SLAM_eval.py <groundtruth> <estimate>
python3 SLAM_eval.py "M4_true_map_7fruits.txt" "lab_output/slam.txt"

### CV ###

cd ECE4078_Lab_2022_Team106/Week10-11/
python3 TargetPoseEstSim_final.py

python3 CV_eval.py --truth M4_true_map_7fruits.txt --est lab_output/targets.txt
(CV eval creates the new combined_map.txt with aruco and fruits)
### Navigation ###
cd ECE4078_Lab_2022_Team106/Week10-11/
python3 operate_level1.py --ip 192.168.50.1 --port 8080
python3 operate_level2.py --ip 192.168.50.1 --port 8080