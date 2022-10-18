search_list.txt in main folder
"M5_true_map.txt" in main folder and catkin source

######## Simulation ######

#### Launching sim ###

source ~/catkin_ws/devel/setup.bash
roslaunch penguinpi_gazebo ECE4078.launch

source ~/catkin_ws/devel/setup.bash
rosrun penguinpi_gazebo scene_manager.py -l "M5_true_map.txt"


### SLAM ###

Run slam with enter and click "s" to save map (also creates "slam_map.txt" for CV)

cd M5_106/ECE4078_Lab_2022_Team106/Week10-11/
python3 slam_operate.py

python3 SLAM_eval.py "M5_true_map.txt"

### CV ###
cd M5_106/ECE4078_Lab_2022_Team106/Week10-11/
python3 cv_operate.py --ckpt yolo-sim.pt 

cd M5_106/ECE4078_Lab_2022_Team106/Week10-11/
python3 TargetPoseEst.py --type "sim" (also creates "combined_map.txt" for the path)

### Final EVAL ###
python3 mapping_eval.py --true-map "M5_true_map.txt"

### Navigation ###
cd M5_106/ECE4078_Lab_2022_Team106/Week10-11/
python3 operate_level1.py

python3 operate_level2.py 
(Z to generate new map, A to start route)




### Physical ###
copy physical param to main param folder

### SLAM ###

cd M5_106/ECE4078_Lab_2022_Team106/Week10-11/
python3 slam_operate.py --ip 192.168.50.1 --port 8080

python3 SLAM_eval.py "M5_true_map.txt"

### CV ###
cd M5_106/ECE4078_Lab_2022_Team106/Week10-11/
python3 cv_operate.py --ckpt yolo-phy.pt --ip 192.168.50.1 --port 8080

cd M5_106/ECE4078_Lab_2022_Team106/Week10-11/
python3 TargetPoseEst.py --type "phy" (also creates "combined_map.txt" for the path)

### Final EVAL ###
python3 mapping_eval.py --true-map "M5_true_map.txt"

### Navigation ###
cd M5_106/ECE4078_Lab_2022_Team106/Week10-11/
python3 operate_level1.py --ip 192.168.50.1 --port 8080
python3 operate_level2.py --ip 192.168.50.1 --port 8080

