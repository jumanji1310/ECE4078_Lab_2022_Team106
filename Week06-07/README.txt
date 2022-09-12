

<-- Setup for yolov5 -->

### Run the following commands to initalise environement for yolov5 ###
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install

#########################################################

<-- SIMULATION -->

<-- Launch environment -->
source ~/catkin_ws/devel/setup.bash
roslaunch penguinpi_gazebo ECE4078_brick.launch

<-- Load map (REPLACE FruitMap.txt with MarkingMap.txt) -->
source ~/catkin_ws/devel/setup.bash
rosrun penguinpi_gazebo scene_manager.py -l FruitMap.txt

<-- Run operate.py -->
### It'll take a while on initial load to download models from github and load custom weights ###

cd "ECE4078_Lab_2022_Team106/Week06-07/"
python3 operate.py --ckpt yolov5-sim.pt

### Hit Enter to enter SLAM, p to detect and display bounding boxes for detected fruits, n to save outputs. ###
### All fruits with bounding boxes will be used for calculation (compared to default max of 1 fruit per type) and a corresponding
### text file for each prediction image will be created for use in TargetPoseEst.py in "lab_output" folder

<-- Run TargetPoseEst.py -->

### Outputs to lab_output/targets.txt ###
cd "ECE4078_Lab_2022_Team106/Week06-07/"
python3 TargetPoseEst.py

<-- Run evaluation (REPLACE FruitMap.txt with MarkingMap.txt) -->
python3 CV_eval.py --truth FruitMap.txt lab_output/targets.txt

#########################################################

<-- PHYSICAL -->

<-- Run operate.py -->
### It'll take a while on initial load to download models from github and load custom weights ###

### Replace with corresponding ip address and port ###
cd "ECE4078_Lab_2022_Team106/Week06-07/"
python3 operate.py --ckpt yolov5-physical.pt --ip 192.168.50.1 --port 8080

### Hit Enter to enter SLAM, p to detect and display bounding boxes for detected fruits, n to save outputs. ###
### All fruits with bounding boxes will be used for calculation (compared to default max of 1 fruit per type) and a corresponding
### text file for each prediction image will be created for use in TargetPoseEst.py in "lab_output" folder

<-- Run TargetPoseEst.py -->

### Outputs to lab_output/targets.txt ###
cd "ECE4078_Lab_2022_Team106/Week06-07/"
python3 TargetPoseEst.py

<-- Run evaluation (REPLACE FruitMap.txt with MarkingMap.txt) -->
python3 CV_eval.py --truth FruitMap.txt lab_output/targets.txt