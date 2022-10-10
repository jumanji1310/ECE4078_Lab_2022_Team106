### Notes ###
- Download yolo-phy.pt from here https://drive.google.com/file/d/1gHooid_QF7BXJ4egS8WPjlGw1MkBhu3t/view?usp=sharing
- Download yolo-sim.py from here https://drive.google.com/file/d/1-wSqR_y7in-RNGYUgSS7_4AfK6efEvUe/view?usp=sharing
- yolo-phy.pt and yolo-sim.pt should be placed in the Week06-07 Folder
- detector.py, TargetPoseEst and operate.py has been changed.
- param folder has been split into sim and physical
- catkin_ws and ECE4078_Lab_2022_Team106 should be in home


## IMPORTANT for MARKING ##
- Images must be taken from operate.py using "p" and "n" to generate necessary text files otherwise
  TargetPoseEstSim.py and TargetPoseEstPhy.py won't work
- If using groundtruth robot poses, must replace pose values for each prediction in images.txt

@@ MAKE SURE TO DELETE/MOVE files in lab_output in between SIMULATION/PHYSICAL robots to avoid outputs from both getting mixed up @@

<-- Setup for yolov5 -->

### Run the following commands to initalise environment for yolov5 ###
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install

#########################################################

<-- SIMULATION -->

# Might have to use the following to fix permission issues
chmod 777 -R ./

<-- Launch environment -->
(new terminal)
source ~/catkin_ws/devel/setup.bash
roslaunch penguinpi_gazebo ECE4078_brick.launch

<-- Load map (REPLACE FruitMap.txt with MarkingMap.txt) -->
(new terminal)
source ~/catkin_ws/devel/setup.bash
rosrun penguinpi_gazebo scene_manager.py -l FruitMap.txt

## Might have to rebuild if there's errors by deleting "build" folder and catkin build ###
	-- OR --
Launch environment however it's meant to be launched


<-- Run operate.py -->
### It'll take a while on initial load to download models from github and load custom weights ###
(new terminal)
cd "M3_106/MECE4078_Lab_2022_Team106/Week06-07/"
python3 operate.py --ckpt yolo-sim.pt 
### Use the following if CUDA error occurs to force it to not use CUDA """
export CUDA_VISIBLE_DEVICES="" 

### Hit Enter to enter SLAM, p to detect and display bounding boxes for detected fruits, n to save outputs. ###
### All fruits with bounding boxes will be used for calculation (compared to default max of 1 fruit per type) and a corresponding
### text file for each prediction image will be created for use in TargetPoseEst.py in "lab_output" folder

<-- Run TargetPoseEst.py -->

### Outputs to lab_output/targets.txt ###
(new terminal)
cd "M3_106/ECE4078_Lab_2022_Team106/Week06-07/"
python3 TargetPoseEstSim.py

<-- Run evaluation (REPLACE FruitMap.txt with MarkingMap.txt) -->
python3 CV_eval.py --truth FruitMap.txt --est lab_output/targets.txt

#########################################################

<-- PHYSICAL -->

# Might have to use the following to fix permission issues
chmod 777 -R ./

<-- Run operate.py -->
### It'll take a while on initial load to download models from github and load custom weights ###

### Replace with corresponding ip address and port ###
(new terminal)
cd "M3_106/ECE4078_Lab_2022_Team106/Week06-07/"
python3 operate.py --ckpt yolo-phy.pt --ip 192.168.50.1 --port 8080
### Use the following if CUDA error occurs to force it to not use CUDA """
export CUDA_VISIBLE_DEVICES="" 


### Hit Enter to enter SLAM, p to detect and display bounding boxes for detected fruits, n to save outputs. ###
### All fruits with bounding boxes will be used for calculation (compared to default max of 1 fruit per type) and a corresponding
### text file for each prediction image will be created for use in TargetPoseEst.py in "lab_output" folder

<-- Run TargetPoseEst.py -->

### Outputs to lab_output/targets.txt ###
(new terminal)
cd "M3_106/ECE4078_Lab_2022_Team106/Week06-07/"
python3 TargetPoseEstPhy.py

<-- Run evaluation (REPLACE FruitMap.txt with MarkingMap.txt) -->
python3 CV_eval.py --truth FruitMap.txt --est lab_output/targets.txt