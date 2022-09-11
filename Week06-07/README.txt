#Launch environment
source ~/catkin_ws/devel/setup.bash
roslaunch penguinpi_gazebo ECE4078_brick.launch

#Load map
source ~/catkin_ws/devel/setup.bash
rosrun penguinpi_gazebo scene_manager.py -l FruitMap.txt

#Run operate.py
cd "ECE4078_Lab_2022_Team106/Week06-07/"
python3 operate.py --ckpt yolov5-physical.pt

#Run TargetPoseEst.py
cd "ECE4078_Lab_2022_Team106/Week06-07/"
python3 TargetPoseEst.py

#Run evaluation
python3 CV_eval.py --truth FruitMap.txt lab_output/targets.txt