source ~/LiveDemo/catkin_ws/devel/setup.bash
rosrun penguinpi_gazebo scene_manager.py -l "M4_true_map_7fruits.txt"


### SLAM ###

Run slam with enter and click "s" to save map

python3 slam_operate.py

python3 SLAM_eval.py <groundtruth> <estimate>
python3 SLAM_eval.py "M4_true_map_7fruits.txt" "lab_output/slam.txt"



### CV ###
python3 cv_operate.py --ckpt yolo-sim.pt 

cd "M3_106/ECE4078_Lab_2022_Team106/Week10-11/"
python3 TargetPoseEstSim.py

python3 CV_eval.py --truth M4_true_map_7fruits.txt --est lab_output/targets.txt