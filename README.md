# Real-Time Object Detection and Avoidance in Robotics: Comparing Tiny-YOLOv3 and YOLOv3

Authors: Guillem Ribes Espurz (5229154), Ricardo Ramautar 

## Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Motivation](#motivation)
  - [Training](#training)
  - [Robot Implementation](#robot-implementation)
- [Results](#results)
- [Discussion and Limitations](#discussion-and-limitations)
  - [Discussion](#discussion)
  - [Limitations](#limitations)
- [References](#references)
    
## Abstract

## Introduction

## Motivation 
In numerous applications, robots must operate in unregulated and dynamic environments filled with moving obstacles. To navigate in these challenging environments, robots require obstacle avoidance systems to ensure they do not collide. For instance, self-driving vehicles need to account for other vehicles, pedestrians, and cyclists, whose actions can be unpredictable. Consequently, the obstacle avoidance systems in these self-driving vehicles must be able to account for sudden, sporatic behavior. 

Nowadays, in robotics in order to perform obstacle avoidance other sensors such as LiDAR or ultrasound are being used instead of cameras. This introduces an issue of not knowing what obstacle you are avoiding (no classification performed). In some cases you don’t want to avoid all obstacles, if for example it’s a cleaning robot you don’t want it to avoid litter you instead want it to pick it up. However, if the robot can not differentiate between litter or an actual obstacle then it doesn’t perform accordingly. Which is why cameras should be introduced for obstacle classification in order to perform obstacle avoidance. Nonetheless, the use of cameras introduces larger models to perform obstacle avoidance, which in turn require expensive GPUs. However, not all robots are equipped with such GPUs due to their cost and space constraints. A more efficient solution is to develop smaller, more efficient models that demand minimal computational power. This would enable real-time obstacle detection on any robot, regardless of its hardware.

## Implementation 

### Training 

### Robot Implementation 
After obtaining the weights in the correct format (.weights) for the tiny-YOLOv3 model, the next step was to implement them on the robot. Initially, the darknetROS repository was utilized, as it supports the integration of tiny-YOLOv3 with ROS and requires only the implementation of the correct weights. The model was first tested on a laptop to evaluate its performance. As shown in the figure below, the model consistently detected manure and people. However, the frame rate achieved on the laptop was approximately 1.5 fps, resulting in noticeable lag during visualization.

To address this issue, the model was then run directly on the robot. Unfortunately, this led to a significant drop in performance, with the frame rate plummeting to 0.2 fps, which was far from acceptable.

To improve performance, the darknet configuration was converted to ncnn, a high-performance neural network inference framework optimized for mobile platforms. This conversion enabled the model to run on the robot at 5 fps, meeting the desired frame rate and providing smooth visualization.

[ObjectDetection](images/ObjectDetection.jpeg)

**Add a table of the fps of YOLOv3 and tiny-YOLOv3**
## Results

## Discussion and Limitations

### Discussion

### Limitations

## References 
