import os
import time

import cmd_printer
import numpy as np
import torch
from args import args
from res18_skip import Resnet18Skip
from torchvision import transforms
import cv2

class Detector:
    def __init__(self, ckpt, use_gpu=False):
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=ckpt, force_reload = True)#load model and weights,
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=ckpt)#load model and weights,

        #use CUDA for inference
        if torch.cuda.device_count() > 0 and use_gpu:
            self.use_gpu = True
            self.model = self.model.cuda()
        else:
            self.use_gpu = False

        self.model.eval() #set model to evaluation mode

    def detect_single_image(self, img):
        tick = time.time()
        with torch.no_grad():
            pred = self.model(img)
        dt = time.time() - tick
        # print(f'Inference Time {dt:.2f}s, approx {1/dt:.2f}fps', end="\r")

        #Calculating image info
        pred_count = len(pred.pandas().xyxy[0]) #calculating how many predictions on page

        result_list = []
        for i in range(pred_count): #adding class, name and bounding box info entry for each prediction
            xmin = pred.pandas().xyxy[0]['xmin'][i]
            ymin = pred.pandas().xyxy[0]['ymin'][i]
            xmax = pred.pandas().xyxy[0]['xmax'][i]
            ymax = pred.pandas().xyxy[0]['ymax'][i]
            name = pred.pandas().xyxy[0]['name'][i]
            class_no = int(pred.pandas().xyxy[0]['class'][i])
            result_list.append({'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax, 'name':name, 'class': class_no})

        return pred.render()[0], pred.render()[0], result_list, pred_count
