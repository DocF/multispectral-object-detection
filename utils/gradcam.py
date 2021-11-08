#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Richard Fang
import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse

import warnings
warnings.filterwarnings("ignore")


def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input


def show_cam_on_image(img, mask, epoch, layer):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # img = cv2.flip(img, 1)
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    print(type(cam))
    cv2.imwrite("./cam%d_%d.jpg" %(epoch, layer), np.uint8(255 * cam))


def calcGradCam(imgpath, feature, epoch, layer):
    """

    :param feature: 特征图
    :param grad_val: 对应的特征图的梯度
    :return
    """
    # ------------ Image Preprocess ---------------
    image_path = imgpath
    img = cv2.imread(image_path, 1)
    # print(np.shape(img))
    img = np.float32(cv2.resize(img, (640, 640))) / 255
    input = preprocess_image(img)

    # ------------ GradCam -------------------------
    feature = feature.cpu().data.numpy()
    print("feature shape", feature.shape)
    cam = np.zeros(feature.shape[1:], dtype=np.float32)

    for i in range(feature.shape[0]):
        cam += feature[i, :, :]

    cam = np.maximum(cam, 0) # 比较cam的元素与0的大小，<0的都置0
    # min = np.min(cam)
    # cam -= min
    cam = cv2.resize(cam, (640, 640))

    # 归一化 cam
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    print(cam.shape)
    # ---------- Show -------------
    show_cam_on_image(img, cam, epoch, layer)

    return cam

