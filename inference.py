#-*-coding:utf-8-*-
# date:2021-04-5
# Author: Eric.Lee
# function: Inference

import os
import argparse
import torch
import torch.nn as nn
import numpy as np

import time
import datetime
import os
import math
from datetime import datetime
import cv2
import torch.nn.functional as F

from models.resnet import resnet18,resnet34,resnet50,resnet101
from models.squeezenet import squeezenet1_1,squeezenet1_0
from models.shufflenetv2 import ShuffleNetV2
from models.shufflenet import ShuffleNet
from models.mobilenetv2 import MobileNetV2
from torchvision.models import shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0
from models.rexnetv1 import ReXNetV1
from PIL import Image, ImageSequence

from utils.common_utils import *
import copy
from hand_data_iter.datasets import draw_bd_handpose
from pose_recog import MLP as PoseClassifier

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Hand Pose Inference')
    parser.add_argument('--model_path', type=str, default = './weights/ReXNetV1-size-256-wingloss102-0.122.pth',
        help = 'model_path') # 模型路径
    parser.add_argument('--model', type=str, default = 'ReXNetV1',
        help = '''model : resnet_34,resnet_50,resnet_101,squeezenet1_0,squeezenet1_1,shufflenetv2,shufflenet,mobilenetv2
            shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0,ReXNetV1''') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 42,
        help = 'num_classes') #  手部21关键点， (x,y)*2 = 42
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
    parser.add_argument('--test_path', type=str, default = './image/',
        help = 'test_path') # 测试图片路径
    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--vis', type=bool , default = True,
        help = 'vis') # 是否可视化图片

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    ops = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    print('----------------------------------')

    unparsed = vars(ops) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    #---------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

    test_path =  ops.test_path # 测试图片文件夹路径

    #---------------------------------------------------------------- 构建模型
    print('use model : %s'%(ops.model))

    if ops.model == 'resnet_50':
        model_ = resnet50(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == 'resnet_18':
        model_ = resnet18(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == 'resnet_34':
        model_ = resnet34(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == 'resnet_101':
        model_ = resnet101(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == "squeezenet1_0":
        model_ = squeezenet1_0(num_classes=ops.num_classes)
    elif ops.model == "squeezenet1_1":
        model_ = squeezenet1_1(num_classes=ops.num_classes)
    elif ops.model == "shufflenetv2":
        model_ = ShuffleNetV2(ratio=1., num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x1_5":
        model_ = shufflenet_v2_x1_5(pretrained=False,num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x1_0":
        model_ = shufflenet_v2_x1_0(pretrained=False,num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x2_0":
        model_ = shufflenet_v2_x2_0(pretrained=False,num_classes=ops.num_classes)
    elif ops.model == "shufflenet":
        model_ = ShuffleNet(num_blocks = [2,4,2], num_classes=ops.num_classes, groups=3)
    elif ops.model == "mobilenetv2":
        model_ = MobileNetV2(num_classes=ops.num_classes)
    elif ops.model == "ReXNetV1":
        model_ = ReXNetV1(num_classes=ops.num_classes)

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval() # 设置为前向推断模式
    pose_model = PoseClassifier(input_size=42, hidden_size=64, output_size=6)
    pose_model.load_state_dict(torch.load('model_params.pth'))

    # print(model_)# 打印模型结构

    # 加载测试模型
    if os.access(ops.model_path,os.F_OK):# checkpoint
        chkpt = torch.load(ops.model_path, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.model_path))

    #---------------------------------------------------------------- 预测图片
    '''建议 检测手bbox后，crop手图片的预处理方式：
    # img 为原图
    x_min,y_min,x_max,y_max,score = bbox
    w_ = max(abs(x_max-x_min),abs(y_max-y_min))

    w_ = w_*1.1

    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2

    x1,y1,x2,y2 = int(x_mid-w_/2),int(y_mid-w_/2),int(x_mid+w_/2),int(y_mid+w_/2)

    x1 = np.clip(x1,0,img.shape[1]-1)
    x2 = np.clip(x2,0,img.shape[1]-1)

    y1 = np.clip(y1,0,img.shape[0]-1)
    y2 = np.clip(y2,0,img.shape[0]-1)
    '''

    cv2.namedWindow('image', 0)
    cv2.resizeWindow('image', 1600, 600)

    # 初始化gif变量
    gif_frame = 0
    gif_start_time = 0
    gif_play_time = 3

    select_area_img = cv2.imread('./image/left.jpg')
    select_area_img = cv2.resize(select_area_img, (129, 406))
    select_bottom_area_img = cv2.imread('./image/bottom.jpg')
    cake_show_img = cv2.imread('./cake_show/plate.jpg')
    background_img = np.zeros((600, 1600, 3)).astype('uint8')
    # 用摄像头获取图像进行预测
    capture = cv2.VideoCapture(0)
    flag = 0
    cake_state = '0'
    cake_gif = Image.open('./cake_show/cake-0-1.gif')
    while True:
        # 获取当前时间
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        ret, img = capture.read()
        if ret:
            img_width = img.shape[1]
            img_height = img.shape[0]
            cake_shwow_img_width = cake_show_img.shape[1]
            cake_shwow_img_height = cake_show_img.shape[0]
            img = cv2.flip(img, 1)
            # 输入图片预处理
            img_ = cv2.resize(img, (ops.img_size[1],ops.img_size[0]), interpolation = cv2.INTER_CUBIC)
            img_ = img_.astype(np.float32)
            img_ = (img_-128.)/256.

            img_ = img_.transpose(2, 0, 1)
            img_ = torch.from_numpy(img_)
            img_ = img_.unsqueeze_(0)

            if use_cuda:
                img_ = img_.cuda()  # (bs, 3, h, w)

            start = time.time()
            pre_ = model_(img_.float()) # 模型推理
            end = time.time()
            print('inference time: ', end-start)

            output = pre_.cpu().detach().numpy()
            output = np.squeeze(output)

            # 判断当前的手势含义，输入output，输出hand_pose
            # np.save('./pose_recog_test_data/pose_1_{}.npy'.format(flag), output)
            # flag = flag + 1
            # if flag > 10:
            #     break

            # 通过姿势分类模型来分类手势类别
            pose_output = pose_model(torch.tensor(output[None:], dtype=torch.float32))
            pose_flag = torch.max(pose_output.data)
            pose_flag = int(pose_flag)
            print('Predicted pose flag:', pose_flag)

            # 根据手势变化蛋糕动画
            if pose_flag == 1:
                cake_state = '0-1-2'

            if cake_state == '0-1-2' and pose_flag == 3:
                # 更换gif
                cake_gif = Image.open('./cake_show/cake-0-1-2.gif')
                gif_frame = 0
                gif_start_time = current_time


            # 计算gif播放时间
            gif_time = current_time - gif_start_time

            # 如果gif播放时间超过3秒
            if gif_time > gif_play_time:
                # 更新gif帧
                gif_frame += 10
                gif_frame %= cake_gif.n_frames

                # 更新gif播放时间
                gif_start_time = current_time

            # 获取gif当前帧
            cake_gif.seek(gif_frame)
            gif_frame_image = cake_gif.convert('RGB')
            cake_show_img = np.array(gif_frame_image)
            cake_show_img = cv2.cvtColor(cake_show_img, cv2.COLOR_RGB2BGR)

            img_show = img.copy()
            # 绘制选择区域
            # 选择区域图像的宽度和高度
            select_area_img_height, select_area_img_width, _ = select_area_img.shape
            bottom_img_height, bottom_img_width, _ = select_bottom_area_img.shape
            # 设置选择区域图像的不透明度
            opacity = 0.8
            # 计算选择区域图像在原始图像中的位置
            x_offset = 0
            y_offset = int(img_height / 2 - select_area_img_height / 2)

            bottom_x_offset = int(img_width / 2 - bottom_img_width / 2)
            bottom_y_offset = int(img_height - bottom_img_height)

            # 将选择图像添加到原始图像的左侧
            img_show[y_offset:y_offset + select_area_img_height, x_offset:x_offset + select_area_img_width] = cv2.addWeighted(select_area_img,
                                                                                                                              opacity,
                                                                                                                              img_show[y_offset:y_offset + select_area_img_height,
                                                                                                                              x_offset:x_offset + select_area_img_width],
                                                                                                                              1 - opacity,
                                                                                                                              0)
            # 将选择图像添加到原始图像的底部
            img_show[bottom_y_offset:bottom_y_offset + bottom_img_height,
            bottom_x_offset:bottom_x_offset + bottom_img_width] = cv2.addWeighted(select_bottom_area_img,
                                                                                  opacity,
                                                                                  img_show[bottom_y_offset:bottom_y_offset + bottom_img_height,
                                                                                  bottom_x_offset:bottom_x_offset + bottom_img_width],
                                                                                  1 - opacity,
                                                                                  0)

            pts_hand = {} #构建关键点连线可视化结构
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(img_width))
                y = (output[i*2+1]*float(img_height))

                pts_hand[str(i)] = {}
                pts_hand[str(i)] = {
                    "x":x,
                    "y":y,
                    }
            draw_bd_handpose(img_show, pts_hand,0,0) # 绘制关键点连线

            #------------- 绘制关键点
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(img_width))
                y = (output[i*2+1]*float(img_height))

                cv2.circle(img_show, (int(x),int(y)), 3, (255,50,60),-1)
                cv2.circle(img_show, (int(x),int(y)), 1, (255,150,180),-1)

            background_img[60:60+img_height, :img_width, :] = img_show
            background_img[:cake_show_img.shape[0], 800:800+cake_show_img.shape[1], :] = cake_show_img
            if ops.vis:
                cv2.imshow('image',background_img)
                if cv2.waitKey(5) == 27 :
                    break

    '''
    with torch.no_grad():
        idx = 0
        for file in os.listdir(ops.test_path):
            if '.jpg' not in file:
                continue
            idx += 1
            print('{}) image : {}'.format(idx,file))
            img = cv2.imread(ops.test_path + file)
            img_width = img.shape[1]
            img_height = img.shape[0]
            # 输入图片预处理
            img_ = cv2.resize(img, (ops.img_size[1],ops.img_size[0]), interpolation = cv2.INTER_CUBIC)
            img_ = img_.astype(np.float32)
            img_ = (img_-128.)/256.

            img_ = img_.transpose(2, 0, 1)
            img_ = torch.from_numpy(img_)
            img_ = img_.unsqueeze_(0)

            if use_cuda:
                img_ = img_.cuda()  # (bs, 3, h, w)
            pre_ = model_(img_.float()) # 模型推理
            output = pre_.cpu().detach().numpy()
            output = np.squeeze(output)

            pts_hand = {} #构建关键点连线可视化结构
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(img_width))
                y = (output[i*2+1]*float(img_height))

                pts_hand[str(i)] = {}
                pts_hand[str(i)] = {
                    "x":x,
                    "y":y,
                    }
            draw_bd_handpose(img,pts_hand,0,0) # 绘制关键点连线

            #------------- 绘制关键点
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(img_width))
                y = (output[i*2+1]*float(img_height))

                cv2.circle(img, (int(x),int(y)), 3, (255,50,60),-1)
                cv2.circle(img, (int(x),int(y)), 1, (255,150,180),-1)

            if ops.vis:
                cv2.namedWindow('image',0)
                cv2.imshow('image',img)
                if cv2.waitKey(600) == 27 :
                    break    
    '''


    cv2.destroyAllWindows()

    print('well done ')
