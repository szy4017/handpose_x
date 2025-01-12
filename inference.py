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
    cake_gif_dict = {'1-1': './right_window/cake1-1.gif', '1-2': './right_window/cake1-2.gif',
                     '1-3': './right_window/cake1-3.gif', '1-4': './right_window/cake1-4.gif',
                     '1-5': './right_window/cake1-5.gif', '1-6': './right_window/cake1-6.gif',
                     '1-7': './right_window/cake1-7.gif',
                     '1-7-1': './right_window/cake1-7-1.gif', '1-7-2': './right_window/cake1-7-2.gif'}
    cake_gif = None
    # cake_gif = Image.open('./right_window/cake1-1.gif')
    pose_cake_dict_3 = {'1': '1-1', '1-1': '1-2'}
    pose_cake_dict_4 = {'1-2': '1-3', '1-3': '1-4', '1-4': '1-5', '1-5': '1-6', '1-6': '1-7'}
    pose_cake_dict_5 = {'1-7': '1-7-1', '1-7-1': '1-7-1'}
    cake_key = None
    cake_change = None
    gif_frame = 0

    background_left = cv2.imread('./left_window/background1.jpg')
    background_left_2 = cv2.imread('./left_window/background2.jpg')
    background_left_3 = cv2.imread('./left_window/background3.jpg')
    while True:
        # 等待键盘输入
        key = input("请选择蛋糕1或2：")

        if key == '1':
            background_right = cv2.imread('./right_window/background1.jpg')
            cake_key = '1'
            break
        elif key == '2':
            background_right = cv2.imread('./right_window/background2.jpg')
            cake_key = '2'
            break
        else:
            continue

    # 用摄像头获取图像进行预测
    capture = cv2.VideoCapture(0)
    pose_flag = 0   # pose flag初始化
    while True:
        # 获取当前时间
        ret, cam_img = capture.read()
        if ret:
            key = cv2.waitKey(10)

            if key == ord(' '):
                # 输入图像进行模型预测
                cam_img = cv2.flip(cam_img, 1)
                cam_img_ = cv2.resize(cam_img, (ops.img_size[1], ops.img_size[0]), interpolation=cv2.INTER_CUBIC)
                cam_img_ = cam_img_.astype(np.float32)
                cam_img_ = (cam_img_ - 128.) / 256.

                cam_img_ = cam_img_.transpose(2, 0, 1)
                cam_img_ = torch.from_numpy(cam_img_)
                cam_img_ = cam_img_.unsqueeze_(0)

                # start = time.time()
                pre_ = model_(cam_img_.float())  # 模型推理
                # end = time.time()
                # print('inference time: ', end - start)
                output = pre_.cpu().detach().numpy()
                output = np.squeeze(output)

                # 可视化手势识别预测结果
                pred_img = cam_img.copy()
                img_width = pred_img.shape[1]
                img_height = pred_img.shape[0]
                pts_hand = {}  # 构建关键点连线可视化结构
                for i in range(int(output.shape[0] / 2)):
                    x = (output[i * 2 + 0] * float(img_width))
                    y = (output[i * 2 + 1] * float(img_height))

                    pts_hand[str(i)] = {}
                    pts_hand[str(i)] = {
                        "x": x,
                        "y": y,
                    }
                draw_bd_handpose(pred_img, pts_hand, 0, 0)  # 绘制关键点连线

                # ------------- 绘制关键点
                for i in range(int(output.shape[0] / 2)):
                    x = (output[i * 2 + 0] * float(img_width))
                    y = (output[i * 2 + 1] * float(img_height))

                    cv2.circle(pred_img, (int(x), int(y)), 3, (255, 50, 60), -1)
                    cv2.circle(pred_img, (int(x), int(y)), 1, (255, 150, 180), -1)

                # 左侧窗口显示
                if cake_key in ['1-2', '1-3', '1-4', '1-5', '1-6', '1-7']:
                    left_window = background_left_2.copy()
                else:
                    left_window = background_left.copy()
                height_l, width_l, _ = left_window.shape
                pred_img_resived = cv2.resize(pred_img, (460, 390))
                left_window[79:469, 170:630, :] = pred_img_resived


                # 根据手势特征预测手势类别
                pose_output = pose_model(torch.tensor(output[None:], dtype=torch.float32))
                pose_flag = torch.max(pose_output.data)
                pose_flag = int(pose_flag)
                print('Predicted pose flag:', pose_flag)


                # 右侧窗口显示
                if cake_gif is None:
                    right_window = background_right.copy()
                else:
                    if gif_frame < cake_gif.n_frames:
                        cake_gif.seek(gif_frame)
                        gif_frame_image = cake_gif.convert('RGB')
                        gif_frame += 1
                    else:
                        gif_frame = cake_gif.n_frames - 1
                        cake_gif.seek(gif_frame)
                        gif_frame_image = cake_gif.convert('RGB')

                    cake_show_img = np.array(gif_frame_image)
                    cake_show_img = cv2.cvtColor(cake_show_img, cv2.COLOR_RGB2BGR)
                    right_window = cake_show_img.copy()


            elif key == -1:
                # 识别pose_flag并设置左右窗口
                if pose_flag == 3 and cake_key in pose_cake_dict_3.keys():
                    cake_change = pose_cake_dict_3[cake_key]
                    cake_gif = Image.open(cake_gif_dict[cake_change])
                    gif_frame = 0   # 更换cake gif后，对frame重置零
                elif pose_flag == 4 and cake_key in pose_cake_dict_4.keys():
                    cake_change = pose_cake_dict_4[cake_key]
                    cake_gif = Image.open(cake_gif_dict[cake_change])
                    gif_frame = 0   # 更换cake gif后，对frame重置零


                # 左侧窗口显示
                if cake_key in ['1-2', '1-3', '1-4', '1-5', '1-6', '1-7']:
                    left_window = background_left_2.copy()
                else:
                    left_window = background_left.copy()
                height_l, width_l, _ = left_window.shape
                cam_img_resived = cv2.resize(cam_img, (460, 390))
                cam_img_resived = cv2.flip(cam_img_resived, 1)
                left_window[79:469, 170:630, :] = cam_img_resived

                # 右侧窗口显示
                if cake_change is not None:
                    cake_key = cake_change
                if cake_gif is None:
                    right_window = background_right.copy()
                else:
                    if gif_frame < cake_gif.n_frames:
                        cake_gif.seek(gif_frame)
                        gif_frame_image = cake_gif.convert('RGB')
                        gif_frame += 1
                    else:
                        gif_frame = cake_gif.n_frames - 1
                        cake_gif.seek(gif_frame)
                        gif_frame_image = cake_gif.convert('RGB')

                    cake_show_img = np.array(gif_frame_image)
                    cake_show_img = cv2.cvtColor(cake_show_img, cv2.COLOR_RGB2BGR)
                    right_window = cake_show_img.copy()
                pose_flag = 0 # pose flag重置

            # 左右窗口拼接
            window = cv2.hconcat([left_window, right_window])
            cv2.imshow('cake show', window)



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
