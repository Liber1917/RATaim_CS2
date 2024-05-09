
import math
import threading
import time
import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

from ScreenShot import screenshot
from SendInput import *

import pynput.mouse
from pynput.mouse import Listener
from pynput import keyboard

import win32gui
import win32con

from smooth_line import SLine

IsX2Pressed = False
should_run = True
toggle_target = 0
toggle_start = 0

def on_press(key):
    global toggle_start
    try:
        # 检查按键是否是"f"键（包括大写和小写）
        if key.char.lower() == 'f' or key.char.upper() == 'F':
            toggle_start = 1
            print('The "f" key was pressed')
    except AttributeError:
        # 如果按键不是字符类型，则可能是一个特殊按键
        pass

def on_release(key):
    global toggle_target, toggle_start
    try:
        if toggle_start:
            toggle_target = 1 if toggle_target == 0 else 0
            print("target changed to ", toggle_target)
            toggle_start = 0
    except AttributeError:
        # 如果按键不是字符类型，则可能是一个特殊按键
        pass

def shift_target():
    global toggle_start, toggle_target

# 设置监听器
def keyboard_listener():
    global should_run
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def mouse_click(x,y,button,pressed):
    global IsX2Pressed
    print(x,y,button,pressed)
    if pressed and button == pynput.mouse.Button.x2:
        IsX2Pressed=True
        print(IsX2Pressed)
    else:
        IsX2Pressed=False

def mouse_listener():
    global should_run
    with Listener(on_click=mouse_click) as listener:
        while should_run:
            listener.join()
@smart_inference_mode()


def get_highest_confidence_target(pred):
    """
    从预测结果中获取置信度最高的目标信息。

    参数：
        pred (list): 预测结果，包含检测到的所有目标信息。

    返回：
        list: 置信度最高的目标信息，格式为 [x_min, y_min, x_max, y_max, confidence, class, distance]。
              如果没有检测到目标，返回 None。
    """
    max_confidence = 0
    best_detection = None

    for det in pred:
        for *xyxy, conf, cls in det:
            if conf > max_confidence:
                max_confidence = conf
                best_detection = [*xyxy, conf, cls]

    if best_detection is not None:
        xywh = (xyxy2xywh(torch.tensor(best_detection[:4]).view(1, 4))).view(-1).tolist()
        distance = math.sqrt((xywh[0]-320)**2 + (xywh[1]-320)**2)
        xywh.append(distance)
        return xywh
    else:
        return None

global image_size
image_size = 640
global control_time_cycle
control_time_cycle = 0.05
def run():
    # Load model
    device = torch.device("cuda:0")
    model = DetectMultiBackend(weights="./weight/best_CS2m.pt", device=device, dnn=False, data=False, fp16=True)
    global IsX2Pressed, toggle_target
    while True:
        #read images
        im = screenshot()
        im0=im
        #process images
        im = letterbox(im,(image_size,image_size),stride=32,auto=True)[0]#paddle resize
        im = im.transpose((2,0,1))[::-1]#HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im) #contiguous

        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        #推理
        start = time.time()
        pred = model(im, augment=False, visualize=False)#debug

        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45, classes=toggle_target, max_det=1000)#pred 0:person,(0,2)
        end = time.time()
        # print(f'推理所需时间{end-start}s')

        # Process predictions
        for i, det in enumerate(pred):  # per image
            #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0,line_width=1)
            if len(det):
                distance_list=[]
                target_list=[]
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):#target info process

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                    #line = cls, *xywh, conf  # label format
                    #print(xywh)
                    X=xywh[0]-image_size/2
                    Y=xywh[1]-image_size/2
                    distance = math.sqrt(X**2+Y**2)
                    xywh.append(distance)
                    annotator.box_label(xyxy, label=f'[{int(cls)}Distance:{round(distance,2)}]',
                                        color=(34, 139, 34),
                                        txt_color=(0, 191, 255))

                    distance_list.append(distance)
                    target_list.append(xywh)

                target_info= target_list[distance_list.index(min(distance_list))]
                # target_info = get_highest_confidence_target([det])
                if IsX2Pressed:
                    print('kaile')
                    mouse_xy(int(target_info[0]-image_size/2),int(target_info[1]-image_size/2))
                    # remap_x = target_info[0]-image_size/2
                    # remap_y = target_info[1]-image_size/2
                    # s_curve_x = [SLine(FStop=remap_x, index=i) for i in range(31)]
                    # s_curve_y = [SLine(FStop=remap_y, index=i) for i in range(31)]
                    # for i in range(11):
                    #     mouse_xy(int(s_curve_x[i]/2),int(s_curve_y[i]/2))
                    #     print(s_curve_x[i],s_curve_y[i])
                    #     time.sleep(control_time_cycle*0.001)
                    time.sleep(control_time_cycle*0.00001)
            # im0 = annotator.result()
            # cv2.imshow('window', im0)
            # hwnd = win32gui.FindWindow(None, 'window')
            # win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
            #                       win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            # cv2.waitKey(1)









if __name__ == "__main__":
    print("start")
    try:
        threading.Thread(target=mouse_listener).start()
        threading.Thread(target=keyboard_listener).start()
        run()
    except KeyboardInterrupt:
        should_run = False
        print("end")

