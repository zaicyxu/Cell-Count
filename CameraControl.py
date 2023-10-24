# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author         :  Guo Youwen
@Version        :
------------------------------------
@File           :  CameraControl.py
@Description    :
@CreateTime     :  2023/9/4 10:11
"""

import toupcam
import cv2
import numpy as np
from time import sleep
import os
import ctypes
import threading

dir = os.path.dirname(os.path.realpath(__file__))

class CameraControl:
    def __init__(self):
        super().__init__()
        self.hcam = None
        self.imgWidth = 0
        self.imgHeight = 0
        self.res = 0
        self.saturation = toupcam.TOUPCAM_SATURATION_DEF
        self.pData = None
        self.frame = None
        self.count = 0
        self.frame_lock = threading.Event()

    # 设置白平衡的色温和色彩
    def setWBTempAndTint(self):
        if self.hcam:
            temp_value = 6503
            tint_value = 1061
            self.hcam.put_TempTint(temp_value, tint_value)

    def put_Saturation(self, n):
        self.hcam.put_Saturation(n)

    # def get_Saturation(self):
    #     x = ctypes.c_int(self.saturation)
    #     self.hcam.Toupcam_get_Saturation(self.__h, ctypes.byref(x))
    #     return x.value
    def onOpen(self):
        if self.hcam:
            self.closeCamera()
        else:
            arr = toupcam.Toupcam.EnumV2()
            if 0 == len(arr):
                print('no camera found!')
            elif 1 == len(arr):
                self.cur = arr[0]
            # self.openCamera()

    def closeCamera(self):
        if self.hcam:
            self.hcam.Close()
        self.hcam = None
        self.pData = None

    def setResolution(self, index):
        if self.hcam:
            self.hcam.Stop()
        self.res = index
        self.imgWidth = self.cur.model.res[index].width
        self.imgHeight = self.cur.model.res[index].height
        if self.hcam:
            self.hcam.put_eSize(self.res)
            self.startCamera()

    def startCamera(self):
        self.pData = bytes(toupcam.TDIBWIDTHBYTES(self.imgWidth * 24) * self.imgHeight)
        try:
            self.hcam.StartPullModeWithCallback(self.event_callback, self)
        except toupcam.HRESULTException:
            self.closeCamera()
            print("Failed to start camera.")

    def openCamera(self):
        self.hcam = toupcam.Toupcam.Open(None)
        if self.hcam:
            self.res = self.hcam.get_eSize()
            self.imgWidth = self.cur.model.res[self.res].width
            self.imgHeight = self.cur.model.res[self.res].height
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_BYTEORDER, 0)  # RGB byte order
            self.hcam.put_AutoExpoEnable(1)
            self.setWBTempAndTint()
            self.put_Saturation(n=8)
            # self.get_Saturation()
            self.startCamera()

    def snapImage(self):
        if self.hcam:
            if self.pData is not None:
                raw_image_array = np.frombuffer(self.pData, dtype=np.uint8)
                image_shape = (3072, 2048, 3)
                raw_image_array = raw_image_array[0:3072 * 2048 * 3].reshape(image_shape)
            # image = cv2.imdecode(self.pData, cv2.IMREAD_COLOR)
            # cv2.imshow("Captured Image", raw_image_array)
            # cv2.imwrite("captured_image.jpg", raw_image_array)
            else:
                # Snap logic if supported by the camera model
                pass

    def onAutoExpo(self, state):
        if self.hcam:
            self.hcam.put_AutoExpoEnable(1 if state else 0)
            self.slider_expoTime.setEnabled(not state)
            self.slider_expoGain.setEnabled(not state)

    def onExpoTime(self, value):
        if self.hcam:
            self.hcam.put_ExpoTime(value)

    @staticmethod
    def event_callback(nEvent, self):
        # if toupcam.TOUPCAM_EVENT_IMAGE == nEvent:
        self.handle_image_event()
        # self.handleStillImageEvent()

    def handle_image_event(self):
        try:
            self.hcam.PullImageV3(self.pData, 0, 24, 0, None)
        except toupcam.HRESULTException:
            return None
        frame = np.frombuffer(self.pData, dtype=np.uint8)
        image_shape = (2048, 3072, 3)
        frame = frame[:3072 * 2048 * 3].reshape(image_shape)
        # print('real time mean:', np.mean(frame))
        self.frame = frame
        # print(frame.shape)
        # frame = cv2.cvtColor(self.pData, cv2.COLOR_RGB2BGR)  # Assuming the image is in RGB format
        # cv2.imshow("Camera Stream", frame)
        # cv2.waitKey(1)

        # return frame



    def handleStillImageEvent(self):
        info = toupcam.ToupcamFrameInfoV3()
        result_image = None
        try:
            # self.hcam.PullImageV3(None, 1, 24, 0, info)  # peek
            self.hcam.PullImageV3(None, 1, 24, 0, info)  # peek
        except toupcam.HRESULTException:
            pass
        else:
            if info.width > 0 and info.height > 0:
                buf = bytes(toupcam.TDIBWIDTHBYTES(info.width * 24) * info.height)
                try:
                    self.hcam.PullImageV3(buf, 1, 24, 0, info)
                except toupcam.HRESULTException:
                    pass
                else:
                    frame = np.frombuffer(buf, dtype=np.uint8)
                    image_shape = (2048, 3072, 3)
                    frame = frame[:3072 * 2048 * 3].reshape(image_shape)
                    # frame = cv2.cvtColor(self.pData, cv2.COLOR_RGB2BGR)  # Assuming the image is in RGB format
                    # cv2.imshow("Camera Stream", frame)
                    # cv2.waitKey(1)

def test_():
    camera = CameraControl()
    # 打开相机
    camera.onOpen()
    # camera.openCamera()
    # 设置曝光为300毫秒
    # 注意: 这里假设toupcam.TOUPCAM_OPTION_EXPOSURE是设置曝光时间的选项，真实的选项可能不同
    # if camera.hcam:
    # 	camera.onExpoTime(3000)
    # camera.hcam.put_Option(toupcam.TOUPCAM_OPTION_EXPOSURE, 300)
    camera.openCamera()
    # a = 1
    for ind in range(100):
        try:
            print('no real time:', np.mean(camera.frame))
        except:
            pass
        sleep(1)


def main():
    camera = CameraControl()
    # 打开相机
    camera.onOpen()
    # camera.openCamera()
    # 设置曝光为300毫秒
    # 注意: 这里假设toupcam.TOUPCAM_OPTION_EXPOSURE是设置曝光时间的选项，真实的选项可能不同
    # if camera.hcam:
    # 	camera.onExpoTime(3000)
    # camera.hcam.put_Option(toupcam.TOUPCAM_OPTION_EXPOSURE, 300)
    camera.openCamera()
    # 拍照并保存图像
    # camera.snapImage()
    # 关闭相机
    camera.closeCamera()


if __name__ == '__main__':
    # main()
    test_()