#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频检测追踪打码工具

这是一个基于YOLO目标检测和OpenCV图像处理的GUI应用程序，
用于对视频中的特定对象进行检测和模糊处理（打码）。

功能特点：
- 支持YOLO模型加载
- 视频文件选择与预览
- 随机抽帧检测目标类别
- 选择特定类别进行模糊处理
- 可调节模糊强度
- 多线程处理避免界面卡顿
- 保持原视频音频

依赖项：
- Python 3.x
- OpenCV (cv2)
- NumPy
- Ultralytics YOLO
- PyQt5
- FFmpeg

使用方法：
1. 运行程序
2. 选择YOLO模型文件(.pt)
3. 选择要处理的视频文件
4. 设置随机抽帧数量进行类别检测
5. 选择需要打码的目标类别
6. 调整模糊强度
7. 开始处理生成打码后的视频

作者：未知
创建时间：未知
最后更新：未知
"""

import sys
import os
import cv2
# 解决类型检查问题，但保持实际功能
# type: ignore
cv2_VideoWriter_fourcc = getattr(cv2, 'VideoWriter_fourcc', lambda *args: 0x7634706d)
import numpy as np
# 修复YOLO导入问题
# type: ignore
from ultralytics import YOLO  # type: ignore

import subprocess
from datetime import datetime
import random
from PyQt5 import QtCore, QtWidgets
# 修复Qt Horizontal属性问题
from PyQt5.QtCore import Qt

# -------------------------
# QRunnable 封装器
# -------------------------
class WorkerRunnable(QtCore.QRunnable):
    """
    QRunnable封装器类
    
    用于将普通函数包装成QRunnable，以便在线程池中执行，
    避免长时间运行的任务阻塞UI主线程。
    """
    def __init__(self, fn):
        """
        初始化WorkerRunnable
        
        Args:
            fn (callable): 要在线程中执行的函数
        """
        super().__init__()
        self.fn = fn

    def run(self):
        """执行包装的函数"""
        self.fn()


# -------------------------
# 主界面
# -------------------------
class VideoBlurApp(QtWidgets.QWidget):
    """
    视频模糊处理应用主界面类
    
    提供图形用户界面，允许用户选择模型和视频文件，
    设置处理参数，并执行视频模糊处理任务。
    """
    def __init__(self):
        """初始化应用界面和组件"""
        super().__init__()

        self.setWindowTitle("视频检测追踪打码工具")
        self.resize(550, 700)

        # -------------------
        # UI 组件
        # -------------------
        self.btn_model = QtWidgets.QPushButton("选择模型文件")
        self.btn_video = QtWidgets.QPushButton("选择视频文件")

        self.label_frame_count = QtWidgets.QLabel("视频总帧数: 未知")
        self.spin_sample = QtWidgets.QSpinBox()
        self.spin_sample.setRange(1, 1000)
        self.spin_sample.setValue(100)
        self.spin_sample.setSuffix(" 帧随机检测")
        self.spin_sample.setEnabled(False)  # 先禁用

        self.btn_detect_classes = QtWidgets.QPushButton("开始检测类别")
        self.btn_detect_classes.setEnabled(False)

        self.combo_class = QtWidgets.QComboBox()
        self.combo_class.setEnabled(False)

        self.radio_blur = QtWidgets.QRadioButton("目标模糊")
        self.radio_blur.setChecked(True)

        # 修复Qt Horizontal属性问题
        self.slider_blur = QtWidgets.QSlider(QtCore.Qt.Horizontal)  # type: ignore
        self.slider_blur.setMinimum(1)
        self.slider_blur.setMaximum(20)  # 最大值模糊最强
        self.slider_blur.setValue(5)

        self.btn_start = QtWidgets.QPushButton("开始处理")
        self.btn_start.setEnabled(False)

        self.status_label = QtWidgets.QTextEdit()
        self.status_label.setReadOnly(True)

        # -------------------
        # 变量
        # -------------------
        self.model_path: str = ""
        self.video_path: str = ""
        self.model = None
        self.all_classes = []
        self.total_frames = 0
        self.threadpool = QtCore.QThreadPool()

        # -------------------
        # 布局
        # -------------------
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.btn_model)
        layout.addWidget(self.btn_video)
        layout.addWidget(self.label_frame_count)
        layout.addWidget(QtWidgets.QLabel("随机抽帧数量"))
        layout.addWidget(self.spin_sample)
        layout.addWidget(self.btn_detect_classes)
        layout.addWidget(QtWidgets.QLabel("选择处理类别"))
        layout.addWidget(self.combo_class)
        layout.addWidget(self.radio_blur)
        layout.addWidget(QtWidgets.QLabel("模糊强度（越大越强）"))
        layout.addWidget(self.slider_blur)
        layout.addWidget(self.btn_start)
        layout.addWidget(QtWidgets.QLabel("状态"))
        layout.addWidget(self.status_label)

        # -------------------
        # 事件绑定
        # -------------------
        self.btn_model.clicked.connect(self.load_model)
        self.btn_video.clicked.connect(self.load_video)
        self.btn_detect_classes.clicked.connect(self.detect_random_frames)
        self.btn_start.clicked.connect(self.process_video_thread)

    # -------------------------
    # 日志
    # -------------------------
    def log(self, text):
        """
        在状态文本框中添加日志信息
        
        Args:
            text (str): 要添加的日志文本
        """
        self.status_label.append(text)
        self.status_label.ensureCursorVisible()

    # -------------------------
    # 加载模型
    # -------------------------
    def load_model(self):
        """打开文件对话框选择并加载YOLO模型文件"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型", "", "YOLO 模型 (*.pt)"
        )
        if path:
            self.model_path = path
            self.log(f"模型选择: {path}")

            def do_load():
                try:
                    self.log("加载模型中...")
                    # 修复可能的None类型问题
                    if self.model_path:
                        self.model = YOLO(self.model_path)
                    self.log("模型加载完成！")
                except Exception as e:
                    self.log(f"模型加载失败: {e}")

            self.threadpool.start(WorkerRunnable(do_load))

    # -------------------------
    # 加载视频
    # -------------------------
    def load_video(self):
        """打开文件对话框选择视频文件并获取基本信息"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mkv)"
        )
        if path:
            self.video_path = path
            self.log(f"视频选择: {path}")

            # 获取总帧数
            cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if self.total_frames > 0:
                self.label_frame_count.setText(f"视频总帧数: {self.total_frames} 帧")
                self.spin_sample.setMaximum(self.total_frames)
                self.spin_sample.setEnabled(True)
                self.btn_detect_classes.setEnabled(True)
                self.log(f"视频总帧数: {self.total_frames}")
            else:
                self.label_frame_count.setText("视频总帧数: 未知")
                self.spin_sample.setEnabled(False)
                self.btn_detect_classes.setEnabled(False)
                self.log("无法读取视频总帧数")

    # -------------------------
    # 随机抽帧检测类别
    # -------------------------
    def detect_random_frames(self):
        """
        随机抽取视频帧进行目标检测，识别可用的类别
        
        通过随机采样视频帧，使用YOLO模型检测其中的对象，
        收集所有检测到的类别并填充到下拉框中。
        """
        if not self.model:
            self.model = YOLO("yolov8n.pt")
            self.log("未选择模型，自动加载 yolov8n.pt")

        sample_num = min(self.spin_sample.value(), self.total_frames)
        indices = random.sample(range(self.total_frames), sample_num)
        self.log(f"随机抽取 {len(indices)} 帧进行检测...")

        cap = cv2.VideoCapture(self.video_path)

        def do_detect():
            detected_classes = set()
            try:
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    # 确保模型不为None
                    if self.model is not None:
                        results = self.model(frame)[0]
                        boxes_obj = results.boxes
                        if hasattr(boxes_obj, "cls"):
                            cls_arr = boxes_obj.cls.cpu().numpy()
                            detected_classes.update(int(c) for c in cls_arr)
                cap.release()

                self.all_classes = sorted(list(detected_classes))
                self.combo_class.clear()
                # 确保模型不为None再访问names属性
                if self.model is not None:
                    for c in self.all_classes:
                        name = self.model.names.get(c, f"class {c}")
                        self.combo_class.addItem(name, c)

                self.combo_class.setEnabled(True)
                self.btn_start.setEnabled(True)
                # 确保模型不为None再访问names属性
                if self.model is not None:
                    self.log(f"检测到类别: {[self.model.names.get(c) for c in self.all_classes]}")
            except Exception as e:
                self.log(f"类别检测失败: {e}")

        self.threadpool.start(WorkerRunnable(do_detect))

    # -------------------------
    # 启动处理线程
    # -------------------------
    def process_video_thread(self):
        """启动视频处理线程"""
        self.threadpool.start(WorkerRunnable(self.process_video))

    # -------------------------
    # 视频处理逻辑
    # -------------------------
    def process_video(self):
        """
        执行视频处理的主要逻辑
        
        对视频进行逐帧处理，检测指定类别的对象并进行模糊处理，
        最后将处理后的视频与原始音频合并生成最终输出文件。
        """
        # 确保必要的变量不为空
        if not self.video_path or self.model is None:
            self.log("错误：未选择视频或模型未加载")
            return
            
        target_cls = self.combo_class.currentData()
        blur_strength = self.slider_blur.value()
        self.log(f"开始处理，目标类别={target_cls}，模糊强度={blur_strength}")

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 确保video_path不为空
        if not self.video_path:
            self.log("错误：视频路径为空")
            return
            
        final_out = self.video_path.replace(".mp4", f"_final_{time_tag}.mp4")

        # 临时无声视频
        temp_out = self.video_path.replace(".mp4", f"_temp_{time_tag}.mp4")
        # 修复VideoWriter_fourcc问题
        fourcc = cv2_VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_out, fourcc, fps, (w, h))

        frame_idx = 0
        cached_mask = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % 3 == 0:
                # 确保模型不为None
                if self.model is not None:
                    results = self.model(frame)[0]
                    boxes_obj = results.boxes
                    if hasattr(boxes_obj, "cls"):
                        cls_arr = boxes_obj.cls.cpu().numpy()
                        xyxy_arr = boxes_obj.xyxy.cpu().numpy()
                    else:
                        cls_arr = []
                        xyxy_arr = []

                    cached_mask = []
                    for cid, box_xy in zip(cls_arr, xyxy_arr):
                        if int(cid) == target_cls:
                            x1, y1, x2, y2 = map(int, box_xy)
                            cached_mask.append((x1, y1, x2, y2))

            if cached_mask:
                for (x1, y1, x2, y2) in cached_mask:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        k = blur_strength * 4 + 1  # 高强度模糊
                        blur = cv2.GaussianBlur(roi, (k, k), 0)
                        frame[y1:y2, x1:x2] = blur

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        # 合并音频生成最终视频
        try:
            self.log("合并原始音频...")
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_out,
                "-i", self.video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                final_out
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.remove(temp_out)
            self.log(f"最终输出: {final_out}")
        except Exception as e:
            self.log(f"音频合并失败: {e}")


# -------------------------
# 启动应用
# -------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = VideoBlurApp()
    win.show()
    sys.exit(app.exec_())