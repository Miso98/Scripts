import sys
import cv2
import os
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QComboBox
)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap


class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("OSBOT Webcam Recorder")
        self.setGeometry(100, 100, 900, 700)

        # --- UI Elements ---
        self.label = QLabel(self)
        self.label.setScaledContents(True)

        self.start_button = QPushButton("Start Recording", self)
        self.stop_button = QPushButton("Stop Recording", self)
        self.stop_button.setEnabled(False)

        # Resolution dropdown
        self.resolution_box = QComboBox(self)
        self.resolution_box.addItems([
            "640x480", "1280x720", "1920x1080", "3840x2160"
        ])

        # FPS dropdown
        self.fps_box = QComboBox(self)
        self.fps_box.addItems(["30", "60"])

        # Layout
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Resolution:"))
        control_layout.addWidget(self.resolution_box)
        control_layout.addWidget(QLabel("FPS:"))
        control_layout.addWidget(self.fps_box)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(control_layout)
        self.setLayout(layout)

        # Camera
        self.cap = cv2.VideoCapture(0)  # may need /dev/videoX

        # Timer for preview
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Recorder
        self.recording = False
        self.out = None

        # Connect buttons
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)

        # Ensure recordings folder exists
        os.makedirs("recordings", exist_ok=True)

        # Apply initial settings
        self.apply_camera_settings()

    def apply_camera_settings(self):
        """Apply resolution + FPS from dropdowns to camera."""
        res = self.resolution_box.currentText().split("x")
        width, height = int(res[0]), int(res[1])
        fps = int(self.fps_box.currentText())

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qt_image))

            if self.recording and self.out is not None:
                self.out.write(frame)

    def start_recording(self):
        # Apply settings before recording
        self.apply_camera_settings()

        # Get resolution + fps from actual camera
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("recordings", f"recording_{timestamp}.mp4")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        self.recording = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        print(f"Recording started: {save_path} @ {width}x{height} {fps}fps")

    def stop_recording(self):
        self.recording = False
        if self.out:
            self.out.release()
            self.out = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        print("Recording stopped.")

    def closeEvent(self, event):
        self.cap.release()
        if self.out:
            self.out.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())
