from collections import deque
from multiprocessing.dummy import freeze_support

from sympy import true
from ultralytics import YOLO
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
import cv2
import os
import sys
import torch
import ultralytics
from ultralytics.utils.plotting import Annotator
import threading

#Author : Owen Reid

#SWAP THIS TO WHATEVER THE NEWEST MODEL IS BEFORE TRAINING / USING
#model = YOLO("C:\\Users\\Socce\\PycharmProjects\\PythonProject\\runs\\detect\\train3\\weights\\best.pt")
model = YOLO("yolo26l-pose.pt")
# use yolo26n-/model type/.pt for smallest and fastest / least resource intensive
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#model.predict(source=..., classes=[0])  # COCO person = 0

#set classes from data set to look for here
#model.set_classes()

detected_objects = []
detected_poses = []

wrist_history = deque(maxlen=15)

# Shared data
frame = None
annotated_frame = None
lock = threading.Lock()
running = True
frame_count = 0

# Check to see if GPU is available

print(sys.executable)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print(ultralytics.__version__)

# validation for keypoints being in frame
def is_valid_point(point):
    FRAME_HEIGHT = frame.shape[0]

    return (
        point is not None and
        len(point) == 2 and
        0 < point[1] < FRAME_HEIGHT
    )


#each keypoint stores an array with the following values: [x, y, visibility]
#for visibility, 0 = keypoint doesn't exist, 1 = keypoint exists but isn't visible in frame, 2 = fully visible
def determine_pose(keypoints):
    if keypoints is None or len(keypoints) == 0:
        return

    for person in keypoints:
        if len(person) <= 10:
            continue

        wrist = person[10]      # right wrist
        shoulder = person[6]    # right shoulder

        # skip invalid points
        if not (wrist is None or shoulder is None):
            # if confidence exists, check it
            if not(len(wrist) == 3 and wrist[2] < 0.5):
                # check if wrist is above shoulder
                if wrist[1] < shoulder[1]:
                    wrist_history.append(float(wrist[0]))

                    diffs = np.diff(wrist_history)
                    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

                    if sign_changes >= 8:
                        travel_distance = max(wrist_history) - min(wrist_history)

                        if travel_distance > 50:
                            if "Waving" not in detected_poses:
                                detected_poses.append("Waving")
                                print(detected_poses)
                                print(wrist_history)



            #sitting -- check to see if the difference between the hip and knee heights is small
            right_knee = person[15]
            left_knee = person[14]
            left_hip = person[12]
            right_hip = person[13]

            if all(is_valid_point(p) for p in [right_knee, left_knee, right_hip, left_hip]):
                print("E")
                if not (right_knee is None or right_hip is None or left_knee is None or left_hip is None):

                    avg_knee_y = (right_knee[1] + left_knee[1]) / 2
                    avg_hip_y = (right_hip[1] + left_hip[1]) / 2

                    print(avg_knee_y, avg_hip_y)
                    print(avg_hip_y - avg_knee_y)

                    if abs(avg_hip_y - avg_knee_y) < 5:
                        if "Sitting" not in detected_poses:
                            detected_poses.append("Sitting")
                            print(detected_poses)

                # Laying Down

                #Falling Detection


# Webcam Capture Thread

def capture_thread():
    global frame
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam failed")
        return

    while running:
        ret, img = cap.read()
        if not ret:
            continue

        with lock:
            frame = img.copy()

    cap.release()


# YOLO Inference Thread

def inference_thread():
    global frame, annotated_frame, frame_count

    while running:
        if frame is None:
            continue

        frame_count = frame_count + 1

        # Prevents multiple threads from accessing this data at the same time
        with lock:
            img = frame.copy()

        if frame_count % 10 != 0:
            continue

        # Run model (reduce size for speed)
        results = model.predict(img, imgsz=320, conf=0.4, verbose=False)

        annotated = results[0].plot()

        # Send Keypoints for Pose Detection
        for r in results:
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                keypoints = r.keypoints.xy.cpu().numpy()
                determine_pose(keypoints)

        # Add detected objects to array
        for r in results:
            # Convert from CUDA to Tensor
            boxes = r.boxes.data.cpu().numpy()

            for box in r.boxes.data:
                cls = int(box[5])
                name = model.names[cls]

                if name not in detected_objects:
                    detected_objects.append(model.names[cls])

        with lock:
            annotated_frame = annotated


# Display Loop (Main)

def display_loop():
    global annotated_frame

    while True:
        with lock:
            img = annotated_frame if annotated_frame is not None else frame

        if img is not None:
            cv2.imshow("Live Feed", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start threads
t1 = threading.Thread(target=capture_thread)
t2 = threading.Thread(target=inference_thread)

t1.start()
t2.start()

display_loop()

running = False
t1.join()
t2.join()
cv2.destroyAllWindows()



def main():
    pass

if __name__ == '__main__':
    main()



class Object:
    type = None
    time_on_frame = None
    quantity_of_objects = None
    confidence = None

    def __init__(self, name, time_on_frame, quantity_of_objects, confidence):
        self.name = name
        self.time_on_frame = time_on_frame
        self.quantity_of_objects = quantity_of_objects
        self.confidence = confidence


class Pose:
    pose = None
    start_of_detection = None
    end_of_detection = None
    quantity_of_pose = None
    people_in_frame = None

    def __init__(self, pose, start_of_detection, end_of_detection, quantity_of_pose):
        self.pose = pose
        self.start_of_detection = start_of_detection
        self.end_of_detection = end_of_detection
        self.quantity_of_pose = quantity_of_pose
        self.people_in_frame = []

