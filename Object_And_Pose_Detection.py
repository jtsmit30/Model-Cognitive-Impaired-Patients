import math
from collections import deque
from ultralytics import YOLO
import numpy as np
import cv2
import sys
import torch
import ultralytics
import threading
import time
from datetime import datetime, timezone

"""
@Author: <Owen Reid>

Handles the Computer-Vision, uses either a video source or webcam and performs object detection
and determines poses such as sitting and waving using keypoints from YOLO26
"""

#SWAP THIS TO WHATEVER THE NEWEST MODEL IS BEFORE TRAINING / USING
#model = YOLO("C:\\Users\\Socce\\PycharmProjects\\PythonProject\\runs\\detect\\train3\\weights\\best.pt")
model = YOLO("yolo26l-pose.pt")
obj_detection = YOLO("yolo26m.pt")
# use yolo26n-/model type/.pt for smallest and fastest / least resource intensive
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
obj_detection.to(device)

#model.predict(source=..., classes=[0])  # COCO person = 0

#set classes from data set to look for here
#model.set_classes()

wrist_history = deque(maxlen=100)
detected_objects = []
detected_poses = []
direction_changes = 0
time_stamp = 0

# Shared data
frame = None
annotated_frame = None
lock = threading.Lock()
running = True
frame_count = 0
#Webcam
#SOURCE = 0

#Video
SOURCE = "C:\\Users\\Socce\\Downloads\\Untitled design.mp4"

cap = cv2.VideoCapture(SOURCE)

allowed_objects = [k for k,v in obj_detection.names.items() if v in
                   ["fork", "cell phone", "person", "bottle", "cup", "knife", "spoon", "bowl", "banana", "apple", "chair", "couch",
                    "potted plant", "bed", "dining table", "tv", "laptop", "mouse", "remote", "keyboard", "microwave", "oven", "toaster",
                    "sink", "refrigerator", "book", "clock", "vase", "scissors"]]

# Check to see if GPU is available

print(sys.executable)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print(ultralytics.__version__)

class Object:
    name = None
    start_of_detection = None
    end_of_detection = None
    quantity_of_objects = None
    confidence = None

    def __init__(self, name, start_of_detection, end_of_detection, quantity_of_objects, confidence):
        self.name = name
        self.start_of_detection = start_of_detection
        self.end_of_detection = end_of_detection
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



def write_data_to_text_file(file_path):
    with open(file_path, "w") as file:
        file.write("Objects\n")
        for obj in detected_objects:
            file.write("Object Name: " + obj.name + ",Start Time: " + str(datetime.fromtimestamp(int(obj.start_of_detection))) + " " + time.tzname[0]
                       + ",End Time: " + str(datetime.fromtimestamp(int(obj.end_of_detection))) + " " + time.tzname[0] + ",Number of Detections: "
                       + str(obj.quantity_of_objects) + ",Confidence: " + str(obj.confidence) + "\n")

        file.write("Poses\n")
        for p in detected_poses:
            file.write("Pose: " + p.pose + ",Start Time: " + str(datetime.fromtimestamp(int(p.start_of_detection))) + " " + time.tzname[0] +
                       ",End Time: " + str(datetime.fromtimestamp(int(obj.end_of_detection))) + " " + time.tzname[0] +
                       ",Number of Detections: " + str(p.quantity_of_pose) + "\n")



# helper function for calculating distance between two keypoints
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# waving detection
def determine_waving(person, wave_threshold):
    global direction_changes
    """
    Checks if a single person is waving.

    :param person: YOLO keypoints array for a single person
    :param wave_threshold: How wide the wave must be in shoulder-widths
    :return: True if waving, False otherwise

    """
    """
    Calculate normalization factor: 
    This is used to make the detection more robust from varying distances since the size of the person
    will vary, so the persons shoulder width will be used to determine how far their hand must travel 
    to count as a wave
    
    """
    # Validation that the keypoints exist
    if len(person) <= 10:
        return False

    l_shoulder = person[5]
    r_shoulder = person[6]
    r_elbow = person[8]
    r_wrist = person[10]

    # Skip if confidence scores are too low
    if len(r_wrist) == 3 and r_wrist[2] < 0.5:
        return False
    if len(r_shoulder) == 3 and r_shoulder[2] < 0.5:
        return False
    if len(l_shoulder) == 3 and l_shoulder[2] < 0.5:
        return False
    if len(r_elbow) == 3 and r_elbow[2] < 0.5:
        return False

    # Calculate shoulder width for normalizing factor
    shoulder_width = calculate_distance(l_shoulder, r_shoulder)

    # Make sure factor isn't zero (shoulders overlapping in that case)
    if shoulder_width < 1:
        return False

    # Make sure hand is raised above elbow since this will always be the case when waving
    # wrist_y > elbow_y means the wrist is below the elbow, so clear the tracking of the wrist
    if r_wrist[1] > r_elbow[1]:
        wrist_history.clear()
        return False

    """
    Add the normalized x coordinates to the wrist history
    The objective here is to track relative to the shoulder to account for body movement while waving,
    as well as account for distance to the camera by using the normalization factor
    """
    relative_x = (r_wrist[0] - r_shoulder[0]) / shoulder_width
    wrist_history.append(float(relative_x))

    # Check for Waving Motion

    # At least 15 frames have been processed before we look for waving
    if len(wrist_history) >= 15:
        max_x = max(wrist_history)
        min_x = min(wrist_history)

        # If the total side-to-side distance exceeds the threshold
        if math.fabs(max_x - min_x) > wave_threshold:
            # Check for Direction Changes
            # We look at the difference between consecutive frames
            diffs = np.diff(list(wrist_history))

            # Filter out 0's from diffs array
            diffs = [d for d in diffs if d != 0]

            # Count how many times the movement changes from left to right
            # We only count significant moves to ignore jitters
            for i in range(len(diffs) - 1):
                # If product is negative, signs are different (one is +, one is -)
                if diffs[i] * diffs[i+1] < 0:
                    # Small dead-zone so tiny jitters don't count

                    if abs(diffs[i]) > 0.02:
                        direction_changes += 1

            # Two direction changes likely means a wave (e.g. Left, then right, then left again)
            if direction_changes >= 2:
                direction_changes = 0
                wrist_history.clear()
                return True

    return False


def determine_sitting(person, sitting_threshold):
    """
    Determines sitting by comparing vertical distances between the persons knees and hips

    :param person: YOLO keypoints array for a single person
    :param sitting_threshold: Minimum difference in vertical distances between the persons knees and hips
    :return: True if sitting, False otherwise
    """

    # Validate that the keypoints are included in person
    if len(person) <= 15:
        return False

    # We need both hips and both knees to make an accurate sitting judgment
    # If any of these are low confidence, return False immediately
    required_joints = [11, 12, 13, 14]  # L_hip, R_hip, L_knee, R_knee

    for idx in required_joints:
        conf = person[idx][2]
        if conf < 0.5:
            return False

    # Visibility Check
    # Hips (11, 12) and Knees (13, 14)
    # If the model returns 0 if they don't exist
    if np.any(person[11:15, :2] == 0):
        return False

    l_hip, r_hip = person[11], person[12]
    l_knee, r_knee = person[13], person[14]

    # Calculate the width of the lap (distance between knees)
    # When sitting dead-on, your knees spread or at least stay shoulder-width apart
    lap_width = abs(l_knee[0] - r_knee[0])

    # Calculate the 'Height' of the thigh (vertical hip-to-knee)
    thigh_height = (abs(l_knee[1] - l_hip[1]) + abs(r_knee[1] - r_hip[1])) / 2

    # Compare Width to Height
    # When standing, thigh_height is large and lap_width is small (Ratio < 1)
    # When sitting, thigh_height shrinks and lap_width stays same or grows (Ratio > 1)
    if thigh_height == 0: return False

    sitting_ratio = lap_width / thigh_height

    # If the lap is wider than the vertical height of the thighs the person is likely sitting.
    if sitting_ratio > sitting_threshold:
        return True

    return False

#each keypoint stores an array with the following values: [x, y, visibility]
def determine_pose(keypoints, frame_height):
    # Use for tracking how far apart detections happen
    current_time = time.time()

    for person in keypoints:
        action_overlap = False

        # Person is waving
        if determine_waving(person, 0.6):

            # Check for waving detected in last 30 seconds, combine actions if it exists
            for p in detected_poses:
                # If another waving event happened in the last 30 seconds, add this action to it
                if p.pose == "Waving" and p.end_of_detection+30 >= current_time:
                    p.end_of_detection = current_time
                    p.quantity_of_pose = p.quantity_of_pose+1
                    action_overlap = True
                    break

            # If no overlapping actions found
            if not action_overlap:
                pose = Pose("Waving", current_time, current_time, 1)
                detected_poses.append(pose)

        action_overlap = False

        if determine_sitting(person, 1):

            for p in detected_poses:
                if p.pose == "Sitting" and p.end_of_detection+30 >= current_time:
                    p.end_of_detection = current_time
                    p.quantity_of_pose = p.quantity_of_pose+1
                    action_overlap = True

            if not action_overlap:
                pose = Pose("Sitting", current_time, current_time, 1)
                detected_poses.append(pose)


# Webcam Capture Thread

def capture_thread():
    global frame, SOURCE, frame_count, cap

    # Initialize values
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    delay = 1/fps
    print(delay)

    if not cap.isOpened():
        print("Error opening video stream")
        return

    while running:
        start_time = time.time()

        ret, img = cap.read()
        if not ret:
            print("Error reading frame")
            break

        with lock:
            frame = img.copy()

        elapsed_time = time.time() - start_time
        time.sleep(max(0, delay - elapsed_time))

    cap.release()

# YOLO Inference Thread

def inference_thread():
    global frame, annotated_frame, frame_count

    while running:
        if frame is None:
            # Prevents wasting CPU cycles
            time.sleep(0.01)
            continue

        # Prevents multiple threads from accessing this data at the same time
        with lock:
            frame_count += 1
            img = frame.copy()

        if frame_count % 3 != 0:
            continue

        # Run Pose model
        results = model.predict(img, imgsz=320, conf=0.5, verbose=False)

        # Run Object Detection model
        obj_results = obj_detection(img, imgsz=640, classes=allowed_objects, conf=0.6, verbose=False)

        annotated = obj_results[0].plot()

        curr_time = time.time()

        # Send Keypoints for Pose Detection
        for r in results:
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                keypoints = r.keypoints.data.cpu().numpy()
                determine_pose(keypoints, frame.shape[0])

        # Add detected objects to array
        for r in obj_results:
            # Convert from CUDA to Tensor
            boxes = r.boxes.data.cpu().numpy()

            # Get data from YOLO output
            for box in r.boxes.data:
                cls = int(box[5])
                name = obj_detection.names[cls]

                overlapping_obj = False

                # Check for preexisting objects of the same type and which have been detected in the last 30 seconds
                for ob in detected_objects:
                    if ob.name == name and ob.end_of_detection+30 >= curr_time:
                        ob.end_of_detection = curr_time
                        ob.quantity_of_objects = ob.quantity_of_objects+1
                        ob.confidence = ((ob.confidence * (ob.quantity_of_objects - 1)) + float(box[4])) / ob.quantity_of_objects
                        overlapping_obj = True
                        break

                # No similar objects in last 30 seconds
                if not overlapping_obj:
                    obj = Object(name, curr_time, curr_time, 1, float(box[4]))
                    detected_objects.append(obj)

        with lock:
            annotated_frame = annotated


# Display Loop (Main)
def display_loop():
    global annotated_frame, time_stamp, SOURCE, cap

    while True:
        current_time = time.time()

        # Update image, make sure nothing else is trying to access it
        with lock:
            img = annotated_frame if annotated_frame is not None else frame

        if img is not None:
            cv2.imshow("Live Feed", img)

        # Close Frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            write_data_to_text_file("computer_vision_output.txt")
            break

        # Write objects and poses to file every 60 seconds
        if current_time - time_stamp >= 60:
            write_data_to_text_file("computer_vision_output.txt")
            time_stamp = current_time


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



