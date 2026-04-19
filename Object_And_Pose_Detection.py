import math
import time
from collections import deque
from ultralytics import YOLO
import numpy as np
import cv2
import sys
import torch
import ultralytics
import threading
import time

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

wrist_history = deque(maxlen=100)
detected_objects = []
detected_poses = []
direction_changes = 0

# Shared data
frame = None
annotated_frame = None
lock = threading.Lock()
running = True
frame_count = 0
#Webcam
SOURCE = 0

#Video
#SOURCE = "C:\\Users\\Socce\\OneDrive\\Pictures\\Camera Roll 1\\WIN_20260323_02_10_52_Pro.mp4"


# Check to see if GPU is available

print(sys.executable)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print(ultralytics.__version__)

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

    # Only check for a wave if we have a full buffer of frames
    if len(wrist_history) == wrist_history.maxlen:
        max_x = max(wrist_history)
        min_x = min(wrist_history)

        print(math.fabs(max_x - min_x))

        # If the total side-to-side distance exceeds our shoulder-width threshold
        if math.fabs(max_x - min_x) > wave_threshold:
            # Check for Direction Changes (The "Back and Forth" check)
            # We look at the difference between consecutive frames
            diffs = np.diff(list(wrist_history))

            # Count how many times the movement changes from left to right
            # We only count significant moves to ignore jitters
            for i in range(len(diffs) - 1):
                # If product is negative, signs are different (one is +, one is -)
                if diffs[i] * diffs[i+1] < 0:
                    # Small deadzone so tiny jitters don't count
                    if abs(diffs[i]) > 0.02:
                        direction_changes += 1
                        print(direction_changes)

            # A wave should have at least 2 direction changes (Left -> Right -> Left)
            if direction_changes >= 2:
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
    # If any of these are invisible (low confidence), return False immediately
    required_joints = [11, 12, 13, 14]  # L_hip, R_hip, L_knee, R_knee

    for idx in required_joints:
        conf = person[idx][2]
        if conf < 0.5:  # 0.5 is the standard 'visibility' cutoff
            return False

    # Visibility Check
    # Hips (11, 12) and Knees (13, 14)
    # If the model returns 0 for these, we are standing too close
    if np.any(person[11:15, :2] == 0):
        return False

    l_hip, r_hip = person[11], person[12]
    l_knee, r_knee = person[13], person[14]

    # Calculate the 'Width' of the lap (distance between knees)
    # When sitting dead-on, your knees spread or at least stay shoulder-width apart
    lap_width = abs(l_knee[0] - r_knee[0])

    # Calculate the 'Height' of the thigh (vertical hip-to-knee)
    thigh_height = (abs(l_knee[1] - l_hip[1]) + abs(r_knee[1] - r_hip[1])) / 2

    # Compare Width to Height
    # When standing, thigh_height is large and lap_width is small (Ratio < 1)
    # When sitting, thigh_height shrinks and lap_width stays same or grows (Ratio > 1)
    if thigh_height == 0: return False

    sitting_ratio = lap_width / thigh_height

    # If the lap is wider than the vertical height of the thighs, you're likely sitting.
    if sitting_ratio > sitting_threshold:
        return True

    return False

#each keypoint stores an array with the following values: [x, y, visibility]
#for visibility, 0 = keypoint doesn't exist, 1 = keypoint exists but isn't visible in frame, 2 = fully visible
def determine_pose(keypoints, frame_height):

    for person in keypoints:

        if "Waving" not in detected_poses and determine_waving(person, 0.4):
            detected_poses.append("Waving")
            print("Waving Detected")

        if "Sitting" not in detected_poses and determine_sitting(person, 1):
            detected_poses.append("Sitting")
            print("Sitting Detected")



# Webcam Capture Thread

def capture_thread():
    global frame, SOURCE, frame_count
    cap = cv2.VideoCapture(SOURCE)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    delay = 1/fps

    if not cap.isOpened():
        print("Webcam failed")
        return

    while running:
        start_time = time.time()

        ret, img = cap.read()
        if not ret:
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

        if frame_count % 10 != 0:
            continue

        # Run model (reduce size for speed)
        results = model.predict(img, imgsz=320, conf=0.4, verbose=False)

        annotated = results[0].plot()

        # Send Keypoints for Pose Detection
        for r in results:
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                keypoints = r.keypoints.data.cpu().numpy()
                determine_pose(keypoints, frame.shape[0])

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

