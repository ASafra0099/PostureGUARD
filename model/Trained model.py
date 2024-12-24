import cv2
import mediapipe as mp
import math
from tkinter import Tk, Label, Button
import time
import sys

# Define your functions
def find_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_angle(x1, y1, x2, y2):
    theta = math.acos((y2 - y1) * (-y1) / (math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180 / math.pi) * theta
    return degree

def send_warning():
    def close_window():
        root.destroy()
    root = Tk()
    root.title("Posture Alert")
    msg = Label(root, text="Warning: Bad posture detected!\nPlease correct your posture.", padx=20, pady=20)
    msg.pack()
    ok_button = Button(root, text="OK", command=close_window)
    ok_button.pack()
    root.mainloop()

# Initialize pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Placeholder colors
yellow = (0, 255, 255)
pink = (255, 0, 255)
green = (0, 255, 0)
red = (0, 0, 255)
light_green = (50, 255, 50)

# Default font
font = cv2.FONT_HERSHEY_SIMPLEX

good_frames = 0
bad_frames = 0
cumulative_good_time = 0
cumulative_bad_time = 0
start_time = time.time()

bad_posture_alert_time = 10  # Time in seconds

# Camera connection code
url = 'https://192.168.0.102:8080/video'
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    sys.exit()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Resize the frame to (600, 400)
    image = cv2.resize(image, (600, 400))

    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    lm = keypoints.pose_landmarks
    if lm is None:
        continue

    lmPose = mp_pose.PoseLandmark
    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
    l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
    l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
    l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
    l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

    offset = find_distance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
    if offset < 100:
        cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
    else:
        cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

    neck_inclination = find_angle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    torso_inclination = find_angle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
    cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
    cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
    cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
    cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
    cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

    angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

    if neck_inclination < 40 and torso_inclination < 10:
        bad_frames = 0
        good_frames += 1
        cumulative_good_time += (1 / fps)
        cumulative_bad_time = 0  # Reset cumulative bad time when posture is corrected
        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)
    else:
        good_frames = 0
        bad_frames += 1
        cumulative_bad_time += (1 / fps)
        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

    total_active_time = time.time() - start_time

    if good_frames > 0:
        time_string_good = 'Good Posture Time : ' + str(round(cumulative_good_time, 1)) + 's'
        cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
    if bad_frames > 0:
        time_string_bad = 'Bad Posture Time : ' + str(round(cumulative_bad_time, 1)) + 's'
        cv2.putText(image, time_string_bad, (10, h - 50), font, 0.9, red, 2)

    total_time_string = 'Total Time : ' + str(round(total_active_time, 1)) + 's'
    cv2.putText(image, total_time_string, (10, h - 110), font, 0.9, yellow, 2)

    if total_active_time > 0:
        posture_score = (cumulative_good_time / total_active_time) * 100
        score_string = 'Posture Score : ' + str(round(posture_score, 1)) + '%'
        cv2.putText(image, score_string, (10, h - 80), font, 0.9, yellow, 2)

    if cumulative_bad_time >= bad_posture_alert_time:
        send_warning()
        cumulative_bad_time = 0  # Reset after the alert

    cv2.imshow('Posture Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
