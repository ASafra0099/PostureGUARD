import cv2
import mediapipe as mp
import math
from tkinter import Tk, Label

def find_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_angle(x1, y1, x2, y2):
    theta = math.acos((y2 - y1) * (-y1) / (math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180 / math.pi) * theta
    return degree

def send_warning():
    root = Tk()
    root.title("Posture Alert")
    msg = Label(root, text="Warning: Bad posture detected!\nPlease correct your posture.", padx=20, pady=20)
    msg.pack()
    root.after(2000, root.destroy)  # Display the message for 2 seconds
    root.mainloop()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

good_frames = 0
bad_frames = 0
cumulative_good_time = 0
cumulative_bad_time = 0
total_time = 0

bad_posture_alert_time = 10  # Time in seconds

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Error fetching frame")
        break

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_time += 1 / fps
    h, w = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    lm = keypoints.pose_landmarks
    if lm:
        lmPose = mp_pose.PoseLandmark
        l_shldr_x, l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        r_shldr_x, r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w), int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        l_ear_x, l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].x * w), int(lm.landmark[lmPose.LEFT_EAR].y * h)

        neck_inclination = find_angle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)

        if neck_inclination < 40:
            good_frames += 1
            cumulative_good_time += 1 / fps
            bad_frames = 0
            cumulative_bad_time = 0
        else:
            bad_frames += 1
            cumulative_bad_time += 1 / fps

        # Calculate posture score
        total_active_time = cumulative_good_time + cumulative_bad_time
        posture_score = (cumulative_good_time / total_active_time) * 100 if total_active_time > 0 else 0

        # Show posture information on screen     
        cv2.putText(image, f'Good Posture Time: {cumulative_good_time:.2f} sec', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Bad Posture Time: {cumulative_bad_time:.2f} sec', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Total Time: {total_time:.2f} sec', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Posture Score: {posture_score:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        if cumulative_bad_time >= bad_posture_alert_time:
            send_warning()
            cumulative_bad_time = 0  # Reset after the alert

    cv2.imshow('Posture Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
