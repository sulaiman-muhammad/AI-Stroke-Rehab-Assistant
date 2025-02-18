import cv2
import google.generativeai as genai
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import win32gui
import win32con

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
genai.configure(api_key="REPLACE_WITH_YOUR_KEY")

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_feedback(exercise, reps, time):
    prompt = f"Stroke rehabilitation. I did {exercise} exercise: {reps} reps in {time:.2f} seconds. Provide short feedback."
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    print("Prompt: " + prompt)
    print("Feedback: " + response.text)
    return response.text

def make_window_topmost(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        win32gui.SetWindowPos(
            hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

def run_sit_to_stand():
    # Webcam input
    cap = cv2.VideoCapture(0)

    # Counter, stage, and time variables
    counter = 0
    stage = "sit"
    start_time = None
    end_time = None

    # Intro instructions
    speak_text("This is the sit-to-stand exercise tracker.")
    speak_text("To perform the sit to stand exercise, sit on a chair with both feet flat on the ground. Stand up and then sit back down, keeping your back straight and using your legs. Repeat this motion for the desired number of repetitions.")

    # Mediapipe pose detection
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the camera.")
                break

            # Convert frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # Convert frame back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                # Extract pose landmarks
                landmarks = results.pose_landmarks.landmark

                # Get required landmark heights
                hip_height = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                knee_height = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
                ankle_height = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                right_ankle_height = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y
                right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y
                nose = landmarks[mp_pose.PoseLandmark.NOSE.value].y

                hip_display = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                
                # Calculate the ratio between hip to ankle and hip to knee heights
                height_ratio = abs(hip_height - ankle_height) / abs(hip_height - knee_height)

                cv2.putText(image, f"Height Ratio: {height_ratio:.2f}",
                            tuple(np.multiply(hip_display, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Sit-to-stand logic
                if height_ratio < 2  and stage == "sit" and left_heel < 1 and right_heel < 1 and nose > 0 and ankle_height - 0.05 <= right_ankle_height <= ankle_height + 0.05 and hip_height <= knee_height - 0.08:
                    stage = "stand"
                    if counter == 0:
                        start_time = time.time() 
                    counter += 1

                if height_ratio > 2.25 and stage == "stand" and left_heel < 1 and right_heel < 1 and nose > 0 and ankle_height - 0.05 <= right_ankle_height <= ankle_height + 0.05 and hip_height <= knee_height - 0.08:
                    stage = "sit"
                    end_time = time.time()
                    threading.Thread(target=speak_text, args=(str(counter),)).start()
                    print(f"Reps: {counter}")

            except Exception as e:
                pass

            # Display counter and stage
            cv2.rectangle(image, (0, 0), (275, 75), (245, 117, 16), -1)
            cv2.putText(image, f"{counter} {stage}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw landmarks
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Show the feed
            cv2.imshow('Mediapipe Feed', image)
            make_window_topmost('Mediapipe Feed')

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Feedback using gemini
    if start_time and end_time:
        total_time = end_time - start_time
        feedback = get_feedback("Sit To Stand (Full Body)", counter, total_time)
        threading.Thread(target=speak_text, args=(feedback,)).start()
    else:
        total_time = 0
        feedback = get_feedback("Sit To Stand (Full Body)", counter, total_time)
        threading.Thread(target=speak_text, args=(feedback,)).start()

    return counter, total_time, feedback