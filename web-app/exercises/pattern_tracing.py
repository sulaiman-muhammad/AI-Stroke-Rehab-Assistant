import cv2
import google.generativeai as genai
import mediapipe as mp
import pyttsx3
import random
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

def get_feedback(exercise, targets, time):
    prompt = f"Stroke rehabilitation. I did {exercise} exercise: {targets} targets touched in {time:.2f} seconds. Provide short feedback."
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

def run_pattern_tracing():
    # Webcam input
    cap = cv2.VideoCapture(0)

    # Counter, target variables, and timing
    counter = 0
    num_targets = 5  # Total number of targets
    target_position = [random.randint(100, 500), random.randint(100, 400)]  # Initial target position
    target_radius = 15  # Radius of the target
    move_interval = 0.5  # Time in seconds between movements
    last_move_time = time.time()
    start_time = None
    end_time = None

    # Check if a landmark is near the target
    def is_near_target(point, target, threshold=30):
        distance = ((point[0] - target[0])**2 + (point[1] - target[1])**2)**0.5
        return distance < threshold

    # Generate a random position for the target within the screen bounds
    def generate_random_position():
        return [random.randint(100, 500), random.randint(100, 400)]

    # Intro instructions
    speak_text("This is the pattern tracing exercise tracker.")
    speak_text("To perform the pattern tracing exercise, trace the moving points on the screen using your hands or legs. Repeat this motion until you hit 5 targets.")

    # Mediapipe pose detection
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the camera.")
                break

            # Mirror the frame (flip horizontally)
            frame = cv2.flip(frame, 1)

            # Convert frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # Convert frame back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Check if the required number of targets has been hit
            if counter >= num_targets:
                break

            try:
                # Extract pose landmarks
                landmarks = results.pose_landmarks.landmark

                # Get positions for hand and foot
                left_hand = [int(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x * 640),
                            int(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y * 480)]
                left_foot = [int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * 640),
                            int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * 480)]
                right_hand = [int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x * 640),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y * 480)]
                right_foot = [int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * 640),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * 480)]

                # Draw the current target point
                cv2.circle(image, tuple(target_position), target_radius, (0, 0, 255), -1)

                # Check if a hand or foot is near the current target
                if (is_near_target(left_hand, target_position) or
                        is_near_target(left_foot, target_position) or
                        is_near_target(right_hand, target_position) or
                        is_near_target(right_foot, target_position)):
                    if counter == 0:
                        start_time = time.time()
                    end_time = time.time()
                    counter += 1
                    threading.Thread(target=speak_text, args=(str(counter),)).start()
                    target_position = generate_random_position()  # Move to the next random position

                # Move the target periodically
                if time.time() - last_move_time > move_interval:
                    target_position[0] += random.choice([-20, 20])
                    target_position[1] += random.choice([-20, 20])
                    # Keep target within screen bounds
                    target_position[0] = max(50, min(target_position[0], 590))
                    target_position[1] = max(50, min(target_position[1], 430))
                    last_move_time = time.time()

            except Exception as e:
                pass

            # Display counter
            cv2.rectangle(image, (0, 0), (350, 75), (245, 117, 16), -1)
            cv2.putText(image, f"Targets Hit: {counter} / {num_targets}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

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
        feedback = get_feedback("Pattern Tracing (Balance and Coordination)", counter, total_time)
        threading.Thread(target=speak_text, args=(feedback,)).start()
    else:
        total_time = 0
        feedback = get_feedback("Pattern Tracing (Balance and Coordination)", counter, total_time)
        threading.Thread(target=speak_text, args=(feedback,)).start()

    return counter, total_time, feedback