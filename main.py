import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize MediaPipe Hand Detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Screen size
screen_width, screen_height = pyautogui.size()

# Status display
status_text = ""
status_start_time = 0
status_duration = 2  # seconds

# Cooldowns
last_scroll_time = 0
scroll_cooldown = 1  # seconds
last_click_time = 0
click_cooldown = 1  # seconds

# Sensitivity
left_click_threshold = 0.05

# Distance function
def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

# Webcam start
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    height, width, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_img)

    # Use only the right hand
    if result.multi_hand_landmarks and result.multi_handedness:
        right_hand_data = []

        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            handedness_label = hand_info.classification[0].label  # 'Right' or 'Left'
            if handedness_label == 'Right':
                # Estimate closeness for sorting
                tips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]
                avg_x = sum(t.x for t in tips) / 5
                avg_y = sum(t.y for t in tips) / 5
                dist_to_center = math.hypot(avg_x - 0.5, avg_y - 0.5)
                right_hand_data.append((hand_landmarks, dist_to_center))

        if right_hand_data:
            # Use the closest right hand
            right_hand_data.sort(key=lambda x: x[1])
            closest_hand = right_hand_data[0][0]

            # Landmarks
            thumb_tip = closest_hand.landmark[4]
            thumb_ip = closest_hand.landmark[3]
            index_tip = closest_hand.landmark[8]
            index_base = closest_hand.landmark[6]
            middle_tip = closest_hand.landmark[12]
            middle_base = closest_hand.landmark[10]
            ring_tip = closest_hand.landmark[16]
            ring_base = closest_hand.landmark[14]
            pinky_tip = closest_hand.landmark[20]
            pinky_base = closest_hand.landmark[18]

            # Map index to screen
            screen_x = screen_width * index_tip.x
            screen_y = screen_height * index_tip.y

            # --- Mouse Move ---
            if index_tip.y < index_base.y and \
               middle_tip.y > middle_base.y and \
               ring_tip.y > ring_base.y and \
               pinky_tip.y > pinky_base.y:
                pyautogui.moveTo(screen_x, screen_y)
                status_text = "Mouse Moved"
                status_start_time = time.time()

            # --- Left Click ---
            if distance(index_tip, thumb_tip) < left_click_threshold:
                if time.time() - last_click_time > click_cooldown:
                    pyautogui.click()
                    status_text = "Click"
                    last_click_time = time.time()
                    status_start_time = time.time()

            # --- Right Click ---
            if index_tip.y < index_base.y and \
               middle_tip.y < middle_base.y and \
               ring_tip.y > ring_base.y and \
               pinky_tip.y > pinky_base.y:
                if time.time() - last_click_time > click_cooldown:
                    pyautogui.rightClick()
                    status_text = "Right Click"
                    last_click_time = time.time()
                    status_start_time = time.time()

            # --- Scroll Up ---
            if index_tip.y < index_base.y and \
               middle_tip.y < middle_base.y and \
               ring_tip.y < ring_base.y and \
               pinky_tip.y < pinky_base.y and \
               thumb_tip.x < thumb_ip.x:
                if time.time() - last_scroll_time > scroll_cooldown:
                    pyautogui.scroll(300)
                    status_text = "Scroll Up"
                    last_scroll_time = time.time()
                    status_start_time = time.time()

            # --- Scroll Down ---
            if index_tip.y < index_base.y and \
               pinky_tip.y < pinky_base.y and \
               middle_tip.y > middle_base.y and \
               ring_tip.y > ring_base.y:
                if time.time() - last_scroll_time > scroll_cooldown:
                    pyautogui.scroll(-300)
                    status_text = "Scroll Down"
                    last_scroll_time = time.time()
                    status_start_time = time.time()

            # Draw right hand only
            mp_drawing.draw_landmarks(img, closest_hand, mp_hands.HAND_CONNECTIONS)

    # Show status
    if status_text and (time.time() - status_start_time < status_duration):
        cv2.putText(img, status_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show webcam feed
    cv2.imshow("Virtual Mouse", img)

    # ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
