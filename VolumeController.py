import cv2
import mediapipe as mp
import pyautogui

x1 = y1 = x2 = y2 = 0
webcam = cv2.VideoCapture(0)

my_hands = mp.solutions.hands.Hands()  # detecting and tracking hand landmarks
drawing_utils = mp.solutions.drawing_utils  # drawing_utils is a module containing utility functions for drawing
# annotations on images or video frames, which can be useful for visualizing the output of the Hands model.

while True:
    _, image = webcam.read()
    image = cv2.flip(image, 1)
    frame_height, frame_width, _ = image.shape

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                (0, 17), (17, 18), (18, 19), (19, 20)  # Little finger
            ]
            # Draw landmarks and connections
            drawing_utils.draw_landmarks(image, hand, connections)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                # Convert normalized landmark coordinates to pixel coordinates
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:
                    x1 = x
                    y1 = y
                    # Adjust the circle position to be closer to the actual tip of the thumb
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3)
                if id == 4:
                    x2 = x
                    y2 = y
                    # Adjust the circle position to be closer to the actual tip of the index finger
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 0, 255), thickness=3)

        d = ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)

        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        if d > 50:
            pyautogui.press("volumeup")
        else:
            pyautogui.press("volumedown")

    cv2.imshow("Volume Controller with Hand", image)
    # Check for key press or if the close button is clicked
    key = cv2.waitKey(10)
    if key == ord('q') or key == 27:  # 'q' key or ESC key
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
