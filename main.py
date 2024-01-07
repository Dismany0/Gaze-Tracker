import cv2
import numpy as np
import dlib

# Webcam make sure its plugged in
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

debug = False
while True:
    # Read the frame and convert to gray
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    for face in faces:

        # rec of face
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        if debug: cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        landmarks = predictor(gray, face)

        # draw a dot on each landmark
        if debug:
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                # label
                cv2.putText(frame, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        # Left Eye
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = ((landmarks.part(37).x + landmarks.part(38).x) // 2, (landmarks.part(37).y + landmarks.part(38).y) // 2)
        center_bottom = ((landmarks.part(41).x + landmarks.part(40).x) // 2, (landmarks.part(41).y + landmarks.part(40).y) // 2)

        horizontal_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 1)
        vertical_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 1)

        horizontal_line_length = np.linalg.norm(np.array(left_point) - np.array(right_point))
        vertical_line_length = np.linalg.norm(np.array(center_top) - np.array(center_bottom))
        ratio = horizontal_line_length / vertical_line_length

        if ratio > 5:
            cv2.putText(frame, "Left Eye Wink", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            

        # Right Eye
        # left_point = (landmarks.part(42).x, landmarks.part(42).y)
        # right_point = (landmarks.part(45).x, landmarks.part(45).y)
        # center_top = ((landmarks.part(43).x + landmarks.part(44).x) // 2, (landmarks.part(43).y + landmarks.part(44).y) // 2)
        # center_bottom = ((landmarks.part(47).x + landmarks.part(46).x) // 2, (landmarks.part(47).y + landmarks.part(46).y) // 2)

        # horizontal_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 1)
        # vertical_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 1)

        


    # Show the frame
    cv2.imshow("Frame", frame)

    # Quit on q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # Debug on d
    if cv2.waitKey(1) & 0xFF == ord("d"):
        debug = not debug

cap.release()
cv2.destroyAllWindows()
