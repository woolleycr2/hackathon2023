import cv2
import dlib
import time
from scipy.spatial import distance

# Indexare camera
cap = cv2.VideoCapture(0)

# Variabila pentru timer ochi inchisi

ochi_inchisi = None

# Detectare faciala cu ajutorul librariei dlib
face_detector = dlib.get_frontal_face_detector()

# Path to the shape_predictor_68_face_landmarks.dat file
dlib_facelandmark = dlib.shape_predictor(r"C:\\Users\\Claudia\\Desktop\\test\\shape_predictor_68_face_landmarks.dat")

# Function to calculate the aspect ratio for the eyes
def Detect_Eye(eye):
    if len(eye) == 6:
        poi_A = distance.euclidean(eye[1], eye[5])
        poi_B = distance.euclidean(eye[2], eye[4])
        poi_C = distance.euclidean(eye[0], eye[3])
        aspect_ratio_Eye = (poi_A + poi_B) / (2 * poi_C)
        return aspect_ratio_Eye
    else:
        return 0  # or any other default value

# Main loop
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray_scale)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)
        leftEye = []
        rightEye = []

        # Left eye points (42 to 47)
        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        # Right eye points (36 to 41)
        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

        # Calculate aspect ratio for left and right eye
        right_Eye = Detect_Eye(rightEye)
        left_Eye = Detect_Eye(leftEye)
        Eye_Rat = (left_Eye + right_Eye) / 2

        # Round off the value of the average mean of right and left eyes
        Eye_Rat = round(Eye_Rat, 2)

        # Threshold for drowsiness detection
        if Eye_Rat <= 0.160:
            cv2.putText(frame, "Trezeste-te!", (50, 450),
                        cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
        if 0.16 < Eye_Rat < 0.225:
            cv2.putText(frame, "Oboseala detectata", (50, 100),
                        cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)

    cv2.imshow("Detector oboseala", frame)

    # Buton de quit in caz ca nu merge stop window-ul
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()