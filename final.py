import cv2
import dlib
import time
from scipy.spatial import distance

# Indexare camera
cap = cv2.VideoCapture(0)

# Variabila pentru timer ochi inchisi

ochi_inchisi = None

timp_inchis = 2

# Detectare faciala cu ajutorul librariei dlib
face_detector = dlib.get_frontal_face_detector()

# Locatia din care se incarca landmark-urile faciale
dlib_facelandmark = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")

# Functia care calculeaza raportul de aspect al ochilor (Ratio)
def Detectare_ochi(eye):
    if len(eye) == 6:
        poi_A = distance.euclidean(eye[1], eye[5])
        poi_B = distance.euclidean(eye[2], eye[4])
        poi_C = distance.euclidean(eye[0], eye[3])
        aspect_ratio_ochi = (poi_A + poi_B) / (2 * poi_C)
        return aspect_ratio_ochi
    else:
        return 0  # or any other default value
    
def Detectare_gura(mouth):
    if len(mouth) == 12:
        poi_A = distance.euclidean(mouth[2], mouth[10])
        poi_B = distance.euclidean(mouth[4], mouth[8])
        poi_C = distance.euclidean(mouth[0], mouth[6])
        aspect_ratio_gura = (poi_A + poi_B) / (2 * poi_C)
        return aspect_ratio_gura
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
        gura = []

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

        # Landmark-uri pentru gura incep de la (48 - 59)
        for n in range(48, 60):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            gura.append((x, y))
            next_point = n + 1
            if n == 59:
                next_point = 48
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 0, 255), 1)

        # Calculate aspect ratio for left and right eye
        right_Eye = Detectare_ochi(rightEye)
        left_Eye = Detectare_ochi(leftEye)
        Eye_Rat = (left_Eye + right_Eye) / 2
        gura = Detectare_gura(gura)

        # Round off the value of the average mean of right and left eyes
        Eye_Rat = round(Eye_Rat, 2)

        # Threshold for drowsiness detection
        if gura > 0.8:
            cv2.putText(frame, "Cascare detectata", (30, 40),
                        cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
        if Eye_Rat <= 0.12:
            cv2.putText(frame, "Trezeste-te!", (50, 450),
                        cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
        if 0.17 < Eye_Rat < 0.19:
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

## Sursa pentru fisierul .dat cu landmark-uri 
## https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat