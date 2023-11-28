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

# Locatia din care se incarca landmark-urile faciale
dlib_facelandmark = dlib.shape_predictor(r"C:\\Users\\Claudia\\Desktop\\test\\shape_predictor_68_face_landmarks.dat")

# Functia care calculeaza raportul de aspect al ochilor (Ratio)
def Detectare_ochi(eye):
    if len(eye) == 11:
        poi_A = distance.euclidean(eye[4], eye[8])
        poi_B = distance.euclidean(eye[2], eye[10])
      ##  poi_C = distance.euclidean(eye[0], eye[6])
        aspect_ratio_ochi = poi_A + poi_B
        return aspect_ratio_ochi
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
        gura = []

        # Left eye points (48 - 60)
        for n in range(48, 60):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            gura.append((x, y))
            next_point = n + 1
            if n == 59:
                next_point = 48
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        # Calculate aspect ratio for left and right eye
        gura = Detectare_ochi(gura)
        Eye_Rat = gura

        # Round off the value of the average mean of right and left eyes
        Eye_Rat = round(Eye_Rat, 2)

        # Threshold for drowsiness detection
        if Eye_Rat < 0.01:
            cv2.putText(frame, "Oboseala detectata", (50, 100),
                        cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
            print(Eye_Rat)

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