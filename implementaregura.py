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

# Functia care calculeaza raportul de aspect al gurii (Ratio)
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
        gura = []

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
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        # Apelare la functia care calculeaza aspect ratio-ul

        ## Linie de test pentru array-ul cu landmark-uri
        ## print(len(gura))
        gura = Detectare_gura(gura)
        ## Linie de test pentru afisarea valorii gura, pentru unit test mai usor
        ##print(gura)

        # Limitele la care se declanseaza alerta
        if gura > 0.5:
            cv2.putText(frame, "Cascare detectata", (30, 40),
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