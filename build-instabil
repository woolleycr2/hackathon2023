import cv2
import dlib
import time
from scipy.spatial import distance

# Indexare camera
cap = cv2.VideoCapture(0)

# Detectare faciala cu ajutorul librariei dlib
face_detector = dlib.get_frontal_face_detector()

# Locatia din care se incarca landmark-urile faciale
dlib_facelandmark = dlib.shape_predictor(r"C:\\Users\\Claudia\\Desktop\\test\\shape_predictor_68_face_landmarks.dat")

# Functia care calculeaza raportul de aspect al ochilor (Ratio)
def Detectare_ochi(eye):
    if len(eye) == 6:
        poi_A = distance.euclidean(eye[1], eye[5])
        poi_B = distance.euclidean(eye[2], eye[4])
        poi_C = distance.euclidean(eye[0], eye[3])
        aspect_ratio_ochi = (poi_A + poi_B) / (2 * poi_C)
        return aspect_ratio_ochi
    else:
        return 0  # sau orice alta valoare implicita
    
def Detectare_gura(mouth):
    if len(mouth) == 12:
        poi_A = distance.euclidean(mouth[2], mouth[10])
        poi_B = distance.euclidean(mouth[4], mouth[8])
        poi_C = distance.euclidean(mouth[0], mouth[6])
        aspect_ratio_gura = (poi_A + poi_B) / (2 * poi_C)
        return aspect_ratio_gura
    else:
        return 0  # sau orice alta valoare implicita

# Variabile pentru cronometru
cronometru_ochi_start = 0
cronometru_gura_start = 0
cronometru_durata = 2  # se masoara in secunde

# Bucla principala
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray_scale)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)
        ochi_si_gura = {'leftEye': [], 'rightEye': [], 'mouth': []}

        # Punctele ochiului stang (42 pana la 47), ochiului drept (36 până la 41) si gurii (48 pana la 59)
        for feature_name, start, end, color in [('leftEye', 42, 47, (255, 255, 0)),
                                                ('rightEye', 36, 41, (0, 255, 0)),
                                                ('mouth', 48, 59, (0, 0, 255))]:
            feature_points = ochi_si_gura[feature_name]
            for n in range(start, end + 1):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                feature_points.append((x, y))
                next_point = n + 1 if n < end else start
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame, (x, y), (x2, y2), color, 1)

        # Calcul raport de aspect pentru ochii stang si drept
        right_eye_ratio = Detectare_ochi(ochi_si_gura['rightEye'])
        left_eye_ratio = Detectare_ochi(ochi_si_gura['leftEye'])
        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
        mouth_ratio = Detectare_gura(ochi_si_gura['mouth'])
        # Rotunjirea valorii medii a ochilor stang si drept
        eye_ratio = round(eye_ratio, 2)

        ## citire valori pentru teste
        print("Ochi")
        print(eye_ratio)
        print("Gura")
        print(mouth_ratio)

        # Prag pentru detectarea somnolentei
        if mouth_ratio > 0.5:
            # Verifica daca durata alertei a trecut
            if time.time() - cronometru_gura_start >= cronometru_durata:
                cv2.putText(frame, "Cascare detectata", (30, 40),
                cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)

        if eye_ratio <= 0.160:
            # Verifica daca durata alertei a trecut
            if time.time() - cronometru_ochi_start >= cronometru_durata:
                cv2.putText(frame, "Ochi inchisi", (30, 60),
                cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)

        if 0.16 < eye_ratio < 0.225:
            # Verifica daca durata alertei a trecut
            if time.time() - cronometru_ochi_start >= cronometru_durata:
                cv2.putText(frame, "Oboseala detectata", (30, 80),
                cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
        else:
            # Reseteaza cronometrul daca nu se indeplinesc niciuna dintre conditii
            cronometru_gura_start = time.time()
            cronometru_ochi_start = time.time()

    cv2.imshow("Detector oboseala", frame)

    # Buton de quit in caz ca nu merge stop window-ul
    key = cv2.waitKey(1)
    if key == 27:
        break

# Eliberare camera si inchidere ferestre
cap.release()
cv2.destroyAllWindows()