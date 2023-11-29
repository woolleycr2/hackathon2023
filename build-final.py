import cv2
import dlib
import time
from scipy.spatial import distance
import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk

# Setări pentru Watermark #
text_watermark = "505EXY!"  # Schimbă cu textul dorit
font = cv2.FONT_HERSHEY_COMPLEX
font_scale = 0.7
font_thickness = 2
font_color = (255, 255, 255)  # Alb

# Clasa pentru aplicația de detectare a oboselii
class AplicatieDetectorOboseala:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Oboseala")
        self.root.configure(bg="black")
        
        # Contor pentru oboseala acumulata de catre sofer 
        
        self.contor_oboseala = 0
        # Inițializare captură video
        self.cap = cv2.VideoCapture(0)
        self.paused = False

        # Blank Label (spatiu intre grid si titlu)
        self.label_sus = ttk.Label(root, background="black")
        self.label_sus.pack(pady=0)

        # Etichetă pentru titlul aplicației
        self.label = ttk.Label(root, text="Detector de Oboseala", font=("Times New Roman", 20, "bold"), background="black", foreground="white")
        self.label.pack(pady=5)

        # Cadru pentru afișarea imaginii video
        self.video_frame = ttk.Label(root)
        self.video_frame.pack(pady=5)

        # Linie pentru aranjarea butoanelor
        button_frame = ttk.Frame(root, padding="5")
        button_frame.pack()

        # Butoanele de pornire și oprire
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_detection)
        self.start_button.pack(side="left", padx=3)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_detection)
        self.stop_button.pack(side="left", padx=3)

        # Linie Buton Close
        close_frame = ttk.Frame(root, padding="4")
        close_frame.pack()

        # Buton pentru închiderea aplicației
        self.close_button = ttk.Button(close_frame, text="Închide", command=self.close)
        self.close_button.pack(pady=0)

        # Încarcă imaginea și redimensionează-o
        watermark_image = Image.open("logo2.png")  # înlocuiește cu calea ta

        self.background_image = ImageTk.PhotoImage(watermark_image)

        # Afișează imaginea sub butoane într-un frame separat
        image_frame = Frame(root)
        image_frame.pack(side="bottom")

        # Adaugă label pentru imaginea de fundal
        self.background_label = Label(image_frame, image=self.background_image, borderwidth=0, highlightthickness=0)
        self.background_label.pack()

        # Variabile cod nou
        # Durata pentru afișarea alertei (în secunde)
        self.alert_duration = 2.0

        # Variabile pentru alerte la închiderea ochilor și a gurii
        self.alert_ochi_active = False
        self.alert_gura_active = False
        self.alert_ochi_gura_active = False
        self.alert_ochi_start_time = 0
        self.alert_gura_start_time = 0
        self.alert_ochi_gura_start_time = 0

        # Variabilă pentru întârzierea dintre cadre (în milisecunde)
        self.frame_delay = 1  # Setează întârzierea dorită în milisecunde

        # Inițializare buclă principală
        self.update()

    # Metodă pentru pornirea detecției
    def start_detection(self):
        self.paused = False
    
    # Metodă pentru oprirea detecției
    def stop_detection(self):
        self.paused = True

    # Metodă pentru închiderea aplicației
    def close(self):
        # Eliberare resurse camera
        self.cap.release()
        self.root.destroy()

    # Variabila de detectare a oboselii 
    
    
    # Metodă pentru detectarea oboselii într-un frame
    def detect_drowsiness(self, frame):
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_scale)
        cv2.putText(frame, "Status:", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if self.contor_oboseala >= 15 and self.contor_oboseala < 40:
                cv2.putText(frame, "Pauza recomandata", (10, 420),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        if self.contor_oboseala >= 40:
                cv2.putText(frame, "Opriti vehiculul intr-un loc sigur!", (10, 420),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        cv2.putText(frame, "Oboseala acumulata:", (10, 445),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, str(self.contor_oboseala), (250, 445),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for face in faces:
            face_landmarks = dlib_facelandmark(gray_scale, face)
            ochi_si_gura = {'leftEye': [], 'rightEye': [], 'mouth': []}

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

            aspect_ratio_ochi_drept = Detectare_ochi(ochi_si_gura['rightEye'])
            aspect_ratio_ochi_stang = Detectare_ochi(ochi_si_gura['leftEye'])
            aspect_ratio_ochi = (aspect_ratio_ochi_stang + aspect_ratio_ochi_drept) / 2
            aspect_ratio_gura = Detectare_gura(ochi_si_gura['mouth'])

            aspect_ratio_ochi = round(aspect_ratio_ochi, 2)

            if aspect_ratio_gura > 0.50 and aspect_ratio_ochi <= 0.20:
                if not self.alert_ochi_gura_active:
                    self.alert_ochi_gura_start_time = time.time()
                    self.alert_ochi_gura_active = True

                if time.time() - self.alert_ochi_gura_start_time >= self.alert_duration:
                    cv2.putText(frame, "Adormit", (90, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (21, 56, 210), 2)
                    if self.contor_oboseala <= 40:
                        self.contor_oboseala += 40

            if aspect_ratio_ochi <= 0.20 and aspect_ratio_gura < 0.50:
                if not self.alert_ochi_active:
                    self.alert_ochi_start_time = time.time()
                    self.alert_ochi_active = True

                if time.time() - self.alert_ochi_start_time >= self.alert_duration:
                    cv2.putText(frame, "Adormit", (90, 470),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (21, 56, 210), 2)
                    if self.contor_oboseala <= 40:
                        self.contor_oboseala += 20
            else:
                self.alert_ochi_active = False

            if aspect_ratio_gura > 0.50 and aspect_ratio_ochi > 0.20:
                if not self.alert_gura_active:
                    self.alert_gura_start_time = time.time()
                    self.alert_gura_active = True

                if time.time() - self.alert_gura_start_time >= self.alert_duration:
                    cv2.putText(frame, "Obosit", (90, 470),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (21, 56, 210), 2)
                    if self.contor_oboseala <= 40:
                        self.contor_oboseala += 1
            else:
                self.alert_gura_active = False

        return frame

    # Metodă pentru actualizarea imaginii în timp real
    def update(self):
        if not self.paused:
            ret, frame = self.cap.read()
            if ret:
                frame_with_detection = self.detect_drowsiness(frame)

                # Adaugă watermark text în colțul dreapta jos
                text_size = cv2.getTextSize(text_watermark, font, font_scale, font_thickness)[0]
                text_x = frame_with_detection.shape[1] - text_size[0] - 10
                text_y = frame_with_detection.shape[0] - 10

                cv2.putText(frame_with_detection, text_watermark, (text_x, text_y),
                        font, font_scale, font_color, font_thickness)

                frame_with_detection = cv2.cvtColor(frame_with_detection, cv2.COLOR_BGR2RGB)
                frame_with_detection = Image.fromarray(frame_with_detection)
                frame_with_detection = ImageTk.PhotoImage(frame_with_detection)
                self.video_frame.imgtk = frame_with_detection
                self.video_frame.configure(image=frame_with_detection)

        self.root.after(self.frame_delay, self.update)  # Programarea următoarei actualizări cu o întârziere

def Detectare_ochi(ochi):
    if len(ochi) == 6:
        punct_A = distance.euclidean(ochi[1], ochi[5])
        punct_B = distance.euclidean(ochi[2], ochi[4])
        punct_C = distance.euclidean(ochi[0], ochi[3])
        aspect_ratio_ochi = (punct_A + punct_B) / (2 * punct_C)
        return aspect_ratio_ochi
    else:
        return 0  # sau orice altă valoare implicită
    
def Detectare_gura(gura):
    if len(gura) == 12:
        punct_A = distance.euclidean(gura[2], gura[10])
        punct_B = distance.euclidean(gura[4], gura[8])
        punct_C = distance.euclidean(gura[0], gura[6])
        aspect_ratio_gura = (punct_A + punct_B) / (2 * punct_C)
        return aspect_ratio_gura
    else:
        return 0  # sau orice altă valoare implicită

# Indexarea camerei video
cap = cv2.VideoCapture(0)

# Variabilă pentru temporizatorul ochilor închiși
ochi_inchisi = None
timp_inchis = 2

# Detectarea facială cu ajutorul bibliotecii dlib
face_detector = dlib.get_frontal_face_detector()

# Locația de unde se încarcă punctele de referință faciale (landmark-urile)
dlib_facelandmark = dlib.shape_predictor(r"C:\\Users\\Claudia\\Desktop\\test\\shape_predictor_68_face_landmarks.dat")

def main():
    # Crearea ferestrei principale
    root = tk.Tk()
    root.geometry("800x720")
    root.configure(bg="black")

    # Inițializarea aplicației
    app = AplicatieDetectorOboseala(root)

    # Pornirea buclei principale a ferestrei
    root.mainloop()

if __name__ == "__main__":
    main()
