import cv2
import dlib
import time
from scipy.spatial import distance
import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk

# Watermark Settings #
text_watermark = "505EXY!"
font = cv2.FONT_HERSHEY_COMPLEX
font_scale = 0.7
font_thickness = 2
font_color = (255, 255, 255)  # White

# Class for Drowsiness Detection App
class AplicatieDetectorOboseala:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Oboseala")
        self.root.configure(bg="black")   
        self.contor_oboseala = 0

        # Video Init
        self.cap = cv2.VideoCapture(0)
        self.paused = False

        # Blank Label (space between grid and title)
        self.label_sus = ttk.Label(root, background="black")
        self.label_sus.pack(pady=0)

        # Title Label
        self.label = ttk.Label(root, text="Detector de Oboseala", font=("Times New Roman", 20, "bold"), background="black", foreground="white")
        self.label.pack(pady=5)

        # Video Label
        self.video_frame = ttk.Label(root)
        self.video_frame.pack(pady=5)

        # Line for Buttons
        button_frame = ttk.Frame(root, padding="5")
        button_frame.pack()

        # On/Off Buttons
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_detection)
        self.start_button.pack(side="left", padx=3)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_detection)
        self.stop_button.pack(side="left", padx=3)

        # Line Close Button
        close_frame = ttk.Frame(root, padding="4")
        close_frame.pack()

        # Close Button
        self.close_button = ttk.Button(close_frame, text="Închide", command=self.close)
        self.close_button.pack(pady=0)

        # Watermark Image
        watermark_image = Image.open("logo.png")

        self.background_image = ImageTk.PhotoImage(watermark_image)

        # Watermark Frame
        image_frame = Frame(root)
        image_frame.pack(side="bottom")

        self.background_label = Label(image_frame, image=self.background_image, borderwidth=0, highlightthickness=0)
        self.background_label.pack()

        # Show alert time duration (2 seconds)
        self.alert_duration = 2.0

        # Variables for closed eyes/mouth alerts
        self.alert_ochi_active = False
        self.alert_gura_active = False
        self.alert_ochi_gura_active = False
        self.alert_ochi_start_time = 0
        self.alert_gura_start_time = 0
        self.alert_ochi_gura_start_time = 0

        # Delay
        self.frame_delay = 1

        self.update()

    def start_detection(self):
        self.paused = False
    
    def stop_detection(self):
        self.paused = True

    def close(self):
        self.cap.release()
        self.root.destroy()
    
    # Drowsiness detection function
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

    # Update the image in real time
    def update(self):
        if not self.paused:
            ret, frame = self.cap.read()
            if ret:
                frame_with_detection = self.detect_drowsiness(frame)

                # Add Watermark
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

        self.root.after(self.frame_delay, self.update)

def Detectare_ochi(ochi):
    if len(ochi) == 6:
        punct_A = distance.euclidean(ochi[1], ochi[5])
        punct_B = distance.euclidean(ochi[2], ochi[4])
        punct_C = distance.euclidean(ochi[0], ochi[3])
        aspect_ratio_ochi = (punct_A + punct_B) / (2 * punct_C)
        return aspect_ratio_ochi
    else:
        return 0
    
def Detectare_gura(gura):
    if len(gura) == 12:
        punct_A = distance.euclidean(gura[2], gura[10])
        punct_B = distance.euclidean(gura[4], gura[8])
        punct_C = distance.euclidean(gura[0], gura[6])
        aspect_ratio_gura = (punct_A + punct_B) / (2 * punct_C)
        return aspect_ratio_gura
    else:
        return 0

# Camera Index
cap = cv2.VideoCapture(0)

# Closed Eyes Variable
ochi_inchisi = None
timp_inchis = 2

face_detector = dlib.get_frontal_face_detector()

# .dat Landmarks file path
dlib_facelandmark = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")

def main():
    # Main Window
    root = tk.Tk()
    root.geometry("800x720")
    root.configure(bg="black")

    # Start App
    app = AplicatieDetectorOboseala(root)

    root.mainloop()

if __name__ == "__main__":
    main()
