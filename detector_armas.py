"""
detector_armas_pronto.py
- Usa YOLOv8 (ultralytics) para detectar armas na webcam.
- Otimizado para CPU: reduz resolução, processa apenas 1 a cada N frames.
- Reduz falsos positivos: exige detecção persistente (K detections em janela W).
- Usa modelo local 'gun-detection.pt' se existir; caso contrário usa 'yolov8n.pt'.
- Toca 'alerta.mp3' se disponível (playsound), caso contrário usa winsound.Beep.
"""

import os
import cv2
import time
import threading
import numpy as np
from collections import deque, defaultdict

# try playsound (mp3). if não disponível, usa winsound (Windows)
try:
    from playsound import playsound
    def play_alert_sound():
        if os.path.exists("alerta.mp3"):
            playsound("alerta.mp3")
        else:
            # fallback beep
            import winsound
            winsound.Beep(1000, 700)
except Exception:
    import winsound
    def play_alert_sound():
        winsound.Beep(1000, 700)

# ultralytics YOLO
from ultralytics import YOLO

# ------------- CONFIGURAÇÃO -------------
SPECIAL_MODEL = "gun-detection.pt"
FALLBACK_MODEL = "yolov8n.pt"

PROCESS_EVERY_N_FRAMES = 2
RESIZE_WIDTH = 640
CONF_THRESHOLD = 0.35
SMOOTH_WINDOW = 7
SMOOTH_REQUIRED = 3
ALERT_COOLDOWN = 5.0
# ----------------------------------------

# threat keywords in English and Portuguese
THREAT_KEYWORDS = [
    'knife', 'serrated', 'serrated knife', 'gun', 'pistol', 'rifle', 'weapon', 'blade',
    'toothbrush', 'phone', 'cell phone', 'celular'
]
IGNORED_CLASSES = ['person', 'dog', 'cat', 'horse', 'cow', 'bird', 'car', 'truck']

# --- NOVO: imagem de alerta personalizada ---
alert_img_path = "alerta_img.png"  # nome do arquivo da imagem de alerta
alert_img = None
if os.path.exists(alert_img_path):
    alert_img = cv2.imread(alert_img_path, cv2.IMREAD_UNCHANGED)  # suporta PNG com transparência

model_path = SPECIAL_MODEL if os.path.exists(SPECIAL_MODEL) else FALLBACK_MODEL
print(f"[INFO] Carregando modelo: {model_path}")
model = YOLO(model_path)

recent_threats = deque(maxlen=SMOOTH_WINDOW)
last_alert_time = 0.0
show_alert = False  # flag para mostrar o alerta visual

def trigger_alert(reason_text):
    global last_alert_time, show_alert
    now = time.time()
    if now - last_alert_time < ALERT_COOLDOWN:
        return
    last_alert_time = now
    show_alert = True
    threading.Thread(target=play_alert_sound, daemon=True).start()
    print(f"[ALERTA] {reason_text}  - {time.strftime('%Y-%m-%d %H:%M:%S')}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Erro: não foi possível abrir a webcam.")
    exit(1)

frame_idx = 0
print("✅ Sistema iniciado — pressionar 'q' para sair.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if w != RESIZE_WIDTH:
            scale = RESIZE_WIDTH / w
            frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

        show_frame = frame.copy()

        if frame_idx % PROCESS_EVERY_N_FRAMES == 0:
            results = model(frame)
            boxes = results[0].boxes

            threat_found = False
            detected_labels = []

            for box in boxes:
                conf = float(box.conf)
                if conf < CONF_THRESHOLD:
                    continue

                cls_id = int(box.cls)
                cls_name = model.names[cls_id].lower() if hasattr(model, "names") else str(cls_id)
                detected_labels.append((cls_name, conf))

                if any(k in cls_name for k in THREAT_KEYWORDS):
                    threat_found = True

                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                except Exception:
                    coords = box.xyxy
                    if hasattr(coords, 'tolist'):
                        coords = coords.tolist()
                    if coords:
                        x1, y1, x2, y2 = map(int, coords[0])
                    else:
                        continue
                box_color = (0, 0, 255) if threat_found else (0, 255, 0)
                cv2.rectangle(show_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(show_frame, f"{cls_name} {conf:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            recent_threats.append(1 if threat_found else 0)

            if sum(recent_threats) >= SMOOTH_REQUIRED:
                trigger_alert("Objeto suspeito detectado (detecções persistentes)")
                recent_threats.clear()

        # ALERTA VISUAL: exibe imagem personalizada se show_alert estiver ativo
        if show_alert and alert_img is not None:
            h, w = show_frame.shape[:2]
            ah, aw = alert_img.shape[:2]
            # Redimensiona a imagem de alerta para ~30% da tela
            scale = min(w * 0.3 / aw, h * 0.3 / ah)
            new_size = (int(aw * scale), int(ah * scale))
            resized_alert = cv2.resize(alert_img, new_size, interpolation=cv2.INTER_AREA)
            ah, aw = resized_alert.shape[:2]
            # Coordenadas para centralizar
            x1 = w // 2 - aw // 2
            y1 = h // 2 - ah // 2
            x2 = x1 + aw
            y2 = y1 + ah
            # Se PNG com alpha, faz blending
            if resized_alert.shape[2] == 4:
                alpha_s = resized_alert[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(3):
                    show_frame[y1:y2, x1:x2, c] = (alpha_s * resized_alert[:, :, c] +
                                                   alpha_l * show_frame[y1:y2, x1:x2, c])
            else:
                show_frame[y1:y2, x1:x2] = resized_alert
            if time.time() - last_alert_time > 1.0:
                show_alert = False
        elif show_alert:
            # fallback: triângulo + exclamação se não houver imagem
            h, w = show_frame.shape[:2]
            triangle_height = int(h * 0.4)
            triangle_width = int(w * 0.3)
            center_x = w // 2
            center_y = h // 2
            pt1 = (center_x, center_y - triangle_height // 2)
            pt2 = (center_x - triangle_width // 2, center_y + triangle_height // 2)
            pt3 = (center_x + triangle_width // 2, center_y + triangle_height // 2)
            pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
            cv2.drawContours(show_frame, [pts], 0, (0, 0, 255), thickness=10)
            cv2.fillPoly(show_frame, [pts], (0, 0, 0))
            excl_font = cv2.FONT_HERSHEY_SIMPLEX
            excl_text = "!"
            excl_scale = triangle_height / 300
            excl_thickness = 10
            excl_size, _ = cv2.getTextSize(excl_text, excl_font, excl_scale, excl_thickness)
            excl_x = center_x - excl_size[0] // 2
            excl_y = center_y + excl_size[1] // 2
            cv2.putText(show_frame, excl_text, (excl_x, excl_y), excl_font, excl_scale, (0, 0, 255), excl_thickness, cv2.LINE_AA)
            if time.time() - last_alert_time > 1.0:
                show_alert = False

        cv2.putText(show_frame, f"FPS aprox: {int(1.0 / max(1e-3, (time.time() - (last := globals().get('___t', time.time())))))}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        globals()['___t'] = time.time()

        cv2.putText(show_frame, "Pressione 'q' para sair", (10, show_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Detector de Armas - Pronto", show_frame)

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrompido pelo usuário.")

finally:
    cap.release()
    cv2.destroyAllWindows()