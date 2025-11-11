import os
import cv2
import time
import threading
import requests
import numpy as np
from ultralytics import YOLO
import queue

# =====================================
# CONFIGURAÇÕES
# =====================================
CONF_THRESHOLD = 0.35
WEAPON_CONF_THRESHOLD = 0.25
RESOLUTION = (1280, 720)
FRAME_QUEUE_SIZE = 2  # Limita o backlog de frames

KNIFE_CLASSES = [
    'knife', 'blade', 'sword', 'machete', 'dagger',
    'cutting tool', 'sharp weapon', 'edged weapon', 'bayonet'
]

# =====================================
# ALERTA SONORO
# =====================================
try:
    from playsound import playsound
    def play_alert_sound():
        if os.path.exists("alerta.mp3"):
            playsound("alerta.mp3")
        else:
            import winsound
            winsound.Beep(1500, 1000)
except Exception:
    import winsound
    def play_alert_sound():
        winsound.Beep(1500, 1000)

# =====================================
# DOWNLOAD DE MODELOS
# =====================================
def download_knife_model():
    """Baixa modelo YOLOv8 para detecção de facas"""
    print("\n=== BAIXANDO MODELOS DE DETECÇÃO DE FACAS ===")

    knife_models = {
        "1": ("knife_yolov8n.pt", "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"),
        "2": ("knife_yolov8m.pt", "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"),
        "3": ("knife_yolov8l.pt", "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt")
    }

    print("\nEscolha o modelo YOLOv8:")
    print("1 - YOLOv8n (rápido e leve)")
    print("2 - YOLOv8m (equilíbrio entre precisão e velocidade) - RECOMENDADO")
    print("3 - YOLOv8l (alta precisão, mais pesado)")

    choice = input("Digite 1, 2 ou 3: ").strip()
    if choice not in knife_models:
        choice = "2"

    model_name, model_url = knife_models[choice]
    if os.path.exists(model_name) and os.path.getsize(model_name) > 1_000_000:
        print(f"✅ Modelo já existe: {model_name}")
        return model_name

    print(f"⬇️ Baixando {model_name}...")
    try:
        response = requests.get(model_url, stream=True, timeout=60)
        total_size = int(response.headers.get("content-length", 0))
        with open(model_name, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progresso: {percent:.1f}%", end="\r")
        print(f"\n✅ Download concluído: {model_name}")
        return model_name
    except Exception as e:
        print(f"❌ Erro ao baixar: {e}")
        return None

# =====================================
# MELHORIAS DE IMAGEM (SIMPLIFICADA)
# =====================================
def enhance_image(frame):
    """Melhora contraste de forma eficiente"""
    try:
        # Apenas CLAHE para melhor performance
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Reduzido clipLimit
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        pass
    return frame

# =====================================
# THREAD DE DETECÇÃO OTIMIZADA
# =====================================
class DetectionThread(threading.Thread):
    def __init__(self, model):
        super().__init__(daemon=True)
        self.model = model
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)  # Fila limitada
        self.result = None
        self.lock = threading.Lock()
        self.running = True
        self.last_frame_time = 0
        self.threat_count = 0  # Contador de ameaças

    def run(self):
        while self.running:
            try:
                # Pega frame da fila com timeout
                frame = self.frame_queue.get(timeout=0.1)
                
                # Processa apenas se tiver passado tempo suficiente desde o último frame
                current_time = time.time()
                if current_time - self.last_frame_time < 0.033:  
                    continue
                
                enhanced = enhance_image(frame)
                
                # Reduz resolução para inferência mais rápida
                inference_frame = cv2.resize(enhanced, (640, 360))
                
                results = self.model(inference_frame, conf=WEAPON_CONF_THRESHOLD, verbose=False)
                threat = False
                display_frame = frame.copy()  # Usa frame original para display

                for box in results[0].boxes:
                    cls_name = self.model.names[int(box.cls)].lower()
                    if any(k in cls_name for k in KNIFE_CLASSES):
                        threat = True
                        # Incrementa contador de ameaças
                        self.threat_count += 1
                        # Ajusta coordenadas para resolução original
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        x1 = int(x1 * 1280 / 640)
                        y1 = int(y1 * 720 / 360)
                        x2 = int(x2 * 1280 / 640)
                        y2 = int(y2 * 720 / 360)
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                        cv2.putText(display_frame, f"FACA {float(box.conf):.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 255), 2)

                with self.lock:
                    self.result = (display_frame, threat, self.threat_count)
                    self.last_frame_time = current_time
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erro na thread de detecção: {e}")
                continue

    def update_frame(self, frame):
        """Adiciona frame à fila se houver espaço"""
        try:
            # Limpa fila se estiver cheia para manter os frames mais recentes
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put(frame.copy(), block=False)
        except queue.Full:
            pass  # Descarta frame se a fila estiver cheia

    def get_result(self):
        with self.lock:
            return self.result

    def stop(self):
        self.running = False

# =====================================
# DETECTOR PRINCIPAL OTIMIZADO
# =====================================
def rodar_detector():
    print("🎯 Iniciando sistema de detecção de facas...")
    knife_path = download_knife_model()
    if not knife_path:
        print("❌ Falha ao obter modelo. Encerrando.")
        return

    model = YOLO(knife_path)
    
    # Configura câmera para melhor performance
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, 30)  # Define FPS desejado
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduz buffer para menor latência

    # Verifica configuração da câmera
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📷 Câmera configurada: {RESOLUTION[0]}x{RESOLUTION[1]} @ {actual_fps:.1f} FPS")

    detector = DetectionThread(model)
    detector.start()

    last_alert = 0
    count = 0
    frame_count = 0
    start_time = time.time()

    print("🚀 Iniciando detecção... Pressione 'q' para sair")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro ao capturar frame")
            break

        frame_count += 1

        # Envia frame para processamento (thread gerencia a fila)
        detector.update_frame(frame)

        # Recupera último resultado processado
        result = detector.get_result()
        if result:
            display_frame, threat, threat_count = result
        else:
            display_frame, threat, threat_count = frame, False, 0

        # Sistema de alerta
        if threat:
            count += 1
            if count > 2 and time.time() - last_alert > 3:
                threading.Thread(target=play_alert_sound, daemon=True).start()
                print("⚠️ FACA DETECTADA!")
                last_alert = time.time()
        else:
            count = max(count - 0.2, 0)

        # Mostra FPS atual
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # =====================================
        # INTERFACE VISUAL MELHORADA
        # =====================================
        
        # Fundo semi-transparente para textos
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # Informações do sistema
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Contador de ameaças
        cv2.putText(display_frame, f"Ameacas: {threat_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Status de detecção
        status_text = "ALERTA! FACA DETECTADA" if threat else "Sistema Ativo"
        status_color = (0, 0, 255) if threat else (0, 255, 0)
        cv2.putText(display_frame, status_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Instruções para o usuário
        instructions = [
            "INSTRUCOES:",
            "1. Posicione o objeto na camera",
            "2. Sistema detecta automaticamente",
            "3. Alerta sonoro para ameacas",
            "Pressione 'Q' para sair"
        ]
        
        # Fundo para instruções
        instructions_overlay = display_frame.copy()
        cv2.rectangle(instructions_overlay, (10, display_frame.shape[0] - 140), 
                     (450, display_frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.addWeighted(instructions_overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # Texto das instruções
        for i, line in enumerate(instructions):
            y_pos = display_frame.shape[0] - 110 + (i * 25)
            cv2.putText(display_frame, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Detector de Facas YOLOv8 - Sistema de Seguranca", display_frame)
        
        # Processa eventos da janela com timeout curto
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    detector.stop()
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Detector finalizado. Total de ameaças detectadas: {threat_count}")

# =====================================
# MENU PRINCIPAL
# =====================================
if __name__ == "__main__":
    while True:
        print("\n=== SISTEMA DE DETECÇÃO DE FACAS ===")
        print("1 - Baixar modelo YOLOv8")
        print("2 - Iniciar detector")
        print("3 - Sair")
        op = input("Escolha: ").strip()
        if op == "1":
            download_knife_model()
        elif op == "2":
            rodar_detector()
        elif op == "3":
            print("Saindo...")
            break
        else:
            print("Opção inválida.")