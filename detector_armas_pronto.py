import os
import cv2
import time
import threading
import queue
import requests
from ultralytics import YOLO

# CONFIGURAÇÕES PRINCIPAIS
CUSTOM_MODEL = "weapon_detector.pt"
CONF_THRESHOLD = 0.5
THREAT_CLASSES = ['gun', 'pistol', 'rifle', 'knife', 'weapon', 'firearm', 'sword', 'blade']

# Configurações de performance
FRAME_SKIP = 2  # Processar 1 a cada 3 frames
FRAME_QUEUE_SIZE = 2
RESOLUTION = (640, 480)  # Reduzir resolução para melhor performance

# SOM DE ALERTA
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

# DOWNLOAD DE MODELO PRÉ-TREINADO (OTIMIZADO)
def download_pretrained_model():
    print("\n=== BAIXANDO MODELO PRÉ-TREINADO ===")
    
    weapon_models = {
        "1": ("yolov8n.pt", "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"),
        "2": ("yolov8s.pt", "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt")
    }
    
    print("Escolha o modelo base:")
    print("1 - YOLOv8 Nano (rápido, menos preciso)")
    print("2 - YOLOv8 Small (equilibrado)")
    
    choice = input("Digite 1 ou 2: ").strip()
    
    if choice in weapon_models:
        model_name, model_url = weapon_models[choice]
        
        if not os.path.exists(model_name):
            print(f"Baixando {model_name}...")
            try:
                response = requests.get(model_url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(model_name, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"Progresso: {percent:.1f}%", end='\r')
                print(f"\n✅ Modelo baixado: {model_name}")
            except Exception as e:
                print(f"❌ Erro no download: {e}")
                return None
        else:
            print(f"✅ Modelo já existe: {model_name}")
        
        return model_name
    else:
        print("❌ Opção inválida")
        return None

# PROCESSAMENTO EM THREAD SEPARADA
class DetectorThread(threading.Thread):
    def __init__(self, model, frame_queue, result_queue):
        threading.Thread.__init__(self)
        self.model = model
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.running = True
        self.daemon = True
        
    def run(self):
        frame_count = 0
        while self.running:
            try:
                # Pegar frame da queue com timeout
                frame_data = self.frame_queue.get(timeout=1)
                frame_count += 1
                
                # Processar apenas 1 a cada FRAME_SKIP frames
                if frame_count % FRAME_SKIP != 0:
                    self.frame_queue.task_done()
                    continue
                
                frame, frame_id = frame_data
                
                # Realizar detecção
                results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)
                
                # Processar resultados
                threat_detected = False
                processed_frame = frame.copy()
                
                if len(results) > 0 and hasattr(results[0], 'boxes'):
                    detections = results[0].boxes
                    
                    for box in detections:
                        conf = float(box.conf)
                        if conf < CONF_THRESHOLD:
                            continue
                            
                        cls = int(box.cls)
                        name = self.model.names[cls].lower()

                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        # Verificar se é uma ameaça
                        is_threat = any(threat in name for threat in THREAT_CLASSES)
                        
                        if is_threat:
                            threat_detected = True
                            # Desenhar caixa vermelha para ameaças
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(processed_frame, f"PERIGO: {name} {conf:.2f}", 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # Desenhar caixa verde para objetos não perigosos
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(processed_frame, f"{name} {conf:.2f}", 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Enviar resultado de volta (APENAS dados, não o frame)
                self.result_queue.put({
                    'threat_detected': threat_detected,
                    'frame_id': frame_id,
                    'detections': threat_detected  # Apenas flag
                })
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erro no processamento: {e}")
                continue
    
    def stop(self):
        self.running = False

# ==============================
# DETECTOR EM TEMPO REAL OTIMIZADO
# ==============================
def rodar_detector():
    # Carregar modelo
    if os.path.exists(CUSTOM_MODEL):
        model_path = CUSTOM_MODEL
        print("🔫 Usando modelo customizado de armas")
    else:
        model_path = download_pretrained_model()
        if not model_path:
            model_path = "yolov8n.pt"
            print("⚠️ Usando modelo YOLO padrão")
    
    # Carregar modelo uma única vez
    model = YOLO(model_path)
    print(f"[INFO] Modelo carregado: {model_path}")
    print(f"[INFO] Classes detectáveis: {list(model.names.values())}")

    # Configurar webcam com configurações otimizadas
    cap = cv2.VideoCapture(0)
    
    # Configurações para melhor performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, 30)  # Limitar FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduzir buffer
    
    if not cap.isOpened():
        print("❌ Erro ao abrir a webcam.")
        return

    # Criar queues para comunicação entre threads
    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    result_queue = queue.Queue()
    
    # Iniciar thread de detecção
    detector_thread = DetectorThread(model, frame_queue, result_queue)
    detector_thread.start()
    
    # Configurações da janela
    cv2.namedWindow("Detector de Armas e Facas - Otimizado", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detector de Armas e Facas - Otimizado", RESOLUTION[0], RESOLUTION[1])
    
    # Variáveis de controle
    last_alert = 0
    alert_count = 0
    frame_id = 0
    last_frame = None
    fps = 0
    frame_time = time.time()
    
    print("\n✅ Detector otimizado iniciado!")
    print("🔴 Alertas vermelhos: Armas/Facas detectadas")
    print("🟢 Caixas verdes: Outros objetos")
    print("💡 Pressione 'q' para sair | 'p' para pausar\n")

    paused = False
    
    try:
        while True:
            current_time = time.time()
            
            # CAPTURAR FRAME (sempre, mesmo quando pausado)
            ret, frame = cap.read()
            if not ret:
                print("❌ Erro ao capturar frame")
                break
            
            # Redimensionar frame para melhor performance
            frame = cv2.resize(frame, RESOLUTION)
            display_frame = frame.copy()
            
            if not paused:
                # Tentar colocar frame na queue (não bloquear se cheia)
                try:
                    frame_queue.put((frame, frame_id), timeout=0.001)
                    frame_id += 1
                except queue.Full:
                    # Descarta frames antigos se a queue estiver cheia
                    try:
                        frame_queue.get_nowait()
                        frame_queue.task_done()
                        frame_queue.put((frame, frame_id), timeout=0.001)
                        frame_id += 1
                    except:
                        pass
            
            # VERIFICAR RESULTADOS (não-bloqueante)
            threat_detected = False
            try:
                result = result_queue.get_nowait()
                threat_detected = result['threat_detected']
                
                # Ativar alerta sonoro se ameaça detectada
                if threat_detected and (current_time - last_alert > 3):
                    threading.Thread(target=play_alert_sound, daemon=True).start()
                    alert_count += 1
                    print(f"🚨 ALERTA {alert_count}: Arma/faca detectada! - {time.strftime('%H:%M:%S')}")
                    last_alert = current_time
                
                result_queue.task_done()
            except queue.Empty:
                pass
            
            # ADICIONAR ELEMENTOS VISUAIS NO DISPLAY_FRAME (não no processed_frame)
            if threat_detected:
                cv2.putText(display_frame, "ALERTA DE SEGURANCA!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Mostrar contadores
            cv2.putText(display_frame, f"Alertas: {alert_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # BOTÕES NA TELA - SEMPRE ADICIONADOS POR ÚLTIMO
            cv2.putText(display_frame, "[P] Pausar/Continuar", (10, RESOLUTION[1] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "[Q] Sair", (10, RESOLUTION[1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Status do sistema
            status_text = "PAUSADO" if paused else "ATIVO"
            status_color = (0, 255, 255) if paused else (0, 255, 0)
            cv2.putText(display_frame, f"Status: {status_text}", (RESOLUTION[0] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Se pausado, mostrar mensagem grande no centro
            if paused:
                cv2.putText(display_frame, "SISTEMA PAUSADO", 
                           (RESOLUTION[0]//2 - 120, RESOLUTION[1]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Calcular FPS
            if current_time - frame_time >= 1.0:
                fps = frame_id / (current_time - frame_time)
                frame_time = current_time
                frame_id = 0
            
            # MOSTRAR FRAME (sempre o display_frame)
            cv2.imshow("Detector de Armas e Facas - Otimizado", display_frame)
            
            # CONTROLES DE TECLADO - AGORA DEVE FUNCIONAR
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🎯 Saindo...")
                break
            elif key == ord('p'):
                paused = not paused
                status = "PAUSADO" if paused else "RETOMADO"
                print(f"⏸️ {status}")
            elif key == ord('c'):
                # Limpar alertas
                alert_count = 0
                print("🔄 Contador de alertas zerado")
            elif key != 255:  # Apenas para debug
                print(f"Tecla pressionada: {key} (chr: {chr(key) if key < 128 else '?'})")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário")
    
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
    
    finally:
        # Limpeza
        print("🔄 Finalizando...")
        detector_thread.stop()
        detector_thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Sessão finalizada. Total de alertas: {alert_count}")

# FUNÇÃO SIMPLIFICADA DE TREINAMENTO
def treinar_modelo_simples():
    print("\n🎯 Para treinar um modelo customizado:")
    print("1. Baixe um dataset de armas/facas no formato YOLO")
    print("2. Coloque na pasta 'dataset' com a estrutura:")
    print("   dataset/")
    print("   ├── images/train/")
    print("   ├── images/val/")
    print("   ├── labels/train/")
    print("   ├── labels/val/")
    print("   └── data.yaml")
    print("3. Execute: python treinar.py")
    
    input("\nPressione Enter para voltar...")

# ==============================
# MENU PRINCIPAL OTIMIZADO
# ==============================
if __name__ == "__main__":
    print("=== SISTEMA DE DETECÇÃO DE ARMAS - OTIMIZADO ===")
    print("1 - Baixar modelo pré-treinado")
    print("2 - Rodar detector otimizado (RECOMENDADO)")
    print("3 - Informações sobre treinamento")
    print("4 - Sair")
    
    opcao = input("Escolha uma opção (1-4): ").strip()

    if opcao == "1":
        download_pretrained_model()
        input("\nPressione Enter para voltar ao menu...")
    elif opcao == "2":
        rodar_detector()
    elif opcao == "3":
        treinar_modelo_simples()
    elif opcao == "4":
        print("👋 Até logo!")
    else:
        print("❌ Opção inválida.")