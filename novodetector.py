import os
import cv2
import time
import threading
import queue
import requests
from ultralytics import YOLO

# CONFIGURAÇÕES PRINCIPAIS
CUSTOM_MODEL = "weapon_detector.pt"
WEAPON_MODEL = "gun_detection.pt"  # NOVO: modelo específico para armas
CONF_THRESHOLD = 0.5
WEAPON_CONF_THRESHOLD = 0.3  # MAIS BAIXO para armas

# Classes expandidas para armas
THREAT_CLASSES = ['gun', 'pistol', 'rifle', 'knife', 'weapon', 'firearm', 'sword', 'blade', 
                  'handgun', 'revolver', 'shotgun', 'machine gun', 'firearm', 'ammunition']

# Classes ESPECÍFICAS para armas de fogo 
FIREARM_CLASSES = ['gun', 'pistol', 'rifle', 'handgun', 'revolver', 'shotgun', 
                   'firearm', 'machine gun', 'ammunition', 'bullet']

# Configurações de performance
FRAME_SKIP = 2
FRAME_QUEUE_SIZE = 2
RESOLUTION = (640, 480)

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

# ==============================
# DOWNLOAD DE MODELOS DE ARMAS (NOVO)
# ==============================
def download_weapon_model():
    """Baixa modelo ESPECIALIZADO em detecção de armas"""
    print("\n=== BAIXANDO MODELO ESPECIALIZADO EM ARMAS ===")
    
    weapon_models = {
        "1": ("gun_detection.pt", "https://github.com/kkrtolwyk/weapon_detection/releases/download/v1.0/yolov8n_weapon.pt"),
        "2": ("firearm_detector.pt", "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt")  # Fallback
    }
    
    print("Escolha o modelo de detecção de armas:")
    print("1 - Modelo Especializado em Armas (Recomendado)")
    print("2 - YOLO Padrão (Fallback)")
    
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
                
                print(f"\n✅ Modelo de armas baixado: {model_name}")
                return model_name
            except Exception as e:
                print(f"❌ Erro no download do modelo de armas: {e}")
                return None
        else:
            print(f"✅ Modelo de armas já existe: {model_name}")
            return model_name
    else:
        print("❌ Opção inválida")
        return None

def download_pretrained_model():
    """Baixa modelo YOLO padrão"""
    print("\n=== BAIXANDO MODELO PRÉ-TREINADO ===")
    
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    model_name = "yolov8n.pt"
    
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

# ==============================
# SISTEMA DE DETECÇÃO DUPLA (NOVO)
# ==============================
class DualDetectorThread(threading.Thread):
    def __init__(self, model_weapon, model_general, frame_queue, result_queue):
        threading.Thread.__init__(self)
        self.model_weapon = model_weapon  # Modelo específico para armas
        self.model_general = model_general  # Modelo geral
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.running = True
        self.daemon = True
        
    def run(self):
        frame_count = 0
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=1)
                frame_count += 1
                
                if frame_count % FRAME_SKIP != 0:
                    self.frame_queue.task_done()
                    continue
                
                frame, frame_id = frame_data
                processed_frame = frame.copy()
                
                # DETECÇÃO DUPLA (NOVO)
                threat_detected = False
                firearm_detected = False
                all_detections = []
                
                # 1. PRIMEIRO: Usar modelo ESPECÍFICO para armas
                if self.model_weapon:
                    weapon_results = self.model_weapon(frame, conf=WEAPON_CONF_THRESHOLD, verbose=False)
                    threat_detected, firearm_detected, processed_frame = self.process_weapon_detections(
                        weapon_results, processed_frame, threat_detected, firearm_detected, all_detections, "WEAPON"
                    )
                
                # 2. SEGUNDO: Usar modelo GERAL como fallback
                general_results = self.model_general(frame, conf=CONF_THRESHOLD, verbose=False)
                threat_detected, firearm_detected, processed_frame = self.process_general_detections(
                    general_results, processed_frame, threat_detected, firearm_detected, all_detections, "GENERAL"
                )
                
                # Enviar resultado
                self.result_queue.put({
                    'threat_detected': threat_detected,
                    'firearm_detected': firearm_detected,
                    'frame_id': frame_id,
                    'detections': all_detections
                })
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erro no processamento: {e}")
                continue
    
    def process_weapon_detections(self, results, frame, threat_detected, firearm_detected, all_detections, source):
        """Processa detecções do modelo de armas"""
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            for box in results[0].boxes:
                conf = float(box.conf)
                cls = int(box.cls)
                
                # Obter nome da classe
                if hasattr(self.model_weapon, 'names'):
                    name = self.model_weapon.names[cls].lower()
                else:
                    name = str(cls)
                
                # Verificar se é ameaça
                is_threat = any(threat in name for threat in THREAT_CLASSES)
                is_firearm = any(firearm in name for firearm in FIREARM_CLASSES)
                
                if is_threat or is_firearm:
                    threat_detected = True
                    if is_firearm:
                        firearm_detected = True
                    
                    # Desenhar detecção
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # CORES DIFERENCIADAS (NOVO)
                    if is_firearm:
                        color = (0, 0, 255)  # VERMELHO para armas de fogo
                        label = f"ARMA: {name} {conf:.2f}"
                    else:
                        color = (0, 165, 255)  # LARANJA para outras ameaças
                        label = f"AMEAÇA: {name} {conf:.2f}"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    all_detections.append({
                        'name': name,
                        'confidence': conf,
                        'is_firearm': is_firearm,
                        'source': source
                    })
        
        return threat_detected, firearm_detected, frame
    
    def process_general_detections(self, results, frame, threat_detected, firearm_detected, all_detections, source):
        """Processa detecções do modelo geral"""
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            for box in results[0].boxes:
                conf = float(box.conf)
                if conf < CONF_THRESHOLD:
                    continue
                    
                cls = int(box.cls)
                name = self.model_general.names[cls].lower()
                
                # Verificar se é ameaça
                is_threat = any(threat in name for threat in THREAT_CLASSES)
                is_firearm = any(firearm in name for firearm in FIREARM_CLASSES)
                
                if is_threat:
                    threat_detected = True
                    if is_firearm:
                        firearm_detected = True
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Só desenhar se não foi detectado pelo modelo de armas
                    already_detected = any(det['name'] == name for det in all_detections)
                    if not already_detected:
                        color = (255, 0, 0)  # AZUL para detecções gerais
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"GERAL: {name} {conf:.2f}", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    all_detections.append({
                        'name': name,
                        'confidence': conf,
                        'is_firearm': is_firearm,
                        'source': source
                    })
        
        return threat_detected, firearm_detected, frame
    
    def stop(self):
        self.running = False

# ==============================
# DETECTOR EM TEMPO REAL COM SISTEMA DUPLO
# ==============================
def rodar_detector_duplo():
    """Sistema de detecção DUPLO - Modelo específico + modelo geral"""
    
    # CARREGAR DOIS MODELOS (NOVO)
    models_loaded = []
    
    # 1. Tentar carregar modelo ESPECÍFICO para armas
    weapon_model = None
    if os.path.exists(WEAPON_MODEL):
        try:
            weapon_model = YOLO(WEAPON_MODEL)
            print("🔫 Modelo ESPECÍFICO de armas carregado!")
            models_loaded.append("Específico-Armas")
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo de armas: {e}")
    
    # 2. Se não tem modelo específico, tentar baixar
    if weapon_model is None:
        weapon_path = download_weapon_model()
        if weapon_path and os.path.exists(weapon_path):
            try:
                weapon_model = YOLO(weapon_path)
                print("🔫 Modelo de armas baixado e carregado!")
                models_loaded.append("Baixado-Armas")
            except Exception as e:
                print(f"⚠️ Erro ao carregar modelo baixado: {e}")
    
    # 3. Carregar modelo GERAL
    general_model = None
    if os.path.exists(CUSTOM_MODEL):
        general_model = YOLO(CUSTOM_MODEL)
        print("🎯 Modelo customizado carregado!")
        models_loaded.append("Customizado")
    else:
        general_path = download_pretrained_model()
        if general_path:
            general_model = YOLO(general_path)
            print("🌐 Modelo geral carregado!")
            models_loaded.append("Geral")
    
    if not models_loaded:
        print("❌ Nenhum modelo pôde ser carregado!")
        return
    
    print(f"[INFO] Modelos ativos: {', '.join(models_loaded)}")
    
    # Configurar webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Erro ao abrir a webcam.")
        return

    # Criar queues
    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    result_queue = queue.Queue()
    
    # Iniciar thread de detecção DUPLA (NOVO)
    detector_thread = DualDetectorThread(weapon_model, general_model, frame_queue, result_queue)
    detector_thread.start()
    
    # Configurações da janela
    cv2.namedWindow("Sistema Duplo - Detecção de Armas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sistema Duplo - Detecção de Armas", RESOLUTION[0], RESOLUTION[1])
    
    # Variáveis de controle
    last_alert = 0
    alert_count = 0
    frame_id = 0
    fps = 0
    frame_time = time.time()
    
    print("\n✅ SISTEMA DUPLO INICIADO!")
    print("🔴 Vermelho: Armas de fogo (Modelo Específico)")
    print("🟠 Laranja: Outras ameaças (Modelo Específico)") 
    print("🔵 Azul: Detecções do modelo geral")
    print("💡 Pressione 'q' para sair | 'p' para pausar\n")

    paused = False
    
    try:
        while True:
            current_time = time.time()
            
            # Capturar frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Erro ao capturar frame")
                break
            
            frame = cv2.resize(frame, RESOLUTION)
            display_frame = frame.copy()
            
            if not paused:
                # Enviar frame para processamento
                try:
                    frame_queue.put((frame, frame_id), timeout=0.001)
                    frame_id += 1
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                        frame_queue.task_done()
                        frame_queue.put((frame, frame_id), timeout=0.001)
                        frame_id += 1
                    except:
                        pass
            
            # Verificar resultados
            threat_detected = False
            firearm_detected = False
            try:
                result = result_queue.get_nowait()
                threat_detected = result['threat_detected']
                firearm_detected = result['firearm_detected']
                
                # ALERTA SONORO
                if threat_detected and (current_time - last_alert > 3):
                    threading.Thread(target=play_alert_sound, daemon=True).start()
                    alert_count += 1
                    
                    if firearm_detected:
                        print(f"🚨 ALERTA CRÍTICO {alert_count}: ARMA DE FOGO DETECTADA! - {time.strftime('%H:%M:%S')}")
                    else:
                        print(f"🚨 ALERTA {alert_count}: Ameaça detectada! - {time.strftime('%H:%M:%S')}")
                    
                    last_alert = current_time
                
                result_queue.task_done()
            except queue.Empty:
                pass
            
            # INTERFACE VISUAL MELHORADA (NOVO)
            if firearm_detected:
                cv2.putText(display_frame, "ALERTA CRÍTICO: ARMA DE FOGO!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif threat_detected:
                cv2.putText(display_frame, "ALERTA DE SEGURANCA!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            # LEGENDA DE CORES (NOVO)
            cv2.putText(display_frame, "VERMELHO: Arma de Fogo", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display_frame, "LARANJA: Outras Ameacas", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(display_frame, "AZUL: Detecao Geral", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Contadores
            cv2.putText(display_frame, f"Alertas: {alert_count}", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Status do sistema
            status_text = "PAUSADO" if paused else "ATIVO - SISTEMA DUPLO"
            status_color = (0, 255, 255) if paused else (0, 255, 0)
            cv2.putText(display_frame, f"Status: {status_text}", (RESOLUTION[0] - 250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Botões
            cv2.putText(display_frame, "[P] Pausar/Continuar", (10, RESOLUTION[1] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "[Q] Sair", (10, RESOLUTION[1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if paused:
                cv2.putText(display_frame, "SISTEMA PAUSADO", 
                           (RESOLUTION[0]//2 - 120, RESOLUTION[1]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Calcular FPS
            if current_time - frame_time >= 1.0:
                fps = frame_id / (current_time - frame_time)
                frame_time = current_time
                frame_id = 0
            
            # Mostrar frame
            cv2.imshow("Sistema Duplo - Detecção de Armas", display_frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🎯 Saindo...")
                break
            elif key == ord('p'):
                paused = not paused
                status = "PAUSADO" if paused else "RETOMADO"
                print(f"⏸️ {status}")
            elif key == ord('c'):
                alert_count = 0
                print("🔄 Contador de alertas zerado")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
    finally:
        print("🔄 Finalizando...")
        detector_thread.stop()
        detector_thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Sessão finalizada. Total de alertas: {alert_count}")

# ==============================
# MENU PRINCIPAL ATUALIZADO
# ==============================
if __name__ == "__main__":
    print("=== SISTEMA DUPLO DE DETECÇÃO DE ARMAS ===")
    print("1 - Baixar modelo especializado em armas")
    print("2 - Rodar detector DUPLO (RECOMENDADO)")
    print("3 - Rodar detector SIMPLES (original)")
    print("4 - Sair")
    
    opcao = input("Escolha uma opção (1-4): ").strip()

    if opcao == "1":
        download_weapon_model()
        input("\nPressione Enter para voltar ao menu...")
    elif opcao == "2":
        rodar_detector_duplo()  # NOVO: sistema duplo
    elif opcao == "3":
        # Aqui você manteria a função original rodar_detector()
        print("⚠️ Em desenvolvimento - use a opção 2")
        rodar_detector_duplo() 
    elif opcao == "4":
        print("👋 Até logo!")
    else:
        print("❌ Opção inválida.")