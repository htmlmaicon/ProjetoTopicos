import os
import cv2
import time
import threading
import queue
import requests
import numpy as np
from ultralytics import YOLO

# CONFIGURAÇÕES OTIMIZADAS
CUSTOM_MODEL = "weapon_detector.pt"
WEAPON_MODEL = "knife_detector.pt"  # Foco em facas
CONF_THRESHOLD = 0.35  # MAIS BAIXO para melhor sensibilidade
WEAPON_CONF_THRESHOLD = 0.25  # AINDA MAIS BAIXO para armas

# Classes OTIMIZADAS para facas e armas
THREAT_CLASSES = [
    'knife', 'gun', 'pistol', 'rifle', 'weapon', 'firearm', 'sword', 'blade', 
    'handgun', 'revolver', 'shotgun', 'machine gun', 'ammunition', 'machete',
    'dagger', 'cutting tool', 'sharp weapon', 'edged weapon', 'bayonet'
]

# Classes ESPECÍFICAS para armas de fogo 
FIREARM_CLASSES = [
    'gun', 'pistol', 'rifle', 'handgun', 'revolver', 'shotgun', 
    'firearm', 'machine gun', 'ammunition', 'bullet'
]

# Classes ESPECÍFICAS para facas (NOVO)
KNIFE_CLASSES = [
    'knife', 'blade', 'sword', 'machete', 'dagger', 
    'cutting tool', 'sharp weapon', 'edged weapon', 'bayonet'
]

# Configurações de performance OTIMIZADAS
FRAME_SKIP = 1  # REDUZIDO para mais sensibilidade
FRAME_QUEUE_SIZE = 1
RESOLUTION = (1280, 720)  # AUMENTADO para mais detalhes

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
# DOWNLOAD DE MODELOS ESPECIALIZADOS EM FACAS
# ==============================
def download_knife_model():
    """Baixa modelos ESPECIALIZADOS em detecção de facas"""
    print("\n=== BAIXANDO MODELOS ESPECIALIZADOS EM FACAS ===")
    
    knife_models = {
        "1": ("knife_detector.pt", "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt"),
        "2": ("yolov8m_knife.pt", "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt"),
        "3": ("best_knife.pt", "https://github.com/keremberke/yolov8n-knife/releases/download/v0.1.0/best.pt")
    }
    
    print("Escolha o modelo de detecção de facas:")
    print("1 - YOLOv8s (Balanço entre velocidade e precisão) - RECOMENDADO")
    print("2 - YOLOv8m (Mais preciso, ideal para longa distância)")
    print("3 - Modelo Especializado em Facas (keremberke)")
    
    choice = input("Digite 1, 2 ou 3: ").strip()
    
    if choice in knife_models:
        model_name, model_url = knife_models[choice]
        
        if not os.path.exists(model_name):
            print(f"Baixando {model_name}...")
            try:
                response = requests.get(model_url, stream=True, timeout=30)
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
                
                print(f"\n✅ Modelo de facas baixado: {model_name}")
                return model_name
            except Exception as e:
                print(f"❌ Erro no download do modelo de facas: {e}")
                # Fallback para modelo padrão
                return download_pretrained_model()
        else:
            print(f"✅ Modelo de facas já existe: {model_name}")
            return model_name
    else:
        print("❌ Opção inválida, usando modelo padrão")
        return download_pretrained_model()

def download_pretrained_model():
    """Baixa modelo YOLO padrão otimizado"""
    print("\n=== BAIXANDO MODELO PRÉ-TREINADO ===")
    
    # Usar modelo médio para melhor detecção à distância
    model_url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt"
    model_name = "yolov8m.pt"
    
    if not os.path.exists(model_name):
        print(f"Baixando {model_name} (otimizado para detecção à distância)...")
        try:
            response = requests.get(model_url, stream=True, timeout=30)
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
            # Fallback para modelo nano
            model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
            model_name = "yolov8n.pt"
            if not os.path.exists(model_name):
                print("Tentando baixar YOLOv8n como fallback...")
                response = requests.get(model_url)
                with open(model_name, 'wb') as f:
                    f.write(response.content)
    else:
        print(f"✅ Modelo já existe: {model_name}")
    
    return model_name

# ==============================
# PRÉ-PROCESSAMENTO PARA DETECÇÃO À DISTÂNCIA
# ==============================
def enhance_image(frame):
    """Melhora a imagem para melhor detecção de objetos pequenos"""
    try:
        # Aumentar contraste usando CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Sharpening leve para realçar bordas
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel)
        
    except Exception as e:
        print(f"⚠️ Erro no enhancement: {e}")
    
    return frame

# ==============================
# SISTEMA DE DETECÇÃO TRIPLO OTIMIZADO
# ==============================
class AdvancedDetectorThread(threading.Thread):
    def __init__(self, model_knife, model_weapon, model_general, frame_queue, result_queue):
        threading.Thread.__init__(self)
        self.model_knife = model_knife    # Modelo específico para facas
        self.model_weapon = model_weapon  # Modelo para armas
        self.model_general = model_general  # Modelo geral
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.running = True
        self.daemon = True
        self.detection_cache = []  # Cache para melhorar detecção contínua
        
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
                
                # APLICAR MELHORIA DE IMAGEM
                enhanced_frame = enhance_image(frame)
                processed_frame = enhanced_frame.copy()
                
                # DETECÇÃO TRIPLA OTIMIZADA
                threat_detected = False
                firearm_detected = False
                knife_detected = False
                all_detections = []
                
                # 1. PRIMEIRO: Modelo ESPECÍFICO para facas
                if self.model_knife:
                    knife_results = self.model_knife(enhanced_frame, conf=WEAPON_CONF_THRESHOLD, verbose=False)
                    threat_detected, firearm_detected, knife_detected, processed_frame = self.process_knife_detections(
                        knife_results, processed_frame, threat_detected, firearm_detected, knife_detected, all_detections, "KNIFE"
                    )
                
                # 2. SEGUNDO: Modelo para armas em geral
                if self.model_weapon:
                    weapon_results = self.model_weapon(enhanced_frame, conf=WEAPON_CONF_THRESHOLD, verbose=False)
                    threat_detected, firearm_detected, knife_detected, processed_frame = self.process_weapon_detections(
                        weapon_results, processed_frame, threat_detected, firearm_detected, knife_detected, all_detections, "WEAPON"
                    )
                
                # 3. TERCEIRO: Modelo GERAL como fallback
                general_results = self.model_general(enhanced_frame, conf=CONF_THRESHOLD, verbose=False)
                threat_detected, firearm_detected, knife_detected, processed_frame = self.process_general_detections(
                    general_results, processed_frame, threat_detected, firearm_detected, knife_detected, all_detections, "GENERAL"
                )
                
                # Atualizar cache de detecções
                self.update_detection_cache(all_detections)
                
                # Enviar resultado
                self.result_queue.put({
                    'threat_detected': threat_detected,
                    'firearm_detected': firearm_detected,
                    'knife_detected': knife_detected,
                    'frame_id': frame_id,
                    'detections': all_detections,
                    'processed_frame': processed_frame
                })
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erro no processamento: {e}")
                continue
    
    def update_detection_cache(self, current_detections):
        """Mantém cache de detecções para melhor rastreamento"""
        self.detection_cache = current_detections
        if len(self.detection_cache) > 5:
            self.detection_cache = self.detection_cache[-5:]
    
    def process_knife_detections(self, results, frame, threat_detected, firearm_detected, knife_detected, all_detections, source):
        """Processa detecções específicas de facas"""
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            for box in results[0].boxes:
                conf = float(box.conf)
                cls = int(box.cls)
                
                # Obter nome da classe
                if hasattr(self.model_knife, 'names'):
                    name = self.model_knife.names[cls].lower()
                else:
                    name = str(cls)
                
                # Verificar se é faca
                is_knife = any(knife in name for knife in KNIFE_CLASSES)
                is_threat = is_knife or any(threat in name for threat in THREAT_CLASSES)
                
                if is_threat:
                    threat_detected = True
                    if is_knife:
                        knife_detected = True
                    
                    # Desenhar detecção
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # COR ESPECIAL PARA FACAS (NOVO)
                    color = (0, 255, 255)  # AMARELO para facas
                    label = f"FACA: {name} {conf:.2f}"
                    
                    # Desenhar mesmo objetos pequenos (removido filtro de área)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    all_detections.append({
                        'name': name,
                        'confidence': conf,
                        'is_firearm': False,
                        'is_knife': True,
                        'source': source
                    })
        
        return threat_detected, firearm_detected, knife_detected, frame
    
    def process_weapon_detections(self, results, frame, threat_detected, firearm_detected, knife_detected, all_detections, source):
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
                is_knife = any(knife in name for knife in KNIFE_CLASSES)
                
                if is_threat:
                    threat_detected = True
                    if is_firearm:
                        firearm_detected = True
                    if is_knife:
                        knife_detected = True
                    
                    # Desenhar detecção
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # CORES DIFERENCIADAS
                    if is_firearm:
                        color = (0, 0, 255)  # VERMELHO para armas de fogo
                        label = f"ARMA: {name} {conf:.2f}"
                    elif is_knife:
                        color = (0, 255, 255)  # AMARELO para facas
                        label = f"FACA: {name} {conf:.2f}"
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
                        'is_knife': is_knife,
                        'source': source
                    })
        
        return threat_detected, firearm_detected, knife_detected, frame
    
    def process_general_detections(self, results, frame, threat_detected, firearm_detected, knife_detected, all_detections, source):
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
                is_knife = any(knife in name for knife in KNIFE_CLASSES)
                
                if is_threat:
                    threat_detected = True
                    if is_firearm:
                        firearm_detected = True
                    if is_knife:
                        knife_detected = True
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Só desenhar se não foi detectado pelos modelos específicos
                    already_detected = any(
                        det['name'] == name and det['confidence'] > conf - 0.1 
                        for det in all_detections
                    )
                    
                    if not already_detected:
                        color = (255, 0, 0)  # AZUL para detecções gerais
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"GERAL: {name} {conf:.2f}", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    all_detections.append({
                        'name': name,
                        'confidence': conf,
                        'is_firearm': is_firearm,
                        'is_knife': is_knife,
                        'source': source
                    })
        
        return threat_detected, firearm_detected, knife_detected, frame
    
    def stop(self):
        self.running = False

# ==============================
# DETECTOR EM TEMPO REAL SUPER OTIMIZADO
# ==============================
def rodar_detector_avancado():
    """Sistema de detecção TRIPLO - Máxima sensibilidade para facas"""
    
    print("🎯 INICIANDO SISTEMA AVANÇADO DE DETECÇÃO DE FACAS")
    
    # CARREGAR TRÊS MODELOS (NOVO)
    models_loaded = []
    
    # 1. Modelo ESPECÍFICO para facas
    knife_model = None
    knife_path = download_knife_model()
    if knife_path and os.path.exists(knife_path):
        try:
            knife_model = YOLO(knife_path)
            print("🔪 Modelo ESPECÍFICO de facas carregado!")
            models_loaded.append("Específico-Facas")
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo de facas: {e}")
    
    # 2. Modelo para armas em geral
    weapon_model = None
    if os.path.exists(WEAPON_MODEL):
        try:
            weapon_model = YOLO(WEAPON_MODEL)
            print("🔫 Modelo de armas carregado!")
            models_loaded.append("Armas")
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo de armas: {e}")
    
    # 3. Modelo GERAL
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
    print(f"[INFO] Sensibilidade: FACAS={WEAPON_CONF_THRESHOLD}, GERAL={CONF_THRESHOLD}")
    
    # Configurar webcam OTIMIZADA
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Aumentar brilho para melhor detecção
    
    if not cap.isOpened():
        print("❌ Erro ao abrir a webcam.")
        return

    # Criar queues
    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    result_queue = queue.Queue()
    
    # Iniciar thread de detecção AVANÇADA
    detector_thread = AdvancedDetectorThread(knife_model, weapon_model, general_model, frame_queue, result_queue)
    detector_thread.start()
    
    # Configurações da janela
    cv2.namedWindow("Sistema Avançado - Detecção de Facas e Armas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sistema Avançado - Detecção de Facas e Armas", RESOLUTION[0], RESOLUTION[1])
    
    # Variáveis de controle
    last_alert = 0
    alert_count = 0
    frame_id = 0
    fps = 0
    frame_time = time.time()
    consecutive_detections = 0
    
    print("\n✅ SISTEMA AVANÇADO INICIADO!")
    print("🟡 AMARELO: Facas (Alta Prioridade)")
    print("🔴 Vermelho: Armas de fogo") 
    print("🟠 Laranja: Outras ameaças")
    print("🔵 Azul: Detecções do modelo geral")
    print("💡 Pressione 'q' para sair | 'p' para pausar | 's' para ajustar sensibilidade\n")

    paused = False
    
    try:
        while True:
            current_time = time.time()
            
            # Capturar frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Erro ao capturar frame")
                break
            
            if frame.shape[1] != RESOLUTION[0] or frame.shape[0] != RESOLUTION[1]:
                frame = cv2.resize(frame, RESOLUTION)
            
            display_frame = frame.copy()
            
            if not paused:
                # Enviar frame para processamento
                try:
                    if frame_queue.qsize() < FRAME_QUEUE_SIZE:
                        frame_queue.put((frame, frame_id), timeout=0.001)
                        frame_id += 1
                    else:
                        # Limpar queue se estiver cheia
                        try:
                            frame_queue.get_nowait()
                            frame_queue.task_done()
                        except:
                            pass
                except:
                    pass
            
            # Verificar resultados
            threat_detected = False
            firearm_detected = False
            knife_detected = False
            
            try:
                result = result_queue.get_nowait()
                threat_detected = result['threat_detected']
                firearm_detected = result['firearm_detected']
                knife_detected = result['knife_detected']
                
                # Atualizar display com frame processado
                if 'processed_frame' in result:
                    display_frame = result['processed_frame']
                
                # Sistema de confirmação para alertas
                if threat_detected:
                    consecutive_detections = min(consecutive_detections + 1, 5)
                else:
                    consecutive_detections = max(consecutive_detections - 0.5, 0)
                
                # ALERTA SONORO com confirmação
                confirmed_threat = consecutive_detections >= 2
                
                if confirmed_threat and (current_time - last_alert > 2):  # Alerta mais frequente
                    threading.Thread(target=play_alert_sound, daemon=True).start()
                    alert_count += 1
                    
                    if firearm_detected:
                        print(f"🚨 ALERTA CRÍTICO {alert_count}: ARMA DE FOGO! - {time.strftime('%H:%M:%S')}")
                    elif knife_detected:
                        print(f"⚠️ ALERTA FACAS {alert_count}: FACA DETECTADA! - {time.strftime('%H:%M:%S')}")
                    else:
                        print(f"🔔 ALERTA {alert_count}: Ameaça detectada - {time.strftime('%H:%M:%S')}")
                    
                    last_alert = current_time
                
                result_queue.task_done()
            except queue.Empty:
                consecutive_detections = max(consecutive_detections - 0.2, 0)
                pass
            
            # INTERFACE VISUAL AVANÇADA
            # Fundo semi-transparente para informações
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (RESOLUTION[0], 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
            
            # Status de ameaça em DESTAQUE
            if firearm_detected and confirmed_threat:
                cv2.putText(display_frame, "🚨 ALERTA MÁXIMO: ARMA DE FOGO!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                # Piscar tela em vermelho
                if int(time.time() * 3) % 2 == 0:
                    display_frame = cv2.addWeighted(display_frame, 0.8, 
                                                  np.full(display_frame.shape, (0, 0, 255), dtype=np.uint8), 
                                                  0.2, 0)
            elif knife_detected and confirmed_threat:
                cv2.putText(display_frame, "⚠️ ALERTA: FACA DETECTADA!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
            elif threat_detected:
                cv2.putText(display_frame, "🔍 ANALISANDO POSSÍVEL AMEAÇA...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(display_frame, "✅ SISTEMA SEGURO - MONITORANDO", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # LEGENDA DE CORES COMPLETA
            cv2.putText(display_frame, "🎯 SISTEMA AVANÇADO - DETECÇÃO À DISTÂNCIA", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "🟡 AMARELO: Facas", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(display_frame, "🔴 VERMELHO: Armas de Fogo", (10, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(display_frame, "🟠 LARANJA: Outras Ameaças", (10, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            cv2.putText(display_frame, "🔵 AZUL: Detecções Gerais", (10, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Estatísticas em tempo real
            cv2.putText(display_frame, f"📊 Alertas: {alert_count}", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"🎯 FPS: {fps:.1f}", (150, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"🔍 Confiança: {consecutive_detections:.1f}/5", (280, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Sensibilidade atual
            cv2.putText(display_frame, f"📈 Sensibilidade: FACAS={WEAPON_CONF_THRESHOLD}", (450, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Status do sistema
            status_text = "⏸️ PAUSADO" if paused else "✅ ATIVO - ALTA SENSIBILIDADE"
            status_color = (0, 255, 255) if paused else (0, 255, 0)
            cv2.putText(display_frame, status_text, (RESOLUTION[0] - 400, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Instruções
            cv2.putText(display_frame, "[P] Pausar/Continuar", (RESOLUTION[0] - 200, RESOLUTION[1] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "[Q] Sair", (RESOLUTION[0] - 200, RESOLUTION[1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if paused:
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (RESOLUTION[0], RESOLUTION[1]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
                cv2.putText(display_frame, "SISTEMA PAUSADO", 
                           (RESOLUTION[0]//2 - 150, RESOLUTION[1]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            
            # Calcular FPS
            if current_time - frame_time >= 1.0:
                fps = frame_id / (current_time - frame_time)
                frame_time = current_time
                frame_id = 0
            
            # Mostrar frame
            cv2.imshow("Sistema Avançado - Detecção de Facas e Armas", display_frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🎯 Saindo...")
                break
            elif key == ord('p'):
                paused = not paused
                status = "PAUSADO" if paused else "RETOMADO"
                print(f"⏸️ {status}")
            elif key == ord('s'):
                # Ajustar sensibilidade dinamicamente
                new_sens = input("Nova sensibilidade para facas (0.1-0.5): ")
                try:
                    new_val = float(new_sens)
                    if 0.1 <= new_val <= 0.5:
                        global WEAPON_CONF_THRESHOLD
                        WEAPON_CONF_THRESHOLD = new_val
                        print(f"✅ Sensibilidade ajustada para: {new_val}")
                    else:
                        print("❌ Valor deve estar entre 0.1 e 0.5")
                except:
                    print("❌ Valor inválido")
            elif key == ord('c'):
                alert_count = 0
                consecutive_detections = 0
                print("🔄 Contadores zerados")
    
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
    print("=== SISTEMA AVANÇADO DE DETECÇÃO DE FACAS E ARMAS ===")
    print("🔪 ESPECIALIZADO EM DETECÇÃO DE FACAS À DISTÂNCIA")
    print("1 - Baixar modelos especializados em facas")
    print("2 - Rodar detector AVANÇADO (MÁXIMA SENSIBILIDADE)")
    print("3 - Ajustar sensibilidade")
    print("4 - Sair")
    
    opcao = input("Escolha uma opção (1-4): ").strip()

    if opcao == "1":
        download_knife_model()
        input("\nPressione Enter para voltar ao menu...")
    elif opcao == "2":
        rodar_detector_avancado()
    elif opcao == "3":
        new_sens = input("Sensibilidade atual para facas (0.1-0.5): ")
        try:
            new_val = float(new_sens)
            if 0.1 <= new_val <= 0.5:
                WEAPON_CONF_THRESHOLD = new_val
                print(f"✅ Sensibilidade ajustada para: {new_val}")
            else:
                print("❌ Valor deve estar entre 0.1 e 0.5")
        except:
            print("❌ Valor inválido")
        input("\nPressione Enter para continuar...")
        rodar_detector_avancado()
    