import time
import json
import cv2
import mss
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# If you want SERIAL ESP comms, install: pip install pyserial
# and set USE_ESP_SERIAL=True + ESP_PORT/ESP_BAUD below.
try:
    import serial  # type: ignore
except Exception:
    serial = None

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "multi_classifier.pth"  # saved model trained on 4 classes

# Original model classes (must match training order!)
RAW_CLASS_NAMES = ["fire", "no_fire", "no_shock", "shock"]

# Merged final classes:
# - fire
# - shock
# - nominal  (no_fire + no_shock)
FINAL_CLASS_NAMES = ["fire", "shock", "nominal"]

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Screen capture config
AUTO_DETECT_PRIMARY_MONITOR = True  # recommended
FALLBACK_MONITOR = {"top": 0, "left": 0, "width": 1920, "height": 1080}

# ESP comms config (default: SERIAL skeleton that won't crash if pyserial missing)
USE_ESP_SERIAL = False  # set True if using serial
ESP_PORT = "COM5"       # e.g. "COM5" on Windows, "/dev/ttyUSB0" on Linux
ESP_BAUD = 115200


# -----------------------
# LOAD MODEL (4-class output)
# -----------------------
num_raw_classes = len(RAW_CLASS_NAMES)

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, num_raw_classes),
)

state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model = model.to(DEVICE)
model.eval()

print("✅ Model loaded from:", MODEL_PATH)
print("Raw model classes:", RAW_CLASS_NAMES)
print("Final merged classes:", FINAL_CLASS_NAMES)
print("Device:", DEVICE)

# -----------------------
# TRANSFORMS
# -----------------------
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ]
)


# -----------------------
# CLASS MERGING (raw -> final)
# -----------------------
def merge_raw_probs_to_final(raw_probs: torch.Tensor):
    """
    raw_probs: tensor of shape [4] in RAW_CLASS_NAMES order:
      [fire, no_fire, no_shock, shock]

    returns:
      final_probs_np: np.array shape [3] in FINAL_CLASS_NAMES order:
        [fire, shock, nominal]
      final_pred_index: int
      final_pred_label: str
    """
    idx_fire = RAW_CLASS_NAMES.index("fire")
    idx_shock = RAW_CLASS_NAMES.index("shock")
    idx_no_fire = RAW_CLASS_NAMES.index("no_fire")
    idx_no_shock = RAW_CLASS_NAMES.index("no_shock")

    fire_p = float(raw_probs[idx_fire].item())
    shock_p = float(raw_probs[idx_shock].item())
    nominal_p = float(raw_probs[idx_no_fire].item() + raw_probs[idx_no_shock].item())

    final_probs_np = np.array([fire_p, shock_p, nominal_p], dtype=np.float32)
    final_pred_index = int(np.argmax(final_probs_np))
    final_pred_label = FINAL_CLASS_NAMES[final_pred_index]
    return final_probs_np, final_pred_index, final_pred_label


# -----------------------
# PREDICTION FUNCTION
# -----------------------
def predict_frame(frame_bgr: np.ndarray):
    """
    Returns:
      final_pred_label: 'fire' | 'shock' | 'nominal'
      final_pred_index: 0..2
      final_probs: list of tuples [('fire', 12.3), ('shock', 4.5), ('nominal', 83.2)]   # percentages
      final_probs_np: np.array shape [3] values in [0..1]
      raw_probs: list of tuples [('fire', ...), ('no_fire', ...), ('no_shock', ...), ('shock', ...)] # percentages
    """
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        raw_probs_t = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()  # shape [4]

    raw_probs = [
        (RAW_CLASS_NAMES[i], round(float(raw_probs_t[i]) * 100, 1))
        for i in range(num_raw_classes)
    ]

    final_probs_np, final_pred_index, final_pred_label = merge_raw_probs_to_final(raw_probs_t)
    final_probs = [
        (FINAL_CLASS_NAMES[i], round(float(final_probs_np[i]) * 100, 1))
        for i in range(len(FINAL_CLASS_NAMES))
    ]

    return final_pred_label, final_pred_index, final_probs, final_probs_np, raw_probs


# -----------------------
# ESP COMMUNICATION (implemented for SERIAL by default; safe no-op otherwise)
# -----------------------
class ESPSerialClient:
    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self.ser = None

    def connect(self) -> bool:
        if serial is None:
            print("[ESP] pyserial not installed. Run: pip install pyserial")
            return False
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
            time.sleep(1.0)  # give ESP time to reset on serial open
            print(f"[ESP] Connected over serial: {self.port} @ {self.baud}")
            return True
        except Exception as e:
            print(f"[ESP] Serial connect failed: {e}")
            self.ser = None
            return False

    def send_json_line(self, payload: dict) -> bool:
        if self.ser is None:
            return False
        try:
            line = (json.dumps(payload) + "\n").encode("utf-8")
            self.ser.write(line)
            return True
        except Exception as e:
            print(f"[ESP] Serial send failed: {e}")
            return False

    def close(self):
        try:
            if self.ser is not None:
                self.ser.close()
        finally:
            self.ser = None


esp_client = ESPSerialClient(ESP_PORT, ESP_BAUD) if USE_ESP_SERIAL else None
if esp_client is not None:
    esp_client.connect()


def send_action_to_esp(action: str, confidence: float, meta: dict | None = None) -> bool:
    """
    Sends an action to ESP.
    - If USE_ESP_SERIAL=True: sends JSON line over serial
    - Otherwise: prints payload (no-op placeholder)
    """
    payload = {
        "action": action,                       # "FIRE" | "SHOCK" | "NOMINAL"
        "confidence": round(float(confidence), 4),
        "ts": round(time.time(), 3),
        "meta": meta or {},
    }

    if esp_client is not None:
        return esp_client.send_json_line(payload)

    # Fallback / placeholder
    print("[ESP] (not configured) would send:", payload)
    return True


# -----------------------
# TRIGGERING LOGIC (debounce / threshold)
# -----------------------
class ESPTriggerState:
    def __init__(self):
        self.last_label: str | None = None
        self.same_count: int = 0
        self.last_sent_ts: float = 0.0


def maybe_trigger_esp(
    state: ESPTriggerState,
    final_pred_label: str,
    final_probs_np: np.ndarray,
    *,
    min_confidence: float = 0.70,
    stable_frames: int = 3,
    cooldown_sec: float = 0.5,
):
    pred_index = FINAL_CLASS_NAMES.index(final_pred_label)
    confidence = float(final_probs_np[pred_index])
    now = time.time()

    # stability counter
    if state.last_label == final_pred_label:
        state.same_count += 1
    else:
        state.last_label = final_pred_label
        state.same_count = 1

    # conditions
    if confidence < min_confidence:
        return
    if state.same_count < stable_frames:
        return
    if (now - state.last_sent_ts) < cooldown_sec:
        return

    action_map = {"fire": "FIRE", "shock": "SHOCK", "nominal": "NOMINAL"}
    action = action_map[final_pred_label]

    ok = send_action_to_esp(action=action, confidence=confidence, meta={"label": final_pred_label})
    if ok:
        state.last_sent_ts = now


# -----------------------
# SCREEN CAPTURE SETUP (main screen)
# -----------------------
sct = mss.mss()
if AUTO_DETECT_PRIMARY_MONITOR:
    # mss.monitors[1] is typically the primary monitor (index 0 is "all monitors")
    monitor = dict(sct.monitors[1])
else:
    monitor = FALLBACK_MONITOR

print("Capture monitor:", monitor)

# -----------------------
# DISPLAY WINDOW SETUP
# -----------------------
window_name = "Damage Detector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 420, 160)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, 0)

print("Press 'q' to quit")

# -----------------------
# MAIN LOOP
# -----------------------
esp_state = ESPTriggerState()

try:
    while True:
        sct_img = sct.grab(monitor)
        frame_bgra = np.array(sct_img)
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        final_pred_label, final_pred_index, final_probs, final_probs_np, raw_probs = predict_frame(frame_bgr)

        # Trigger ESP actions (debounced)
        maybe_trigger_esp(
            esp_state,
            final_pred_label,
            final_probs_np,
            min_confidence=0.70,
            stable_frames=3,
            cooldown_sec=0.5,
        )

        # Display
        h, w = 160, 420
        display_img = np.zeros((h, w, 3), dtype=np.uint8)

        cv2.putText(
            display_img,
            f"Prediction: {final_pred_label}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        y_offset = 60
        for i, (cls, prob) in enumerate(final_probs):
            color = (0, 0, 255) if i == final_pred_index else (0, 255, 0)
            cv2.putText(
                display_img,
                f"{cls}: {prob}%",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            y_offset += 30

        cv2.imshow(window_name, display_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.03)
finally:
    cv2.destroyAllWindows()
    if esp_client is not None:
        esp_client.close()