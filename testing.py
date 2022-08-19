from ctypes import windll
from math import sqrt, pow
from utils import FOV, grabscreen, millisleep
from mss import mss
from simple_pid import PID
from colorama import Fore, Back, Style
from screenshot import WindowCapture
import numpy as np, cv2, pyautogui, keyboard, pyautogui, torch, time

# GLOBAL params
move_factor = { 'Counter-Strike: Global Offensive': 1.667 }.get('Counter-Strike: Global Offensive', 1)
wincap = WindowCapture('Counter-Strike: Global Offensive - Direct3D 9')
sct = mss()
side_width, side_height = 416, 416
confidence_threshold = 0.5
nms_threshold = 0.3
win_class_name = None
class_names = ['head', 'body']
total_classes = 1
CONFIG_FILE, WEIGHTS_FILE = ['C:\\Users\\jaybe\\OneDrive\\Documents\\YOLOv5 Arduino Aimbot\\weights\\yolov4-tiny.cfg', 'C:\\Users\\jaybe\\OneDrive\\Documents\\YOLOv5 Arduino Aimbot\\weights\\yolov4-tiny.weights']
COLORS = []
model, net = None, None
errors = 0
DPI_Var = [1]
ACTIVATION_RANGE = 250
Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]
originalBox = (int(Wd/2 - ACTIVATION_RANGE/2), int(Hd/2 - ACTIVATION_RANGE/2), int(Wd/2 + ACTIVATION_RANGE/2), int(Hd/2 + ACTIVATION_RANGE/2))
# END GLOBAL params

net = cv2.dnn.readNet(WEIGHTS_FILE, CONFIG_FILE)
# use cuda if available
if torch.cuda.is_available():
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print(Fore.GREEN + "[+] " + Style.RESET_ALL + "Using CUDA")
else:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print(Fore.GREEN + "[+] " + Style.RESET_ALL + "Using CPU")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(side_width, side_height), scale=1/255, swapRB=False)
print(Fore.GREEN + "[+] " + Style.RESET_ALL + "Model loaded")

for i in range(len(class_names)):
    COLORS.append(np.random.randint(256, size=3).tolist())

pidx = PID(0.4, 0.0, 0.06, setpoint=0, sample_time=0.006,)
pidy = PID(0.4, 0.0, 0.06, setpoint=0, sample_time=0.006,)
#pidx = PID(0.6, 0.0, 0.06, setpoint=0, sample_time=0.006,)
#pidy = PID(0.6, 0.0, 0.06, setpoint=0, sample_time=0.006,)
print(Fore.GREEN + "[+] " + Style.RESET_ALL + "PID controllers initialized")

while True:
    frames = np.array(wincap.get_screenshot())
    frames = cv2.cvtColor(src=frames, code=cv2.COLOR_BGR2RGB)
    
    try:
        if frames.any():
            frame_height, frame_width = frames.shape[:2]
        frame_height += 0
        frame_width += 0
    except (cv2.error, AttributeError, UnboundLocalError) as e:
        if errors < 2:
            print(str(e))
            errors += 1
            pass

    x0, y0, fire_pos, fire_close, fire_ok = 0, 0, 0, 0, 0
    classes, scores, boxes, = model.detect(frames, confidence_threshold, nms_threshold)
    threat_list = []

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = class_names[int(classid)]
        x, y, w, h = box
        cv2.rectangle(frames, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frames, label, (int(x + w / 2 - 4 * len(label)), int(y + h / 2 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if classid == total_classes:
            total_classes += 1
        
        h_factor = (0.5 if w >= h or (total_classes > 1 and classid == 0) else 0.25)
        dist = sqrt(pow(frame_width / 2 - (x + w / 2), 2) + pow(frame_height / 2 - (y + h * h_factor), 2))
        threat_var = -(pow(w * h, 1/2) / dist * score if dist else 9999)
        if classid == 0:
            threat_var *= 6
        threat_list.append([threat_var, box, classid])

        if len(threat_list):
            threat_list.sort(key=lambda x: x[0])
            x_threat, y_threat, w_threat, h_threat = threat_list[0][1]
            #fire_close = (1 if frames / w_threat <= 50 else 0)
            x0 = x_threat + (w_threat - frame_width) / 2
            y0 = y_threat + (h_threat - frame_height) / 2
            if abs(x0) <= 1/4 * w_threat and abs(y0) <= 2/5 * h_threat:
                fire_ok = 1
            if threat_list[0][2] == 0:
                fire_pos = 1
            elif h_threat > w_threat:
                y0 -= h_threat / 4
                fire_pos = 2
                if fire_close:
                    y0 -= h_threat / 8
                    fire_pos = 1
            xpos, ypos = x0 + frame_width / 2, y0 + frame_height / 2
            cv2.line(frames, (frame_width // 2, frame_height // 2), (int(xpos), int(ypos)), (0, 0, 255), 2)
            moveX, moveY = x0, y0
            moveX = FOV(moveX, 600) / DPI_Var[0] * move_factor
            moveY = FOV(moveY, 600) / DPI_Var[0] * move_factor
            pid_moveX = -pidx(moveX)
            pid_moveY = -pidy(moveY)
            #print("pid_moveX: {} pid_moveY: {}".format(pid_moveX, pid_moveY))
            print(Fore.GREEN + "[+] " + Style.RESET_ALL + "MoveX: {} MoveY: {}".format(moveX, moveY))
            if keyboard.is_pressed('r'):
                pyautogui.moveRel(round(pid_moveX, 3), round(pid_moveY, 3))

    cv2.imshow("Frame", frames)
    if cv2.waitKey(1) & 0xFF == ord('0'):
        break

cv2.destroyAllWindows()