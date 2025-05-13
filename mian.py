import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# 初始化参数
wCam, hCam = 1280, 720  # 摄像头分辨率
frameR = 150  # 鼠标移动区域边界缩减量
smoothening = 7  # 移动平滑系数

# 初始化MediaPipe手部检测
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 获取屏幕分辨率
screen_w, screen_h = pyautogui.size()

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

plocX, plocY = 0, 0  # 前一点坐标
clocX, clocY = 0, 0  # 当前点坐标

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 获取食指关键点（第8号landmark）
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_tip.x * wCam), int(index_tip.y * hCam)

            # 在画面中绘制食指位置
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            # 将坐标转换为屏幕坐标
            #x = np.interp(cx, (frameR, wCam - frameR), (0, screen_w))
            x = screen_w - np.interp(cx, (frameR, wCam - frameR), (0, screen_w))
            y = np.interp(cy, (frameR, hCam - frameR), (0, screen_h))

            # 平滑移动处理
            clocX = plocX + (x - plocX) / smoothening
            clocY = plocY + (y - plocY) / smoothening

            # 移动鼠标
            pyautogui.moveTo(screen_w - clocX, clocY)
            plocX, plocY = clocX, clocY

            # 绘制操作区域框
            cv2.rectangle(img, (frameR, frameR),
                          (wCam - frameR, hCam - frameR),
                          (255, 0, 255), 2)

    # 显示画面
    cv2.imshow("Finger Mouse Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()