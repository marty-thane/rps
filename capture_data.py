from constants import *
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
from pathlib import Path

# variables (AKA stuff that changes while looping)
current_class = CLASS[0]
capturing = False

window_name = os.path.basename(__file__)
cam = cv2.VideoCapture(0) # open webcam connection

class_keys = [ c[0] for c in CLASS ] # get first letter of each class, used for keyboard shortcuts

# mkdir for classes (if not already made)
for c in CLASS:
    Path(f"{DATA_DIR}/{c}").mkdir(parents = True, exist_ok = True)

while cam.isOpened():
    ret, img = cam.read()
    if not ret:
        continue # new loop if no data received

    if img.shape != (WH, WW, 3):
        img = cv2.resize(img, (WW, WH)) # scale if cam resolution unexpected
    img = cv2.flip(img, 1)

    current_sample = len(os.listdir(f"{DATA_DIR}/{current_class}"))

    if capturing:
        out = img[BY:BY+BS, BX:BX+BS]
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.resize(out, (DATA_SIZE, DATA_SIZE))
        cv2.imwrite(f"{DATA_DIR}/{current_class}/{current_sample}.png", out)

    cv2.rectangle(img, (BX,BY), (BX+BS,BY+BS), RED if capturing else WHITE, 4)
    cv2.putText(img, f"class: {current_class}, sample: {current_sample}", (BX, BY+BS+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
    cv2.putText(img, f"{class_keys} = switch class, SPACEBAR = toggle capture", (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    cv2.imshow(window_name, img)

    # check for user input
    key_pressed = cv2.waitKey(FPS)
    for c in class_keys:
        if key_pressed == ord(c): # r/p/s/e
            current_class = CLASS[class_keys.index(c)]
    if key_pressed == ord(" "): # spacebar
        capturing = not capturing

    # exit if window closed
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        cam.release()

cv2.destroyAllWindows()
