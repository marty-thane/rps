from constants import *
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
from playsound import playsound
import random # remove if nn-based ai implemented
from threading import Thread

window_name = os.path.basename(__file__)
cam = cv2.VideoCapture(0) # open webcam connection
vision = load_model(f"{ASSET_DIR}/vision.keras")

log_buffer = []
gesture_buffer = []
score = (0,0)
selected_already = False

ai_gestures = {
    "rock": cv2.imread(f"{ASSET_DIR}/rock.jpg"),
    "paper": cv2.imread(f"{ASSET_DIR}/paper.jpg"),
    "scissors": cv2.imread(f"{ASSET_DIR}/scissors.jpg"),
        }

winning_combinations = {
        "rock": "scissors",
        "scissors": "paper",
        "paper": "rock",
        }

sounds = {
        "win": f"{ASSET_DIR}/win.wav",
        "lose": f"{ASSET_DIR}/lose.wav",
        "tie": f"{ASSET_DIR}/tie.wav",
        }

# prints to both stdout and window
def log(message):
    log_buffer.append(message)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {window_name}: {message}")

# prevents running out of memory by keeping the buffers small
def shrink_buffer(buffer, size):
    if len(buffer) > size:
        buffer[:] = buffer[-size:]

# prevents lagging by parallelizing audio
def play_audio(sound):
    audio_thread = Thread(target = playsound, args = (sound,))
    audio_thread.start()

# chooses winner, writes to log, plays sounds, returns new score difference
def make_results(player, ai):
    if winning_combinations[player] == ai:
        play_audio(sounds["win"])
        log("PLAYER WINS!")
        return (1,0)
    elif winning_combinations[ai] == player:
        play_audio(sounds["lose"])
        log("AI WINS!")
        return (0,1)
    else:
        play_audio(sounds["tie"])
        log("IT'S A TIE!")
        return (0,0)

log("GAME STARTED!")
log("waiting for input...")

while cam.isOpened():
    ret, img = cam.read()
    if not ret:
        continue # new loop if no data received

    if img.shape != (WH, WW, 3):
        img = cv2.resize(img, (WW, WH)) # scale if cam resolution unexpected
    img = cv2.flip(img, 1)

    out = img[BY:BY+BS, BX:BX+BS]
    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    out = cv2.resize(out, (DATA_SIZE, DATA_SIZE))
    out = np.expand_dims(out, axis = 0) / 255 # normalize values

    predictions = vision.predict(out, verbose = 0)
    predicted_class_index = np.argmax(predictions)
    predicted_class = CLASS[predicted_class_index]
    # prediction_certainty = predictions[0][predicted_class_index] * 100

    gesture_buffer.append(predicted_class)
    shrink_buffer(gesture_buffer, GESTURE_BUFFER_SIZE)

    # ugly messy gameplay logic, DO NOT TOUCH!
    if gesture_buffer[-1] != CLASS[0] and len(gesture_buffer) == GESTURE_BUFFER_SIZE:
        if len(set(gesture_buffer)) == 1:
            if not selected_already:
                player_choice = gesture_buffer[-1]
                log(f"PLAYER selected {player_choice.upper()}")
                ai_choice = random.choice(CLASS[1:])
                log(f"AI selected {ai_choice.upper()}")
                points = make_results(player_choice, ai_choice)
                score = (score[0]+points[0], score[1]+points[1])
                selected_already = True
            box_color = GREEN
        else:
            selected_already = False
            box_color = YELLOW
    else:
        ai_choice = None
        box_color = WHITE

    shrink_buffer(log_buffer, LOG_BUFFER_SIZE)

    cv2.rectangle(img, (BX,BY), (BX+BS,BY+BS), box_color, 4)
    cv2.rectangle(img, (WW-BX-BS,BY), (WW-BX,BY+BS), GRAY, -1) # fill for ai_gestures
    cv2.rectangle(img, (WW-BX-BS,BY), (WW-BX,BY+BS), WHITE, 4) # outline for ai_gestures
    if ai_choice:
        img[BY+2:BY+BS-2,WW-BX-BS+2:WW-BX-2] = ai_gestures[ai_choice] # 2s compensate for outline thickness
    cv2.putText(img, f"{score[0]}-{score[1]}", (20,WH-20), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 3)
    for i in range(len(log_buffer)):
        cv2.putText(img, log_buffer[-(i+1)], ((WW//2)-100, WH-20-i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    cv2.imshow(window_name, img)
    cv2.waitKey(FPS)

    # exit if window closed
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        cam.release()

cv2.destroyAllWindows()
