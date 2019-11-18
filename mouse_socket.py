# Written by Julie

import socket
import pyautogui
import io
import threading
import time

from pose_server import Skeleton


def make_client():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    sock.connect(('localhost', 38000))
    return sock

def make_speech_client():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    sock.connect(('localhost', 38001))
    return sock

def socket_loop(sock, on_recieve_hook, should_stop_event: threading.Event):
    stream = io.StringIO()
    read_offset = stream.tell()
    n_newlines = 0
    while True:
        if should_stop_event.is_set():
            break
        recieved_message = sock.recv(1024)
        if len(recieved_message) == 0:
            # end of stream
            break
        recieved_message = recieved_message.decode()
        if '\n' in recieved_message:
            n_newlines += 1
        stream.write(recieved_message)
        if n_newlines > 0:
            while n_newlines > 0:
                stream.seek(read_offset)
                line = stream.readline()
                on_recieve_hook(line)
                n_newlines -= 1
            read_offset = stream.tell()
            stream.seek(0, 2)
    sock.close()
    stream.close()
    should_stop_event.set()

def make_hook(skel: Skeleton):
    def on_recieve_hook(line):
        skel.deserialize(line)
    return on_recieve_hook

def make_speech_hook(spoken_words: io.StringIO):
    def on_recieve_hook(line):
        spoken_words.write(line)
    return on_recieve_hook

def reclamp(value, prev_min, prev_max, new_min, new_max):
    value = min(max(value, prev_min), prev_max)
    # 0 to 1
    normalized = (value - prev_min) / (prev_max - prev_min)
    # [0..range] -> [new min .. new max]
    denorm = (normalized * (new_max - new_min)) - new_min
    outp = min(max(denorm, new_min), new_max)
    return outp

def read_and_clear(buf: io.StringIO):
    buf.seek(0)
    contents = buf.read()
    buf.seek(0)
    buf.truncate(0)
    return contents


def main():
    should_stop = threading.Event()
    # Pose socket loop
    skel = Skeleton(1.0, 1.0)
    pose_sock = make_client()
    pose_hook = make_hook(skel)
    pose_thread = threading.Thread(None, socket_loop, name="poseLoop", args=(pose_sock, pose_hook, should_stop))
    pose_thread.start()
    
    # Voice socket loop
    transcribed = io.StringIO()
    speech_sock = make_speech_client()
    speech_hook = make_speech_hook(transcribed)
    speech_thread = threading.Thread(None, socket_loop, name="speechLoop", args=(speech_sock, speech_hook, should_stop))
    speech_thread.start()

    (screen_w, screen_h) = pyautogui.size()
    ''' 
    One of:
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
    '''
    tracking_point = "nose"
    rh_ix = skel.kp_names.index(tracking_point)
    while True:
        if should_stop.is_set():
            break
        posx, posy, score = (*skel.positions[rh_ix], skel.scores[rh_ix])
        if score > 0.5:
            mousex = int(reclamp(posx, -0.5, 0.5, 0, screen_w))
            # down = -0.5 = screen_h
            mousey = int(screen_h - reclamp(posy, -0.5, 0.5, 0, screen_h))
            pyautogui.moveTo(mousex, mousey, duration=0.0)
            #print(mousex, mousey)
        said = read_and_clear(transcribed)
        if len(said) > 1:
            said = said.lower()
            print("Heard %s" % said)
            if 'click' in said:
                pyautogui.click()
                print("Clicking!")
            elif 'down' in said:
                pyautogui.mouseDown(button='left', duration=5.0)
                print("Mouse down")
            elif 'up' in said:
                pyautogui.mouseUp(button='left')
                print("Mouse up")

    should_stop.set()


if __name__ == '__main__':
    main()