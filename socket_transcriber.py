from mic_utils import close_mic, open_mic, record_clip
from transcriber import make_model, infer
import socket
import numpy as np
import io
import threading
from collections import deque

class Constants:
    RECORD_SECONDS = 2

class Server():
    def __init__(self, transcribed_stream: io.StringIO):
        self.tstream = transcribed_stream
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        self.sock.bind(('localhost',38001))
        self.sock.listen(5)
    def send(self):
        # End with a newline
        self.tstream.write('\n')
        # Read the stream
        self.tstream.seek(0)
        contents = self.tstream.read()
        # Delete everything in it
        self.tstream.seek(0)
        self.tstream.truncate(0)
        self.connection.send(contents.encode())
    def start_listening(self):
        self.connection, self.address = self.sock.accept()
        print("Connection established with",self.address)
    def stop_listening(self):
        self.connection.close()
        self.sock.close()

def repeatedly_record(dq: deque, should_stop_event: threading.Event, clip_length):
    audio, stream = open_mic()
    while not should_stop_event.is_set():
        #print("Recording...")
        recorded_stream = record_clip(audio, stream, clip_length_seconds=Constants.RECORD_SECONDS)
        #print("Done recording!")
        recorded_stream.seek(0)
        audio_as_np = np.frombuffer(recorded_stream.read(), dtype=np.int16)
        dq.append(audio_as_np)
    close_mic(audio,stream)

def main():
    model = make_model()
    transcription_stream = io.StringIO()
    serv = Server(transcription_stream)
    
    should_stop = threading.Event()
    dq = deque()
    print("Awaiting connection!")
    serv.start_listening()

    mic_thread = threading.Thread(group=None, target=repeatedly_record, name='recorderloop', args=(dq, should_stop, Constants.RECORD_SECONDS))
    mic_thread.start()
    while True:
        try:
            if len(dq) > 0:
                audio_as_np = dq.popleft()
                transcribed = infer(model, audio_as_np)
                transcription_stream.write(transcribed)
                try:
                    serv.send()
                except:
                    print("Connection dead!")
                    break
                if 'exit' in transcribed:
                    break
        except KeyboardInterrupt:
            break
    print("Exiting!")
    serv.stop_listening()
    should_stop.set()
    

if __name__ == '__main__':
    main()