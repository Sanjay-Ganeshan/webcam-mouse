from mic_utils import close_mic, open_mic, record_clip
from transcriber import make_model, infer
import numpy as np

class Constants:
    RECORD_SECONDS = 5

def main():
    model = make_model()
    audio, stream = open_mic()
    while True:
        try:
            print("Recording...")
            recorded_stream = record_clip(audio, stream, clip_length_seconds=Constants.RECORD_SECONDS)
            print("Done recording!")
            recorded_stream.seek(0)
            audio_as_np = np.frombuffer(recorded_stream.read(), dtype=np.int16)
            transcribed = infer(model, audio_as_np)
            print(transcribed)
            if 'exit' in transcribed:
                break
        except KeyboardInterrupt:
            break
    print("Exiting!")
    close_mic(audio,stream)
    

if __name__ == '__main__':
    main()