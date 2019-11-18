import os
import argparse
import pyaudio
import wave

class Constants:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = 'recorded.wav'


def main():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=Constants.FORMAT, rate=Constants.RATE, channels=Constants.CHANNELS, input=True, frames_per_buffer=Constants.CHUNK)
    print("Recording...")
    frames = []
    
    for i in range(0, int(Constants.RATE / Constants.CHUNK * Constants.RECORD_SECONDS)):
        data = stream.read(Constants.CHUNK)
        frames.append(data)

    print("Done recording!")

    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    waveFile = wave.open(Constants.WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(Constants.CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(Constants.FORMAT))
    waveFile.setframerate(Constants.RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

if __name__ == '__main__':
    main()