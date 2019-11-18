import os
import argparse
import pyaudio
import wave
import io

class Constants:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK_SIZE = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = 'recorded.wav'


def open_mic():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=Constants.FORMAT, rate=Constants.RATE, channels=Constants.CHANNELS, input=True, frames_per_buffer=Constants.CHUNK_SIZE)
    return audio, stream

def close_mic(audio, stream):
    stream.stop_stream()
    stream.close()
    audio.terminate()

def record_clip(audio, stream, clip_length_seconds = 5):
    recorded_stream = io.BytesIO()
    n_desired_samples = Constants.RATE * clip_length_seconds
    n_desired_chunks = n_desired_samples / Constants.CHUNK_SIZE
    for i in range(0, int(Constants.RATE / Constants.CHUNK_SIZE * clip_length_seconds)):
        data = stream.read(Constants.CHUNK_SIZE)
        recorded_stream.write(data)
    return recorded_stream

def write_clip(audio, stream, fname, recorded_stream: io.BytesIO):
    recorded_stream.seek(0)
    data = recorded_stream.read()
    waveFile = wave.open(fname, 'wb')
    waveFile.setnchannels(Constants.CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(Constants.FORMAT))
    waveFile.setframerate(Constants.RATE)
    waveFile.writeframes(data)
    waveFile.close()


def main():
    audio, stream = open_mic()
    print("Recording...")
    recorded_stream = record_clip(audio, stream, clip_length_seconds=Constants.RECORD_SECONDS)
    print("Done recording!")
    close_mic(audio,stream)
    write_clip(audio, stream, Constants.WAVE_OUTPUT_FILENAME, recorded_stream)
    
    

if __name__ == '__main__':
    main()