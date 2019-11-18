import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

import numpy as np
import sys
import wave
import argparse

from deepspeech import Model

class Constants:
    # These constants control the beam search decoder

    # Beam width used in the CTC decoder when building candidate transcriptions
    BEAM_WIDTH = 500

    # The alpha hyperparameter of the CTC decoder. Language Model weight
    LM_ALPHA = 0.75

    # The beta hyperparameter of the CTC decoder. Word insertion bonus.
    LM_BETA = 1.85


    # These constants are tied to the shape of the graph used (changing them changes
    # the geometry of the first layer), so make sure you use the same constants that
    # were used during training

    # Number of MFCC features to use
    N_FEATURES = 26

    # Size of the context window used for producing timesteps in the input vector
    N_CONTEXT = 9

class _KnownPaths:
    def __init__(self):
        self.root = os.path.join(os.path.dirname(__file__), 'deepspeech_models')
        self.alphabet = os.path.join(self.root, 'alphabet.txt')
        self.language_model = os.path.join(self.root, 'lm.binary')
        self.tf_model = os.path.join(self.root, 'output_graph.pbmm')
        self.trie = os.path.join(self.root, 'trie')

KnownPaths = _KnownPaths()

def make_model():
    ds = Model(KnownPaths.tf_model, Constants.N_FEATURES, Constants.N_CONTEXT, KnownPaths.alphabet, Constants.BEAM_WIDTH)
    ds.enableDecoderWithLM(KnownPaths.alphabet, KnownPaths.language_model, KnownPaths.trie, Constants.LM_ALPHA, Constants.LM_BETA)
    return ds

def metadata_to_string(metadata):
    return ''.join(item.character for item in metadata.items)


def read_audio(audio_path):
    input_file = wave.open(audio_path, 'rb')
    sample_rate = input_file.getframerate()
    assert sample_rate == 16000, "Must have a 16 kHZ input"
    num_samples_in_clip = input_file.getnframes()
    audio = np.frombuffer(input_file.readframes(num_samples_in_clip), np.int16)
    clip_length = num_samples_in_clip / sample_rate
    return audio, sample_rate

def infer(model, audio, sample_rate = 16000):
    return model.stt(audio, sample_rate)

def parse_args():
    parser = argparse.ArgumentParser(description='Transcribes some speech')
    parser.add_argument('audio', help='The 16 kHZ input WAV file')
    return parser.parse_args()

def main():
    args = parse_args()
    audio, sample_rate = read_audio(args.audio)
    model = make_model()
    transcribed = infer(model, audio, sample_rate)
    print(transcribed)

if __name__ == '__main__':
    main()