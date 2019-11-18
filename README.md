# Get started
Assuming you're using Anaconda & Python 3.7

You'll need Tensorflow (CPU), Deepspeech v0.5.0a8 (CPU), Pytorch (GPU), PyAudio, OpenCV2, Pyautogui

Should be as easy as:
```bash
pip3 install tensorflow deepspeech==0.5.0a8 opencv-python pyautogui
conda install pyaudio
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

You'll also need to download the language model weights from
https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/deepspeech-0.5.1-models.tar.gz

Extract & move the raw files to `deepspeech_models`


To run, first start the Pose detection server, then the speech transcription server, then the mouse control
```bash
# In first terminal
python pose_server.py --scale_factor 0.5
# In 2nd terminal
python socket_transcriber.py

# In main (3rd) terminal
python mouse_socket.py
```

To quit, alt-tab to the openCV window and press 'q', OR press ctrl+c on the speech server (when one dies all will die)

# Original README

This codebase was adapted from https://github.com/rwightman/posenet-pytorch.
It also uses Deepspeech from Mozilla (https://github.com/mozilla/DeepSpeech)

## PoseNet Pytorch

This repository contains a PyTorch implementation (multi-pose only) of the Google TensorFlow.js Posenet model.

This port is based on my Tensorflow Python (https://github.com/rwightman/posenet-python) conversion of the same model. An additional step of the algorithm was performed on the GPU in this implementation so it is faster and consumes less CPU (but more GPU). On a GTX 1080 Ti (or better) it can run over 130fps.

Further optimization is possible as the MobileNet base models have a throughput of 200-300 fps.

### Install

A suitable Python 3.x environment with a recent version of PyTorch is required. Development and testing was done with Python 3.7.1 and PyTorch 1.0 w/ CUDA10 from Conda.

If you want to use the webcam demo, a pip version of opencv (`pip install python-opencv=3.4.5.20`) is required instead of the conda version. Anaconda's default opencv does not include ffpmeg/VideoCapture support. The python bindings for OpenCV 4.0 currently have a broken impl of drawKeypoints so please force install a 3.4.x version.

A fresh conda Python 3.6/3.7 environment with the following installs should suffice: 
```
conda install -c pytorch pytorch cudatoolkit
pip install requests opencv-python==3.4.5.20
```

### Usage

There are three demo apps in the root that utilize the PoseNet model. They are very basic and could definitely be improved.

The first time these apps are run (or the library is used) model weights will be downloaded from the TensorFlow.js version and converted on the fly.

For all demos, the model can be specified with the '--model` argument by using its integer depth multiplier (50, 75, 100, 101). The default is the 101 model.

#### image_demo.py 

Image demo runs inference on an input folder of images and outputs those images with the keypoints and skeleton overlayed.

`python image_demo.py --model 101 --image_dir ./images --output_dir ./output`

A folder of suitable test images can be downloaded by first running the `get_test_images.py` script.

#### benchmark.py

A minimal performance benchmark based on image_demo. Images in `--image_dir` are pre-loaded and inference is run `--num_images` times with no drawing and no text output.

#### webcam_demo.py

The webcam demo uses OpenCV to capture images from a connected webcam. The result is overlayed with the keypoints and skeletons and rendered to the screen. The default args for the webcam_demo assume device_id=0 for the camera and that 1280x720 resolution is possible.

### Credits

The original model, weights, code, etc. was created by Google and can be found at https://github.com/tensorflow/tfjs-models/tree/master/posenet

This port and my work is in no way related to Google.

The Python conversion code that started me on my way was adapted from the CoreML port at https://github.com/infocom-tpo/PoseNet-CoreML

### TODO (someday, maybe)
* More stringent verification of correctness against the original implementation
* Performance improvements (especially edge loops in 'decode.py')
* OpenGL rendering/drawing
* Comment interfaces, tensor dimensions, etc
* Implement batch inference for image_demo
* Create a training routine and add models with more advanced CNN backbones

