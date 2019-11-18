# Modified by Julie, based on original "webcam_demo.py"

import torch
import cv2
import time
import argparse
from easydict import EasyDict
import socket

import posenet

def parse_camera(s):
    try:
        return int(s)
    except:
        return s

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=parse_camera, default=0)
parser.add_argument('--cam_width', type=int, default=640)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--draw', action='store_true', default=False)
args = parser.parse_args()

class Skeleton():
    def __init__(self, w, h):
        self.nkps = len(posenet.constants.PART_NAMES)
        self.kp_names = posenet.constants.PART_NAMES
        self.scores = [0.0] * self.nkps
        self.positions = [(0.0, 0.0)] * self.nkps
        self.width, self.height = (w, h)
    def update(self, keypoints, scores, minscore = 0.5):
        for i in range(self.nkps):
            #if scores[i] >= minscore:
            self.scores[i] = scores[i]
            self.positions[i] = tuple(keypoints[i])
    def serialize(self):
        s = ','.join(['%.2f %.3f %.3f' % (self.scores[i], (1 - self.positions[i][1] / self.width) - 0.5, (1 - self.positions[i][0] / self.height) - 0.5) for i in range(self.nkps)])
        return s
    def deserialize(self, s):
        parts = s.split(',')
        for ix, each_part in enumerate(parts):
            (score, x, y) = map(float, each_part.split(' '))
            self.scores[ix] = score
            self.positions[ix] = (x, y)
        

        
class Server():
    def __init__(self, skeleton):
        self.skeleton = skeleton
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        self.sock.bind(('localhost',38000))
        self.sock.listen(5)
    def send(self):
        self.connection.send(self.skeleton.serialize().encode())
        self.connection.send(b'\n')
    def start_listening(self):
        self.connection, self.address = self.sock.accept()
        print("Connection established with",self.address)
    def stop_listening(self):
        self.connection.close()
        self.sock.close()


def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    theskel = Skeleton(args.cam_width, args.cam_height)
    serv = Server(theskel)
    serv.start_listening()

    cap = cv2.VideoCapture(args.cam_id)
    cv2.namedWindow('recorder')
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)
    

    should_continue = True

    start = time.time()
    frame_count = 0
    while should_continue:
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

        keypoint_coords *= output_scale

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        
        
        theskel.update(keypoint_coords[0], keypoint_scores[0])
        serv.send()
        
        #center = tuple(map(int,keypoint_coords[0][posenet.constants.PART_NAMES.index('nose')]))[::-1]
        #print(keypoint_scores[0][0])
        #cv2.circle(display_image, center, 10, (0,0,255), thickness=10)

        if args.draw:
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            cv2.imshow('posenet', overlay_image)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            should_continue = False

    serv.stop_listening()
    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()