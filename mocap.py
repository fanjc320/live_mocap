#####################################################################################
# Single View Human Motion Capture, Based on Mediapipe & OpenCV & PyTorch
# 
# Author: Ruicheng Wang
# License: Apache License 2.0
#####################################################################################
import os
import shutil
import argparse
import pickle
import subprocess

import numpy as np
import cv2
import torch
from tqdm import tqdm

from body_keypoint_track import BodyKeypointTrack, show_annotation
from skeleton_ik_solver import SkeletonIKSolver


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--blend', type=str, help='Path to rigged model Blender file. eg. c:\\tmp\\model.blend', default="model.blend")
    parser.add_argument('--blend', type=str, help='Path to rigged model Blender file. eg. c:\\tmp\\model.blend', default="D:\\projs_rm\\mediapipe\\live_mocap\\assets\\mixamo.blend")
    parser.add_argument('--video', type=str, help='Path to video file. eg. c:\\tmp\\video.mp4',default="C:\\Users\\AYA\\Videos\\bowen-normal.mp4")
    parser.add_argument('--track_hands', action='store_true', help='Enable hand tracking', default=False)

    args = parser.parse_args()
    FOV = np.pi / 3

    # Call blender to export skeleton
    os.makedirs('tmp', exist_ok=True)
    print("Export skeleton...")
    if os.path.exists('tmp/skeleton'):
        shutil.rmtree('tmp/skeleton')
    os.system(f"blender {args.blend} --background --python export_skeleton.py")
    if not os.path.exists('tmp/skeleton'):
        raise Exception("Skeleton export failed")

    print("main aaaaaaaaaa...")

    # Open the video capture
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("main bbbbbbbbb failed!!!!!!!!!!!!")
        raise Exception("Video capture failed")
    print("main aaaaaaaaaa...1111111")
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("main aaaaaaaaaa...2222222")
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print("main aaaaaaaaaa...3333333")
    tmp = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("main aaaaaaaaaa...444444"+str(tmp))
    # 这里会停止运行，出现这种原因是，视频中包含了损坏的或不能被opencv解码的帧，opencv就会跳过这些帧，因此造成通过属性（CAP_PROP_FRAME_COUNT）和实际读取的帧数不一样。所以，如果想要获取能够读取的帧数，首先遍历一遍视频。
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("main bbbbbbbbb..."+str(total_frames))
    # Initialize the body keypoint tracker
    body_keypoint_track = BodyKeypointTrack(
        im_width=frame_width,
        im_height=frame_height,
        fov=FOV,
        frame_rate=frame_rate,
        track_hands=args.track_hands,
        smooth_range=10 * (1 / frame_rate),
        smooth_range_barycenter=30 * (1 / frame_rate),
    )
    print("main cccccc 11111")
    # Initialize the skeleton IK solver
    skeleton_ik_solver = SkeletonIKSolver(
        model_path='tmp/skeleton',
        track_hands=args.track_hands,
        smooth_range=15 * (1 / frame_rate),
    )
    print("main cccccc 22222")
    bone_euler_sequence, scale_sequence, location_sequence = [], [], []
    print("main cccccc 22222 111111")
    frame_t = 0.0
    print("main cccccc 22222 222222")
    frame_i = 0
    print("main cccccc 22222 33333")
    print("main cccccc 33333 total_frames"+str(total_frames))
    #bar = tqdm(total=total_frames, desc='Running...')
    print("main cccccc..............end")

    while cap.isOpened():
        # Get the frame image
        ret, frame = cap.read()
        if not ret:
            print("main read fail ret:"+str(ret))
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #ndarray(960,540,3)

        # Get the body keypoints
        body_keypoint_track.track(frame, frame_t)
        kpts3d, valid = body_keypoint_track.get_smoothed_3d_keypoints(frame_t)

        # Solve the skeleton IK
        skeleton_ik_solver.fit(torch.from_numpy(kpts3d).float(), torch.from_numpy(valid).bool(), frame_t)# fit((33,3),

        # Get the skeleton pose
        bone_euler = skeleton_ik_solver.get_smoothed_bone_euler(frame_t) # Tensor(50,3)
        location = skeleton_ik_solver.get_smoothed_location(frame_t) # Tensor(3,)
        scale = skeleton_ik_solver.get_scale() # tensor(1.09)

        bone_euler_sequence.append(bone_euler)
        location_sequence.append(location)
        scale_sequence.append(skeleton_ik_solver.get_scale())

        # Show keypoints tracking result
        show_annotation(frame, kpts3d, valid, body_keypoint_track.K)
        if cv2.waitKey(1) == 27:
            print('Cancelled by user. Exit.')
            exit()

        frame_i += 1
        frame_t += 1.0 / frame_rate
        # bar.update(1)

    # Save animation result
    print("Save animation result...")
    with open('tmp/bone_animation_data.pkl', 'wb') as fp:
        pickle.dump({
            'fov': FOV,
            'frame_rate': frame_rate,
            'bone_names': skeleton_ik_solver.optimizable_bones,
            'bone_euler_sequence': bone_euler_sequence,
            'location_sequence': location_sequence,
            'scale': np.mean(scale_sequence),
            'all_bone_names': skeleton_ik_solver.all_bone_names
        }, fp)

    # Open blender and apply the animation
    print("Open blender and apply animation...")

    proc = subprocess.Popen(f"blender {args.blend} --python apply_animation.py")
    proc.wait()


if __name__ == '__main__':
    main()