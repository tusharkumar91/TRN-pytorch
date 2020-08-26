# test the pre-trained model on a single video
# (working on it)
# Bolei Zhou and Alex Andonian

import os
import re
import cv2
import argparse
import functools
import subprocess
import numpy as np
from PIL import Image
#import moviepy.editor as mpy

import torchvision
import torch.nn.parallel
import torch.optim
from models import TSN
import transforms
from torch.nn import functional as F


def extract_frames(video_file, num_frames=8):
    try:
        os.makedirs(os.path.join(os.getcwd(), 'frames'))
    except OSError:
        pass

    output = subprocess.Popen(['ffmpeg', '-i', video_file],
                              stderr=subprocess.PIPE).communicate()
    # Search and parse 'Duration: 00:05:24.13,' from ffmpeg stderr.
    re_duration = re.compile('Duration: (.*?)\.')
    duration = re_duration.search(str(output[1])).groups()[0]

    seconds = functools.reduce(lambda x, y: x * 60 + y,
                               map(int, duration.split(':')))
    rate = num_frames / float(seconds)

    output = subprocess.Popen(['ffmpeg', '-i', video_file,
                               '-vf', 'fps={}'.format(rate),
                               '-vframes', str(num_frames),
                               '-loglevel', 'panic',
                               'frames/%d.jpg']).communicate()
    frame_paths = sorted([os.path.join('frames', frame)
                          for frame in os.listdir('frames')])

    frames = load_frames(frame_paths)
    subprocess.call(['rm', '-rf', 'frames'])
    return frames


def load_frames(frame_paths, num_frames=8):
    frames = [Image.open(frame).convert('RGB') for frame in frame_paths]
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise ValueError('Video must have at least {} frames'.format(num_frames))


def render_frames(frames, prediction):
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,
                    (1, int(height / 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames


# options
parser = argparse.ArgumentParser(description="test TRN on a single video")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--video_file', type=str, default=None)
group.add_argument('--frame_folder', type=str, default=None)
parser.add_argument('--modality', type=str, default='RGB',
                    choices=['RGB', 'Flow', 'RGBDiff'], )
parser.add_argument('--dataset', type=str, default='moments',
                    choices=['something', 'jester', 'moments', 'somethingv2'])
parser.add_argument('--rendered_output', type=str, default=None)
parser.add_argument('--arch', type=str, default="InceptionV3")
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--test_segments', type=int, default=8)
parser.add_argument('--img_feature_dim', type=int, default=256)
parser.add_argument('--consensus_type', type=str, default='TRNmultiscale')
parser.add_argument('--weights', type=str)

args = parser.parse_args()

# Get dataset categories.
categories_file = 'pretrain/{}_categories.txt'.format(args.dataset)
categories = [line.rstrip() for line in open(categories_file, 'r').readlines()]
num_class = len(categories)

args.arch = 'InceptionV3' if args.dataset == 'moments' else 'BNInception'

# Load model.
net = TSN(2,
          args.test_segments,
          args.modality,
          base_model=args.arch,
          consensus_type=args.consensus_type,
          img_feature_dim=args.img_feature_dim, print_spec=False)

checkpoint = torch.load(args.weights, map_location='cpu')
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
for key in ['consensus.fc_fusion_scales.6.3.bias', 'consensus.fc_fusion_scales.5.3.bias', 'consensus.fc_fusion_scales.4.3.bias',
            'consensus.fc_fusion_scales.3.3.bias', 'consensus.fc_fusion_scales.2.3.bias', 'consensus.fc_fusion_scales.1.3.bias',
            'consensus.fc_fusion_scales.0.3.bias', 'consensus.fc_fusion_scales.6.3.weight', 'consensus.fc_fusion_scales.5.3.weight',
            'consensus.fc_fusion_scales.4.3.weight', 'consensus.fc_fusion_scales.3.3.weight', 'consensus.fc_fusion_scales.2.3.weight',
            'consensus.fc_fusion_scales.1.3.weight', 'consensus.fc_fusion_scales.0.3.weight']:
    del base_dict[key]
#print(base_dict)
net.load_state_dict(base_dict, strict=False)

#print(net)
#exit(0)
net.eval()

# Initialize frame transforms.
transform = torchvision.transforms.Compose([
    transforms.GroupOverSample(net.input_size, net.scale_size),
    transforms.Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
    transforms.ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
    transforms.GroupNormalize(net.input_mean, net.input_std),
])

outputs = [0]*56
import glob
video_dir = '../../NymbleData/segments/*.mp4'
for video_file_name in sorted(glob.glob(video_dir)):
    
    video_file = video_file_name
    index = int(video_file_name.split('/')[-1].split('.')[0])
    print(video_file_name, index)
    # Obtain video frames
    if args.frame_folder is not None:
        print('Loading frames in {}'.format(args.frame_folder))
        import glob
        # Here, make sure after sorting the frame paths have the correct temporal order
        frame_paths = sorted(glob.glob(os.path.join(args.frame_folder, '*.jpg')))
        frames = load_frames(frame_paths)
    else:
        print('Extracting frames using ffmpeg...')
        frames = extract_frames(video_file, args.test_segments)
    
    
    # Make video prediction.
    data = transform(frames)
    input = data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0)
    
    with torch.no_grad():
        logits = net(input)
        h_x = torch.mean(F.sigmoid(logits), dim=0).data
        probs, idx = h_x.sort(0, True)
    print(h_x)
    # Output the prediction.
    video_name = args.frame_folder if args.frame_folder is not None else video_file
    print('RESULT ON ' + video_name)
    top5 = []
    exit(0)
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], categories[idx[i]]))
        top5.append((probs[i].item(), categories[idx[i]]))
    outputs[index] = top5
    print('---------')
    # print(outputs)
    exit(0)
# import pickle
# pickle.dump(outputs, open('somethingv1_1.pkl', 'wb'))
# Render output frames with prediction text.
#if args.rendered_output is not None:
#    prediction = categories[idx[0]]
#    rendered_frames = render_frames(frames, prediction)
#    clip = mpy.ImageSequenceClip(rendered_frames, fps=4)
#    clip.write_videofile(args.rendered_output)
