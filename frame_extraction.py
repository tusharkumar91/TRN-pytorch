import os
import json
import glob
import sys
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle

def extract_video_frames(video_dir, frames_dir):
    obj = open('segment_mix_info.pkl', 'wb')
    info = []
    with open('/home/tusharkurochester/data/COIN.json', 'r') as f:
        coin_data = json.load(f)
    add_verbs = ['add', 'combine', 'add-to', 'pour']
    #mixing_verbs = ['mix', 'beat', 'mix-around', 'stir-with', 'whisk', 'stir', 'blend', 'mix-in', 'stir-in']
    mixing_verbs = ['beat', 'stir-with', 'whisk', 'stir', 'mix-in', 'stir-in', 'mix']


    mix_segments = {}
    add_segments = {}
    mix_segment_info = {}
    keys = list(coin_data['database'].keys())
    for key in keys:
        add_segments_list = []
        mix_segments_list = []
        if coin_data['database'][key]['class'].startswith('Make'):
            for ann in coin_data['database'][key]['annotation']:
                mix_verb_found = False
                for mix_verb in mixing_verbs:
                    if mix_verb in ann['label']:
                        mix_verb_found = True
                add_verb_found = False
                for add_verb in add_verbs:
                    if add_verb in ann['label']:
                        add_verb_found = True
                if mix_verb_found and not add_verb_found:
                    mix_segments_list.append(ann['segment'])
                if add_verb_found and not mix_verb_found:
                    add_segments_list.append(ann['segment'])
                if len(mix_segments_list) > 0:
                    mix_segments[key] = mix_segments_list
                if len(add_segments_list) > 0:
                    add_segments[key] = add_segments_list
    all_videos = set()
    for key in add_segments:
        all_videos.add(key)
    #for key in mix_segments:
    #    all_videos.add(key)

    print(len(all_videos))

    for video in tqdm(all_videos):
        video_file = glob.glob(os.path.join(video_dir, video + '*'))
        if len(video_file) > 0:
            for idx, ann in enumerate(add_segments[video]):
                start = np.ceil(ann[0])
                end = np.floor(ann[1])
                length = (end - start)
                #print(start, end)
                if end - start > 30:
                    continue
                os.makedirs(os.path.join(frames_dir, video+ '_' + str(idx)))
                os.system('ffmpeg -ss {} -t {} -i {} -q:v 2  -f image2 {}/%05d.jpg'.
                          format(start, length, video_file[0], os.path.join(frames_dir, video+ '_' + str(idx))))
                num_frames  = len(glob.glob(os.path.join(frames_dir, video+ '_' + str(idx), "*.jpg")))
                text_to_add = 'yt_frames_add_v2/' + video+ '_' + str(idx) + ' ' + str(num_frames) + ' 1'
                info.append(text_to_add)
            pickle.dump(info, obj)
    print(len(info))

if __name__ == '__main__':
    print('running')
    extract_video_frames('/home/tusharkurochester/data/yt_videos', '/home/tusharkurochester/data/yt_frames_add_v2')
                
        
    
