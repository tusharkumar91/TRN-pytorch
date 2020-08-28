import csv
import json
import numpy as np
from collections import defaultdict


with open('data/COIN.json', 'r') as f:
    coin_data = json.load(f)

segment = 0
add_verbs = ['add', 'combine', 'add-to', 'pour']
mixing_verbs = ['mix', 'beat', 'stir-with', 'whisk', 'stir', 'blend', 'mix-in', 'stir-in']

mix_segments = defaultdict(int)
add_segments = defaultdict(int)


keys = list(coin_data['database'].keys())
for key in keys:
    if coin_data['database'][key]['class'].startswith('Make'):
        for ann in coin_data['database'][key]['annotation']:
            mix_verb_found = False
            for mix_verb in mixing_verbs:
                if mix_verb in ann['label']:
                    mix_verb_found = True
                    break
            add_verb_found = False
            for add_verb in add_verbs:
                if add_verb in ann['label']:
                    add_verb_found = True
                    break
            if mix_verb_found and not add_verb_found:
                mix_segments[key] += 1
            if add_verb_found and not mix_verb_found:
                add_segments[key] += 1

all_videos = set()
for key in add_segments:
    all_videos.add(key)

print(len(all_videos))

import youtube_dl


import youtube_dl
from tqdm import tqdm
ydl_opts = {
    'outtmpl': 'data/yt_videos/%(id)s.%(ext)s',
    }
import os
count = 0
downloaded = os.listdir('data/yt_videos/')
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    for video in tqdm(all_videos):
        skip = False
        for downloads in downloaded:
            if video in downloads:
                skip = True
        if skip:
            print('skipping')
            continue
        else:
            try:
                ydl.download(['https://www.youtube.com/watch?v=' + video])
            except:
                continue
    
print(count)
