import os
import glob


with open('segments_info_v2.txt', 'w') as f:
    for dir_name in glob.glob('/home/tusharkurochester/data/yt_frames_add_v2/*'):
        if '.' in dir_name:
            continue
        else:
            images_number = len(glob.glob(dir_name + '/*.jpg'))
            f.write(dir_name + ' ' + str(images_number) + ' 0\n')
    for dir_name in glob.glob('/home/tusharkurochester/data/yt_frames_mix/*'):
        if '.' in dir_name:
            continue
        else:
            images_number = len(glob.glob(dir_name + '/*.jpg'))
            f.write(dir_name + ' ' + str(images_number) + ' 1\n')
print('Done')
                                
