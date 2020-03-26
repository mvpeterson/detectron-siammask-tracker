'''
Convert tracks in .csv file to '.npy' file containing tensor with shape (id, nframe, [x, y, w, h, v])

['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'] -> (id, nframe, [x, y, w, h, v])
'''

import pandas as pd
import numpy as np
from tracking_demo import _frame_from_video
import cv2
import os

VIDEO_INPUT = '/mnt/fileserver/shared/datasets/cameras/Odessa/Duke_on_the_left/fragments/child/child_set005_00:16:20-00:20:00.mp4'
DATA_FILE = '/mnt/fileserver/shared/datasets/cameras/Odessa/Duke_on_the_left/fragments/child/child_set005_00:16:20-00:20:00.csv'
OUTPUT_FILE = '/mnt/fileserver/shared/datasets/cameras/Odessa/Duke_on_the_left/fragments/child/child_set005_00:16:20-00:20:00.npy'

# VIDEO_INPUT = '/media/mvp/ssd2/datasets/JTA/seq_child/seq_2/child_seq_2_raw.mp4'
# DATA_FILE = '/media/mvp/ssd2/datasets/JTA/seq_child/seq_2/tracks.csv'
# OUTPUT_FILE = '/media/mvp/ssd2/datasets/JTA/seq_child/seq_2/tracks.npy'
#
# VIDEO_INPUT = '/media/mvp/ssd2/datasets/JTA/seq_child/seq_1/child_seq_1_raw.mp4'
# DATA_FILE = '/media/mvp/ssd2/datasets/JTA/seq_child/seq_1/tracks.csv'
# OUTPUT_FILE = '/media/mvp/ssd2/datasets/JTA/seq_child/seq_1/tracks.npy'

frame_idx = 0
df = pd.DataFrame(columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'])

video = cv2.VideoCapture(VIDEO_INPUT)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
basename = os.path.basename(VIDEO_INPUT)

df = pd.read_csv(DATA_FILE, delimiter=",",
                 names  = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'])
df.columns = ((df.columns.str).replace("^ ","")).str.replace(" $","")
# df = df.drop(df.columns[6:], axis=1)

nframes = int(df["FrameId"].max() + 1)

df_id = df["Id"].drop_duplicates().reset_index(drop=True).reset_index()

N = df_id.shape[0]
colors = []
for i in range(N):
    # r = random.randint(0, 32767) % 256
    # g = random.randint(0, 32767) % 256
    # b = 0 if (r + g > 255) else (255 - (r + g))
    colors.append(((30+200*i)%256, (140+160*i)%256, (220+70*i)%256))


tracks_by_obj = -1e8*np.ones([df_id.shape[0], nframes, 5])

frame_gen = _frame_from_video(video)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.8
lineType = 2

for frame in frame_gen:

    df_i = df[df["FrameId"]==(frame_idx+1)].reset_index(drop=True)

    for i in range(df_i.shape[0]):

        #    if (tracks_i[i, 0] >= 0) and tracks_i[i, 1] >= 0 and tracks_i[i, 2] >= 0 and tracks_i[i, 3] >= 0:
        a = df_i.loc[i, 'Id']
        id = df_id[df_id.loc[:,'Id'] == df_i.loc[i, 'Id']]['index'].values[0]
        x = int(df_i.loc[i, 'X'])
        y = int(df_i.loc[i, 'Y'])
        w = int(df_i.loc[i, 'Width'])
        h = int(df_i.loc[i, 'Height'])
        v = df_i.loc[i, 'Visibility']
        if v < 0:
            v=1
        tracks_by_obj[id, frame_idx, :] = np.asarray([x, y, w, h, v])

        color = colors[id]

        if v < 0.5:
            color = (255,255,255)

        cv2.putText(frame, '%d' % id, (x, y), font, fontScale, color, lineType)


        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    cv2.imshow('', frame)
    cv2.waitKey(10)

    frame_idx += 1

np.save(OUTPUT_FILE, tracks_by_obj)