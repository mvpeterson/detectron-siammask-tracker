'''
Convert ground truth .ods files to '.npy' file containing tensor with shape (id, nframe, [x, y, w, h, v])
'''

import ezodf
import numpy as np
import pandas as pd

data_dir = '/mnt/fileserver/shared/datasets/cameras/Odessa/Duke_on_the_left/experiments/detectron_and_siammask_center_tracking/tracks/'
data_file = data_dir + 'result_meeting_set004_00:59:45-00:59:45.mp4.csv'


ODS_TRACKS = './gt_tracks_example/Duke_Tracks.ods'
ODS_VISIBILITY  = './gt_tracks_example/Duke_Visibility.ods'

outfile = data_file[:-4] + '.npy'

doc = ezodf.opendoc(ODS_TRACKS)

print("Spreadsheet contains %d sheet(s)." % len(doc.sheets))
for sheet in doc.sheets:
    print("-"*40)
    print("   Sheet name : '%s'" % sheet.name)
    print("Size of Sheet : (rows=%d, cols=%d)" % (sheet.nrows(), sheet.ncols()) )


# convert the first sheet to a pandas.DataFrame
sheet = doc.sheets[0]
df_dict = {}
for i, row in enumerate(sheet.rows()):
    # row is a list of cells
    # assume the header is on the first row
    if i == 0:
        # columns as lists in a dictionary
        df_dict = {cell.value:[] for cell in row}
        # create index for the column headers
        col_index = {j:cell.value for j, cell in enumerate(row)}
        continue
    for j, cell in enumerate(row):
        # use header instead of column index
        df_dict[col_index[j]].append(cell.value)
# and convert to a DataFrame
df_id = pd.DataFrame(df_dict)
df_id = df_id.dropna(axis='index', how='all')
df_id = df_id.dropna(axis='columns', how='all')


def assign_visibility(ods_path, tracks):
    doc = ezodf.opendoc(ods_path)

    print("Spreadsheet contains %d sheet(s)." % len(doc.sheets))
    for sheet in doc.sheets:
        print("-" * 40)
        print("   Sheet name : '%s'" % sheet.name)
        print("Size of Sheet : (rows=%d, cols=%d)" % (sheet.nrows(), sheet.ncols()))

    # convert the first sheet to a pandas.DataFrame
    sheet = doc.sheets[0]
    df_dict = {}
    for i, row in enumerate(sheet.rows()):
        # row is a list of cells
        # assume the header is on the first row
        if i == 0:
            # columns as lists in a dictionary
            df_dict = {cell.value: [] for cell in row}
            # create index for the column headers
            col_index = {j: cell.value for j, cell in enumerate(row)}
            continue
        for j, cell in enumerate(row):
            # use header instead of column index
            df_dict[col_index[j]].append(cell.value)
    # and convert to a DataFrame
    df_vis = pd.DataFrame(df_dict)
    df_vis = df_vis.dropna(axis='index', how='all')
    df_vis = df_vis.dropna(axis='columns', how='all')

    trv = np.zeros([tracks.shape[0], tracks.shape[1], tracks.shape[2]+1])
    trv[:,:, :-1] = tracks

    for index, row in df_vis.iterrows():
        j = 2
        while (j < df_vis.shape[1]) and (row[j] is not None):
            frame_start = int(row[j].split('[')[1].split('-')[0]) - 1
            frame_end = int(row[j].split('-')[1].split(']')[0])
            trv[index,frame_start:frame_end,-1] = 1
            j += 1

    return trv



df = pd.read_csv(data_file, delimiter=",")
df.columns = ((df.columns.str).replace("^ ","")).str.replace(" $","")
df = df.drop(df.columns[6:], axis=1)

nframes = df["FrameId"].max() + 1

tracks_by_obj = -1e8*np.ones([df_id.shape[0], nframes, 4])

for index, row in df_id.iterrows():
    print("%i/%i"%(index, df_id.shape[0]))

    # if index < 22:
    #     continue

    j = 2
    frame_start = []
    frame_end = []
    id = []
    while (j < df_id.shape[1]) and (row[j] is not None):
        id.append( int(row[j].split('[')[0]) )
        frame_start.append( int(row[j].split('[')[1].split('-')[0])-1 )
        frame_end.append(int(row[j].split('-')[1].split(']')[0])-1)
        j += 1

        # i = frame_start[0]
        # while i <= frame_end[0]:
        #     dfi = df[ df['FrameId'] == i ]
        #     dfi = dfi[ dfi['Id'] == id ]
        #     tracks_by_obj[id, i, :] = np.asarray( [dfi['X'], dfi['Y'], dfi['Width'], dfi['Height'] ])[:, 0]
        #     i += 1

    for n in range(len(frame_start)):
        # print(n)
        i = frame_start[n]
        while i <= frame_end[n]:
            dfi = df[df['FrameId'] == i]
            dfi = dfi[dfi['Id'] == id[n]]
            # if dfi.shape[0] < 1:
            #     print(dfi.shape)
            tracks_by_obj[index, i, 0] = float(dfi['X'])
            tracks_by_obj[index, i, 1] = float(dfi['Y'])
            tracks_by_obj[index, i, 2] = float(dfi['Width'])
            tracks_by_obj[index, i, 3] = float(dfi['Height'])
                # np.asarray([dfi['X'], dfi['Y'], dfi['Width'], dfi['Height']])[:, 0]
            i += 1

        if n > 0:
            nmissed = frame_start[n] - frame_end[n-1] - 1
            if nmissed > 0:
                dx = (tracks_by_obj[index, frame_start[n], 0] - tracks_by_obj[index, frame_end[n-1], 0])/nmissed
                dy = (tracks_by_obj[index, frame_start[n], 1] - tracks_by_obj[index, frame_end[n - 1], 1])/nmissed
                dw = (tracks_by_obj[index, frame_start[n], 2] - tracks_by_obj[index, frame_end[n-1], 2])/nmissed
                dh = (tracks_by_obj[index, frame_start[n], 3] - tracks_by_obj[index, frame_end[n - 1], 3])/nmissed

                k = 1
                for i in range(frame_end[n-1] + 1, frame_start[n]):
                    x = tracks_by_obj[index, frame_end[n - 1], 0] + k*dx
                    y = tracks_by_obj[index, frame_end[n - 1], 1] + k*dy
                    w = tracks_by_obj[index, frame_end[n - 1], 2] + k*dw
                    h = tracks_by_obj[index, frame_end[n - 1], 3] + k*dh
                    tracks_by_obj[index, i, :] = np.asarray([x, y, w, h])

                    k+=1


tracks_by_obj = assign_visibility(ODS_VISIBILITY, tracks_by_obj)
np.save(outfile, tracks_by_obj)

print('')

