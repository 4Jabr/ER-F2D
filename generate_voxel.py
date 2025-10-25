# eventscape doesn't have voxels folder, so generated them from the raw event data
import torch
import os
import numpy as np
import shutil
def events_to_voxel_grid(events, num_bins, width, height):
    
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(len(events.files) == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events['t'][-1]
    first_stamp = events['t'] [0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    ts= (num_bins - 1) * (events['t'] - first_stamp) / deltaT
    #ts = events['t']
    xs = events['x'].astype(int)
    ys = events['y'].astype(int)
    pols = events['p']
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid
    


dataset_folder = "/home/mdl/awb5924/ER-F2D/Test/Town05/" 
for i in sorted(os.listdir(dataset_folder)):
    print("======dataset_folder========",i)
    fi=dataset_folder+i+"/events/"
    if not os.path.isdir(fi+"voxels/"):
        os.mkdir(fi+"voxels/")
    for files in sorted(os.listdir(fi+"/data/")):
        if(files[-1]=='z'):
            #print(files)
            events=np.load(fi+"/data/"+files)
            voxel= events_to_voxel_grid(events,5,512,256)
        #print("==========voxel=========",voxel)
        #print("==========all zero??======",np.sum(voxel))
            np.save(fi+"voxels/"+files[:18]+'.npy',voxel)
        else:
            p=open(fi+"voxels/"+files,"w")
            shutil.copy(fi+"data/"+files,fi+"voxels/"+files)
            #time=np.load(files)
            #np.save(output_path+files,time)"""
