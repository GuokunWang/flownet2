#!/usr/bin/env python2.7

from __future__ import print_function


import os, sys, numpy as np
sys.path.insert(0,'../python/')
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil,sqrt
import cPickle as pickle

def calculate_error(loc_p, loc_g):
    cx_p = loc_p[0] + loc_p[2] / 2
    cy_p = loc_p[1] + loc_p[3] / 2
    cx_g = loc_g[0] + loc_g[2] / 2
    cy_g = loc_g[1] + loc_g[3] / 2
    error = sqrt((cx_p - cx_g)**2 + (cy_p -cy_g)**2)
    print(error)
    return error

evaluate = True
use_key_points = False
use_central_points = True
use_region_mean = False
parser = argparse.ArgumentParser()
parser.add_argument('caffemodel', help='path to model')
parser.add_argument('deployproto', help='path to deploy prototxt template')

args = parser.parse_args()

if (not os.path.exists(args.caffemodel)):
    raise BaseException('caffemodel does not exist: ' + args.caffemodel)
if (not os.path.exists(args.deployproto)):
    raise BaseException('deploy-proto does not exist: ' + args.deployproto)
#seq_path_prefix = "/home/sensetime/Downloads/"
#seq_path_prefix = "/media/sensetime/HardDisk/DataSet/MOT/2DMOT2015/train/"

#seq_path_prefix = "/mnt/lustre/wangguokun/dataset/MOT/2DMOT2015/train/"
#seq_path_prefix = "/media/sensetime/HardDisk/ataset/MOT/2DMOT2015/train"
#seq_name = 'bottle/'
#seq_name = 'frames/'
#seq_name = "TUD-Campus/"
#seq_path_prefix = "/media/sensetime/HardDisk/DataSet/MOT/2DMOT2015/train/

#seq_name = "PETS09-S2L1/"
#start_index = 1
#zero_numbers = 6
flow_gap = 2
#resize_ratio = 0.5 
#zero_numbers = 3

seq_path_prefix = "/media/sensetime/HardDisk/DataSet/MOT/2DMOT2015/train/"
seq = ['ADL-Rundle-8/', 'ADL-Rundle-6/', 'ETH-Bahnhof/', 'Venice-2/', 'TUD-Stadtmitte/', 'TUD-Campus/' ]
start = [1, 60, 210, 200, 1, 1]
seq_lengths = [50,50,50,50,50,50]
zero_numbers = 6

'''
seq_path_prefix = "/media/sensetime/HardDisk/DataSet/Test/"
seq = ['eyeglasses/', 'getout/', 'keyboard/', 'machine/', 'starbucks/', 'mouse/']
seq_lengths = [100, 200, 150, 150, 150, 550]
start = [1, 1, 1, 1, 1, 1]
zero_numbers = 3
'''
#img_path = seq_path_prefix + seq_name  + str(start_index).zfill(zero_numbers) + '.jpg'
num_blobs = 2
f_result = open('/home/sensetime/Documents/result.txt', 'w')
for seq_iter in range(len(seq)):
    seq_name = seq[seq_iter]
    start_index = start[seq_iter]
    '''
    '''


# read gt.txt
#with open(seq_path_prefix + seq_name + 'gt/gt.txt', 'r') as f:

#with open(seq_path_prefix + seq_name + 'gt/region.txt', 'r') as f:
 #   data = f.readlines()
#current_position = [300,274]
#seq_path_prefix = "/home/sensetime/Downloads/"
#start_index = 1
#seq_path_prefix = "/media/sensetime/HardDisk/DataSet/MOT/2DMOT2015/test/"
#seq_path_prefix = "/media/sensetime/HardDisk/DataSet/Test/"
#seq_name = "eyeglasses/"
    img_path = seq_path_prefix + seq_name + 'img1/' + str(start_index).zfill(zero_numbers) + '.jpg'
    if evaluate:
        with open(seq_path_prefix + seq_name + 'gt/gt.txt', 'r') as f:
            data = f.readlines()
        gt = {}
        for line in data:
            ele = line.split(',')
            ele[1] = int(ele[1])
            ele[0] = int(ele[0])
            ele[2:] = [float(i) for i in ele[2:]]
            if ele[1] not in gt.keys():
                gt[ele[1]] = {}
                gt[ele[1]][ele[0]] = ele[2:6]
            else:
                gt[ele[1]][ele[0]] = ele[2:6]
        f.close()
    with open(seq_path_prefix + seq_name + 'gt/region.txt', 'r') as f:
        data = f.readlines()
    output = {}
    current_position ={}
    for line in data:
        ele = line.split(',')
        print(ele[0])
        ele[0] = int(ele[0])
        ele[1:] = [float(i) for i in ele[1:]]
        if ele[0] not in output.keys():
            output[ele[0]] = {}
            output[ele[0]][start_index] = ele[1:]
            current_position[ele[0]] = [ele[1] + ele[3] /2, ele[2] + ele[4] / 2]
        else:
            output[ele[0]][start_index] = ele[1:]
    f.close()

    input_data = []
    img0 = cv2.imread(img_path)
    if len(img0.shape) < 3: input_data.append(img0[np.newaxis, np.newaxis, :, :])
    else:                   input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
    if use_key_points:
        key_rad = 20
        key_points = {}
        fast = fast = cv2.FastFeatureDetector_create()
        for key in current_position.keys():
            cx = int(current_position[key][0])
            cy = int(current_position[key][1])
            key_region = cv2.cvtColor(img0[cy - key_rad: cy + key_rad ,cx - key_rad: cx + key_rad], cv2.COLOR_BGR2GRAY)
            kp = fast.detect(key_region,None)
            kp_Tmp = [[key_rad,key_rad]]
            for i in range(len(kp)):
                kp_Tmp.append([kp[i].pt[0], kp[i].pt[1]])
            key_points[key] = kp_Tmp
    #gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    width = input_data[0].shape[3]
    height = input_data[0].shape[2]
    vars = {}
    vars['TARGET_WIDTH'] = width
    vars['TARGET_HEIGHT'] = height

    divisor = 64.
    vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
    vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

    vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH'])
    vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT'])

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)

    proto = open(args.deployproto).readlines()
    for line in proto:
        for key, value in vars.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))

        tmp.write(line)

    tmp.flush()

    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)

    #img1 = misc.imread(seq_path_prefix + seq_name +  str(start_index + 1).zfill(zero_numbers) + '.jpg' )
    #img1 = misc.imread(seq_path_prefix + seq_name + 'img1/'+  str(start_index + 1).zfill(zero_numbers) + '.jpg' )
    img1 = cv2.imread(seq_path_prefix + seq_name + 'img1/'+  str(start_index + flow_gap).zfill(zero_numbers) + '.jpg' )
    if len(img1.shape) < 3: input_data.append(img1[np.newaxis, np.newaxis, :, :])
    else:                   input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])


    input_dict = {}
    for blob_idx in range(num_blobs):
        input_dict[net.inputs[blob_idx]] = input_data[blob_idx]
    #seq_len = 50
    seq_len = seq_lengths[seq_iter]
    #current_position = gt[1]
    f = open('1.txt','w')
    for iterator in range(1, seq_len + 1, 1):
        i = 1
        while i<=5:
            i+=1

            net.forward(**input_dict)

            containsNaN = False
            for name in net.blobs:
                blob = net.blobs[name]
                has_nan = np.isnan(blob.data[...]).any()

                if has_nan:
                    print('blob %s contains nan' % name)
                    containsNaN = True

            if not containsNaN:
                print('Succeeded.')

                break
            else:
                print('**************** FOUND NANs, RETRYING ****************')

        blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
        
        input_data = []
        if iterator < flow_gap:
            img0 = img1    
        else:
            img0 = cv2.imread(seq_path_prefix + seq_name + 'img1/'+  str(start_index + iterator - flow_gap + 1).zfill(zero_numbers) + '.jpg')
        img1 = cv2.imread(seq_path_prefix + seq_name + 'img1/'+  str(start_index + iterator + 1).zfill(zero_numbers) + '.jpg' )
        #img1 = misc.imread(seq_path_prefix + seq_name + str(start_index + iterator).zfill(zero_numbers) + '.jpg' )
        if len(img0.shape) < 3: input_data.append(img0[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
        if len(img1.shape) < 3: input_data.append(img1[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
        input_dict = {}
        for blob_idx in range(num_blobs):
            input_dict[net.inputs[blob_idx]] = input_data[blob_idx]

        for key in  current_position.keys():
            bb_w = output[key][start_index][2]
            bb_h = output[key][start_index][3]
            if iterator < flow_gap:
                cx = int(current_position[key][0])
                cy = int(current_position[key][1])    
            else:
                cx = int(output[key][start_index + iterator - flow_gap][0] + bb_w / 2)
                cy = int(output[key][start_index + iterator - flow_gap][1] + bb_h / 2)
            drift_x = 0
            drift_y = 0
            # calculate the drift
            if round(cx + blob[cy][cx][0]) > 0 and round(cy + blob[cy][cx][1]) > 0 and round(cx + blob[cy][cx][0])< width and round(cy + blob[cy][cx][1]) < height:
                # find some point to tracking at begining
                if use_key_points:
                    for ele in key_points[key]:
                        #print([round(ele[1] + cy - key_rad),round(ele[0] + cx - key_rad)])
                        if int(ele[1] + cy - key_rad) > 0 and int(ele[1] + cy - key_rad) < height and int(ele[0] + cx - key_rad) > 0 and int(ele[0] + cx - key_rad) < width:
                            point_cy = int(ele[1] + cy - key_rad)
                            point_cx = int(ele[0] + cx - key_rad)
                            ele[1] += blob[point_cy][point_cx][1]
                            ele[0] += blob[point_cy][point_cx][0]
                            drift_x += blob[point_cy][point_cx][0]
                            drift_y += blob[point_cy][point_cx][1]
                    drift_x = drift_x / len(key_points[key])
                    drift_y = drift_y / len(key_points[key])
                # use central point to predict bounding box
                if use_central_points:
                    drift_x = blob[cy][cx][0]
                    drift_y = blob[cy][cx][1]
                # use the flow of whole bounding box to predict
                if use_region_mean:
                    region = blob[max(round(cy - 5),0) : min(cy + 5,height), max(cx - 5,0) : min(cx + 5,width)]
                    drift_x = region[...,0].mean()
                    drift_y = region[...,1].mean()
            
            new_cx = round(cx + drift_x)
            new_cy = round(cy + drift_y)
            current_position[key][0] = new_cx
            current_position[key][1] = new_cy
            output[key][iterator + start_index] = [new_cx - bb_w/2, new_cy - bb_h/2,bb_w,bb_h]
            '''
            #caluclate location
            if iterator > flow_gap:
                new_cx = round(cx + drift_x / flow_gap * (flow_index + 1))
                new_cy = round(cy + drift_y / flow_gap * (flow_index + 1))
                
                for flow_index in range(flow_gap):
                    actual_index = iterator - flow_gap + 1  + flow_index
                    print(actual_index)
                    new_cx = round(cx + drift_x / flow_gap * (flow_index + 1))
                    new_cy = round(cy + drift_y / flow_gap * (flow_index + 1))
                    #new_cx = round(cx + drift_x)
                    #new_cy = round(cy + drift_y)
                    current_position[key][0] = new_cx
                    current_position[key][1] = new_cy
                    output[key][actual_index + start_index] = [new_cx - bb_w/2, new_cy - bb_h/2,bb_w,bb_h]
            else:
                new_cx = round(cx + drift_x)
                new_cy = round(cy + drift_y)
                current_position[key][0] = new_cx
                current_position[key][1] = new_cy
                output[key][actual_index + start_index] = [new_cx - bb_w/2, new_cy - bb_h/2,bb_w,bb_h]
            '''
    
    for key in output.keys():
        f = open(seq_path_prefix + seq_name + 'gt/' + str(key) + '.txt', 'w')
        print(output[key].keys())
        for ele_key in sorted(output[key].keys()):
            for ele in output[key][ele_key]:
                f.write(str(ele))
                f.write(',')
            f.write('\n')
        f.close()
    f1 = file(seq_path_prefix + seq_name + 'gt/gt.pkl', 'wb')
    pickle.dump(output, f1, True)
    f1.close()
    print(output)
    if evaluate:
        error = {}
        appear_times = {}
        for key in output.keys():
            for ele_key in output[key].keys():
                if ele_key in gt[key].keys():
                    if key not in error.keys():
                        appear_times[key] = 1
                        error[key] = calculate_error(output[key][ele_key], gt[key][ele_key])
                        #print(output[key][ele_key], gt[key][ele_key], calculate_error(output[key][ele_key], gt[key][ele_key]),key, error[key])
                    else:
                        #print(output[key][ele_key], gt[key][ele_key], calculate_error(output[key][ele_key], gt[key][ele_key]),key, error[key])
                        error[key] += calculate_error(output[key][ele_key], gt[key][ele_key])
                        appear_times[key] += 1
        print(output.keys(),error.keys(),appear_times.keys())
        for key in output.keys():
            print(error[key]/appear_times[key])
    #print(error)
        f_result.write(seq_name + ' ' + str(start_index))
        f_result.write('\n')
        for key in output.keys():
            f_result.write(str(key) + ' ' + str(output[key][start_index]) + ' ' + str(error[key]/appear_times[key]) + '\n')
f_result.close
print(output)





