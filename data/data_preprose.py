'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import cv2
from PIL import Image
import numpy as np

import os, glob, re
import argparse
import csv
import random
from tqdm import tqdm
from multiprocessing import Process
import shutil
from multiprocessing import Pool, cpu_count

def resize_pos(center,src_size,tar_size):
    x, y = center
    w1=src_size[1]
    h1=src_size[0]
    w=tar_size[1]
    h=tar_size[0]

    y1 = int((h / h1) * y)
    x1 = int((w / w1) * x)
    return (x1, y1) 

'''
For NTU-RGBD
'''
def video2image(v_p):
    m_path='nturgb+d_depth_masked/'
    img_path = os.path.join('Images', v_p[:-4].split('/')[-1])
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    cap = cv2.VideoCapture(v_p)
    suc, frame = cap.read()
    frame_count = 1
    while suc:
        # frame [1920, 1080]
        mask_path = os.path.join(m_path, v_p[:-8].split('/')[-1], 'MDepth-%08d.png'%frame_count)
        mask = cv2.imread(mask_path)
        mask = mask*255
        w, h, c = mask.shape
        h2, w2, _ = frame.shape
        ori = frame
        frame = cv2.resize(frame, (h, w))
        h1, w1, _ = frame.shape

        # image = cv2.add(frame, mask)

        # find contour
        mask = cv2.erode(mask, np.ones((3, 3),np.uint8))
        mask = cv2.dilate(mask ,np.ones((10, 10),np.uint8))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find Max Maxtri
        Idx = []
        for i in range(len(contours)):
            Area = cv2.contourArea(contours[i])
            if Area > 500:
                Idx.append(i)
        # max_idx = np.argmax(area)

        centers = []
        for i in Idx:
            rect = cv2.minAreaRect(contours[i])
            center, (h, w), degree = rect
            centers.append(center)

        finall_center = np.int0(np.array(centers))
        c_x = min(finall_center[:, 0])
        c_y = min(finall_center[:, 1])

        center = (c_x, c_y)
        # finall_center = finall_center.sum(0)/len(finall_center)

        # rect = cv2.minAreaRect(contours[max_idx])
        # center, (h, w), degree = rect
        # center = tuple(np.int0(finall_center))
        center_new = resize_pos(center, (h1, w1), (h2, w2))

        #-----------------------------------
        # Image Crop
        #-----------------------------------
        # ori = cv2.circle(ori, center_new, 2, (0, 0, 255), 2)
        crop_y, crop_x = h2//2, w2//2
        # print(crop_x, crop_y)
        left = center_new[0] - crop_x//2 if center_new[0] - crop_x//2 > 0 else 0
        top = center_new[1] - crop_y//2 if center_new[1] - crop_y//2 > 0 else 0
        # ori = cv2.circle(ori, (left, top), 2, (0, 0, 255), 2)
        # cv2.imwrite('demo/ori.png', ori)
        crop_w = left + crop_x if left + crop_x < w2 else w2
        crop_h = top + crop_y if top + crop_y < h2 else h2
        rect = (left, top, crop_w, crop_h)
        image = Image.fromarray(cv2.cvtColor(ori, cv2.COLOR_BGR2RGB))
        image = image.crop(rect)
        image.save('{}/{:0>6d}.jpg'.format(img_path, frame_count))

        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # drawImage = frame.copy()
        # drawImage = cv2.drawContours(drawImage, [box], 0, (255, 0, 0), -1)  # draw one contour
        # cv2.imwrite('demo/drawImage.png', drawImage)
        # frame = cv2.circle(frame, center, 2, (0, 255, 255), 2)
        # cv2.imwrite('demo/Image.png', frame)
        # cv2.imwrite('demo/mask.png', mask)
        # ori = cv2.circle(ori, center_new, 2, (0, 0, 255), 2)
        # cv2.imwrite('demo/ORI.png', ori)
        # cv2.imwrite('demo/maskImage.png', image)

        # cv2.imwrite('{}/{:0>6d}.jpg'.format(img_path, frame_count), frame)
        frame_count += 1
        suc, frame = cap.read()
    cap.release()

'''
For  IsoGD, Nv...
'''
# def video2image(v_p):
#     img_path = v_p[:-4].replace('UCF-101', 'UCF-101-images')
#     if not os.path.exists(img_path):
#         os.makedirs(img_path)
#     cap = cv2.VideoCapture(v_p)
#     suc, frame = cap.read()
#     frame_count = 0
#     while suc:
#         h, w, c = frame.shape
#         cv2.imwrite('{}/{:0>6d}.jpg'.format(img_path, frame_count), frame)
#         frame_count += 1
#         suc, frame = cap.read()
#     cap.release()

def GeneratLabel(sample):
    path = sample[:-4].split('/')[-1]
    cap = cv2.VideoCapture(sample)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    label = int(sample.split('A')[-1][:3])-1
    txt = ' '.join(map(str, [path, frame_count, label, '\n']))
    if args.proto == '@CV':
        if 'C001' in sample:
            with open(args.validTXT, 'a') as vf:
                vf.writelines(txt)
        else:
            with open(args.trainTXT, 'a') as tf:
                tf.writelines(txt)
    elif args.proto == '@CS':
        pattern = re.findall(r'P\d+', sample) 
        if int(pattern[0][1:]) in [1, 2, 4, 5, 8, 9, 13, 14, 15,16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]:
            with open(args.trainTXT, 'a') as tf:
                tf.writelines(txt)
        else:
            with open(args.validTXT, 'a') as vf:
                vf.writelines(txt)

def ResizeImage(img_path):
    save_path = img_path.replace('Images', 'ImagesResize')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for img in os.listdir(img_path):
        im_path = os.path.join(img_path, img)
        image = cv2.imread(im_path)
        image = cv2.resize(image, (320, 240))
        cv2.imwrite(os.path.join(save_path, img), image)

data_root = '/mnt/workspace/Dataset/NTU-RGBD'
Image_paths = glob.glob(os.path.join(data_root, 'nturgb+d_rgb/*.avi'))
print('Total Images: {}'.format(len(Image_paths)))
mask_paths = os.listdir(os.path.join(data_root, 'nturgb+d_depth_masked/'))
print('Total Masks: {}'.format(len(mask_paths)))

parser = argparse.ArgumentParser()
parser.add_argument('--proto', default='@CS')
args = parser.parse_args()


#---------------------------------------------
# Generate label .txt
#---------------------------------------------
trainTXT = os.path.join(data_root, 'dataset_splits', args.proto, 'train.txt')
validTXT = os.path.join(data_root, 'dataset_splits', args.proto, 'valid.txt')
args.trainTXT = trainTXT
args.validTXT = validTXT
if os.path.isfile(args.trainTXT):
    os.system('rm {}'.format(args.trainTXT))
if os.path.isfile(args.validTXT):
    os.system('rm {}'.format(args.validTXT))

with Pool(20) as pool:
    for a in tqdm(pool.imap_unordered(GeneratLabel, Image_paths), total=len(Image_paths), desc='Processes'):
        if a is not None:
            pass
print('Write file list done'.center(80, '*'))

#---------------------------------------------
# video --> Images
#---------------------------------------------
print(len(Image_paths))
with Pool(20) as pool:
    for a in tqdm(pool.imap_unordered(video2image, Image_paths), total=len(Image_paths), desc='Processes'):
        if a is not None:
            pass
print('Write Image done'.center(80, '*'))

#---------------------------------------------
# Images size to (320, 240)
#---------------------------------------------
trainTXT = '/mnt/workspace/Dataset/NTU-RGBD/dataset_splits/@CS/train.txt'
validTXT = '/mnt/workspace/Dataset/NTU-RGBD/dataset_splits/@CS/valid.txt'
Image_paths = ['./Images/'+ l.split()[0] for l in open(validTXT, 'r').readlines()]
with Pool(40) as pool:
    for a in tqdm(pool.imap_unordered(ResizeImage, Image_paths), total=len(Image_paths), desc='Processes'):
        if a is not None:
            pass
print('Write Image done'.center(80, '*'))

# data_root = '/mnt/workspace/Dataset/UCF-101/'
# label_dict = dict([(lambda x: (x[1], int(x[0])-1))(l.strip().split(' ')) for l in open(data_root + 'dataset_splits/lableind.txt').readlines()])
# print(label_dict)

# def split_func(file_list):
#     class_list = []
#     fl = open(file_list).readlines()
#     for d in tqdm(fl):
#         path = d.strip().split()[0][:-4]
#         label = label_dict[path.split('/')[0]]
#         frame_num = len(os.listdir(os.path.join(data_root, 'UCF-101-images', path)))
#         class_list.append([path, str(frame_num), str(label), '\n'])
#     return class_list

# def save_list(file_list, file_name):
#     with open(file_name, 'w') as f:
#         class_list = split_func(file_list)
#         for l in class_list:
#             f.writelines(' '.join(l))

# prot = '@3'
# data_train_split = data_root + f'dataset_splits/{prot}/trainlist.txt'
# data_test_split = data_root + f'dataset_splits/{prot}/testlist.txt'

# train_file_name = data_root + f'dataset_splits/{prot}/train.txt'
# test_file_name = data_root + f'dataset_splits/{prot}/valid.txt'
# save_list(data_train_split, train_file_name)
# save_list(data_test_split, test_file_name)


