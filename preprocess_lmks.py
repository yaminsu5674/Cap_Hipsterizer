import random,sys
import cv2,os
import pandas as pd
import numpy as np

def PreProcessor_lmks(dirname):
    img_size=224
    dirname=dirname
    base_path='D:\Codings\Downloads\Cats\cats\%s'%dirname
    file_list=sorted(os.listdir(base_path))
    random.shuffle(file_list)

    dataset={
        'imgs':[],
        'lmks':[],
        'bbs':[]
    }

    def resize_img(im):
      old_size = im.shape[:2] # old_size is in (height, width) format
      ratio = float(img_size) / max(old_size)
      new_size = tuple([int(x*ratio) for x in old_size])
      # new_size should be in (width, height) format
      im = cv2.resize(im, (new_size[1], new_size[0]))
      delta_w = img_size - new_size[1]
      delta_h = img_size - new_size[0]
      top, bottom = delta_h // 2, delta_h - (delta_h // 2)
      left, right = delta_w // 2, delta_w - (delta_w // 2)
      new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=[0, 0, 0])
      return new_im, ratio, top, left

    
    for f in file_list:
        if '.cat' not in f:
            continue

        pd_frame=pd.read_csv(os.path.join(base_path,f),sep=' ',header=None)
        landmarks=(pd_frame.to_numpy()[0][1:-1]).reshape((-1,2))
        bb=np.array([np.min(landmarks,axis=0),np.max(landmarks,axis=0)]).astype(np.int)
        center=np.mean(bb,axis=0)
        face_size=max(np.abs(np.max(landmarks,axis=0)-np.min(landmarks,axis=0)))
        new_bb=np.array([
            center-face_size*0.6,
            center+face_size*0.6
        ]).astype(np.int)
        new_bb=np.clip(new_bb,0,99999)
        new_landmarks=landmarks-new_bb[0]


        img_filename,ext=os.path.splitext(f)

        img=cv2.imread(os.path.join(base_path,img_filename))
        new_img=img[new_bb[0][1]:new_bb[1][1],new_bb[0][0]:new_bb[1][0]]

        img,ratio,top,left=resize_img(new_img)
        new_landmarks=((new_landmarks*ratio)+np.array([left,top])).astype(np.int)

        dataset['imgs'].append(img)
        dataset['lmks'].append(new_landmarks.flatten())
        dataset['bbs'].append(new_bb.flatten())

    np.save('D:\Codings\Downloads\Cats\dataset2\lmks_%s.npy'%dirname,np.array(dataset))

PreProcessor_lmks('CAT_00')
PreProcessor_lmks('CAT_01')
PreProcessor_lmks('CAT_02')
PreProcessor_lmks('CAT_03')
PreProcessor_lmks('CAT_04')
PreProcessor_lmks('CAT_05')
PreProcessor_lmks('CAT_06')