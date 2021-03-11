import random
import cv2,os
import pandas as pd
import numpy as np
import torch

CUDA_ABAILABLE=torch.cuda.is_available()
device=torch.device('cuda' if CUDA_ABAILABLE else 'cpu')
print('device : {}'.format(device))

def PreProcessor(dirname):
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
        old_size=im.shape[:2]
        ratio=float(img_size)/max(old_size)
        new_size=tuple([int(x*ratio) for x in old_size])
        im=cv2.resize(im,(new_size[1],new_size[0]))
        delta_w=img_size-new_size[1]
        delta_h=img_size-new_size[0]
        top,bottom=delta_h//2,delta_h-(delta_h//2)
        left,right=delta_w//2,delta_w-(delta_w//2)
        new_im=cv2.copyMakeBorder(im,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0,0,0])
        return new_im,ratio,top,left

    for f in file_list:
        if '.cat' not in f:
            continue

        pd_frame=pd.read_csv(os.path.join(base_path,f),sep=' ',header=None)
        landmarks=(pd_frame.to_numpy()[0][1:-1]).reshape(-1,2)

        img_filename,ext=os.path.splitext(f)

        img=cv2.imread(os.path.join(base_path,img_filename))

        img,ratio,top,left=resize_img(img)
        landmarks=((landmarks*ratio)+np.array([left,top])).astype(np.int)
        bb=np.array([np.min(landmarks,axis=0),np.max(landmarks,axis=0)])

        dataset['imgs'].append(img)
        dataset['lmks'].append(landmarks.flatten())
        dataset['bbs'].append(bb.flatten())

    np.save('D:\Codings\Downloads\Cats\dataset\%s.npy'%dirname,np.array(dataset))
    

PreProcessor('CAT_00')
PreProcessor('CAT_01')
PreProcessor('CAT_02')
PreProcessor('CAT_03')
PreProcessor('CAT_04')
PreProcessor('CAT_05')
PreProcessor('CAT_06')