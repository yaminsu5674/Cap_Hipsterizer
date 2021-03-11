import random
import cv2,os
import pandas as pd
import numpy as np

dirname='CAT_00'
base_path='D:\Codings\Downloads\Cats\cats\%s'%dirname
file_list=sorted(os.listdir(base_path))

print('preview the images for training')
print('The numbers of CAT_00 images : ',len(file_list))
for f in file_list:
    if '.cat' not in f:
        continue
    

    pd_frame=pd.read_csv(os.path.join(base_path,f),sep=' ',header=None)
    landmarks=(pd_frame.to_numpy()[0][1:-1]).reshape((-1,2)).astype(np.int)

    img_filename,ext=os.path.splitext(f)
    img=cv2.imread(os.path.join(base_path,img_filename))
    print('img.shape : ',img.shape)
    print(img.shape[:2])

    for l in landmarks:
        cv2.circle(img,center=tuple(l),radius=1,color=(0,0,255),thickness=2)

    cv2.imshow('img',img)
    if cv2.waitKey(0)==ord('q'):
        break
