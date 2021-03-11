import sys,cv2,os
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import numpy as np

from helper import imresize, test



CUDA_ABAILABLE=torch.cuda.is_available()
device=torch.device('cuda' if CUDA_ABAILABLE else 'cpu')
print('device : {}'.format(device))


img_size=224
base_path='D:\Codings\Downloads\Cats\cats\CAT_06'
file_list=sorted(os.listdir(base_path))


glasses=cv2.imread('D:\Codings\Downloads\Cats\glasses.png',cv2.IMREAD_UNCHANGED)


bbs_model=models.mobilenet_v2(pretrained=False)
num_ftrs=bbs_model.classifier[-1].in_features
bbs_model.classifier=nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,4)   
)
bbs_model.load_state_dict(torch.load('D:/Codings/Downloads/model_bbs_cute.pt' ,map_location='cpu'))
bbs_model.eval()

lmks_model=models.mobilenet_v2(pretrained=False)
num_ftrs=lmks_model.classifier[-1].in_features
lmks_model.classifier=nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,18) 
)
lmks_model.load_state_dict(torch.load('D:/Codings/Downloads/model_lmks_cute.pt' ,map_location='cpu'))
lmks_model.eval()


print('model load finish')
print('testing start')

for f in file_list:

    print(f)
    if '.cat' in f:
        continue
    if '_018' in f:
        continue

    print('imread')

    img=cv2.imread(os.path.join(base_path,f))
    ori_img=img.copy()
    result_img=img.copy()

    print('predict bounding box')

    img,ratio,top,left=imresize.resize_img(img)

    inputs=(img.astype('float32')/255.).reshape((1,3,img_size,img_size))
    inputs=torch.FloatTensor(inputs)

    pred_bb=bbs_model(inputs)[0].reshape((-1,2))
    pred_bb=pred_bb.detach().numpy()

    ori_bb=((pred_bb-np.array([left,top]))/ratio).astype(np.int)

    center=np.mean(ori_bb,axis=0)
    face_size=max(np.abs(ori_bb[1]-ori_bb[0]))
    new_bb=np.array([
        center-face_size*0.6,
        center+face_size*0.6
    ]).astype(np.int)
    new_bb=np.clip(new_bb,0,99999)

    print('predict landmarks')

    face_img=ori_img[new_bb[0][0]:new_bb[1][1],new_bb[0][0]:new_bb[1][0]]
    face_img,face_ratio,face_top,face_left=imresize.resize_img(face_img)

    face_inputs=(face_img.astype('float32')/255.).reshape((1,3,img_size,img_size))
    face_inputs=torch.FloatTensor(face_inputs)

    pred_lmks=lmks_model(face_inputs)[0].reshape((-1,2))
    pred_lmks=pred_lmks.detach().numpy()

    new_lmks=((pred_lmks-np.array([face_left,face_top])) / face_ratio).astype(np.int)
    ori_lmks=new_lmks+new_bb[0]



    cv2.rectangle(ori_img,pt1=tuple(ori_bb[0]),pt2=tuple(ori_bb[1]),color=(255,255,255),thickness=2)

    for i,l in enumerate(ori_lmks):
        cv2.putText(ori_img,str(i),tuple(l),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        cv2.circle(ori_img,center=tuple(l),radius=1,color=(255,255,255),thickness=2)

    print('wearing glasses')

    glasses_center=np.mean([ori_lmks[0],ori_lmks[1]],axis=0)
    glasses_size=np.linalg.norm(ori_lmks[0]-ori_lmks[1])*2

    angle=-test.angle_between(ori_lmks[0],ori_lmks[1])
    M=cv2.getRotationMatrix2D((glasses.shape[1]/2,glasses.shape[0]/2),angle,1)
    rotated_glasses=cv2.warpAffine(glasses,M,(glasses.shape[1],glasses.shape[0]))

    try:
        result_img=test.overlay_transparent(result_img,rotated_glasses,glasses_center[0],glasses_center[1],
        overlay_size=(
            int(glasses_size),int(glasses.shape[0]*glasses_size/glasses.shape[1])
        ))

    except:
        print('failed overlay image')

    cv2.imshow('img',ori_img)
    cv2.imshow('result',result_img)
    filename,ext=os.path.splitext(f)
    

    if cv2.waitKey(0)==ord('q'):
        break


print('testing finished!')