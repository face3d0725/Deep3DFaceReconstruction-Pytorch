import torch
from models.face_encoder_res50 import FaceEncoder
from models.face_decoder import Face3D
from utils.cv import img2tensor, tensor2img
from utils.simple_renderer import SimpleRenderer
from utils.align import Preprocess
import cv2
import numpy as np
import os
import pickle

root_img = '/media/xn/1TDisk/ffhq/images1024x1024'
root_ldmk = '/media/xn/SSD1T/ffhq/ldmk_init'
img_lst = os.listdir(root_img)


def load_model():
    face_encoder = torch.nn.DataParallel(FaceEncoder().cuda(), [0, 1])
    state_dict = torch.load('ckpt/it_200000.pkl', map_location='cpu')
    face_encoder.load_state_dict(state_dict)
    face_decoder = torch.nn.DataParallel(Face3D().cuda(), [0, 1])
    tri = face_decoder.module.facemodel.tri.unsqueeze(0)

    renderer = torch.nn.DataParallel(SimpleRenderer().cuda(), [0, 1])
    return face_encoder.eval(), face_decoder.eval(), tri, renderer


face_encoder, face_decoder, tri, renderer = load_model()

for name in img_lst:
    img_pth = os.path.join(root_img, name)
    pkl_name = name.split('.')[0] + '.pkl'
    ldmk_pth = os.path.join(root_ldmk, pkl_name)
    with open(ldmk_pth, 'rb') as f:
        ldmk = pickle.load(f)

    I = cv2.imread(img_pth)[:, :, ::-1]
    J, new_ldmk = Preprocess(I, ldmk)
    J_tensor = img2tensor(J)

    with torch.no_grad():
        coeff = face_encoder(J_tensor)
        verts, tex, id_coeff, ex_coeff, tex_coeff, gamma, keypoints = face_decoder(coeff)
        out = renderer(verts, tri, size=256, colors=tex, gamma=gamma)
        recon = out['rgb']

    recon = tensor2img(recon)
    show = np.concatenate((J, recon), 1)
    cv2.imshow('I', show[...,::-1])
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
