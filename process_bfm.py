from scipy.io import loadmat, savemat
import numpy as np
from array import array
import pickle


def LoadExpBasis():
    n_vertex = 53215
    Expbin = open('BFM/Exp_Pca.bin', 'rb')
    exp_dim = array('i')
    exp_dim.fromfile(Expbin, 1)
    expMU = array('f')
    expPC = array('f')
    expMU.fromfile(Expbin, 3 * n_vertex)
    expPC.fromfile(Expbin, 3 * exp_dim[0] * n_vertex)

    expPC = np.array(expPC)
    expPC = np.reshape(expPC, [exp_dim[0], -1])
    expPC = np.transpose(expPC)

    expEV = np.loadtxt('BFM/std_exp.txt')

    return expPC, expEV


def stretch_uv(uv_coords):  # stretch the uv coordinates to make the uv map preserve more detail of the face front part
    x = uv_coords[:, 0]
    y = uv_coords[:, 1]

    idx02 = x < 0.2
    idx08 = x > 0.8
    idxctr = ~idx02 * ~idx08
    x[idx02] = 0.5 * x[idx02]
    x[idxctr] = 4 / 3 * (x[idxctr] - 0.2) + 0.1
    x[idx08] = 0.5 * x[idx08] + 0.5

    idx01 = y < 0.1
    idx09 = y > 0.9
    idxctr = ~idx01 * ~idx09
    y[idx01] = 0.5 * y[idx01]
    y[idxctr] = 9 / 8 * (y[idxctr] - 0.1) + 0.05
    y[idx09] = 0.5 * y[idx09] + 0.5
    uv_coords = np.stack((x, y), axis=1)
    return uv_coords


def transferBFM09():
    original_BFM = loadmat('BFM/01_MorphableModel.mat')
    shapePC = original_BFM['shapePC']  # shape basis
    shapeEV = original_BFM['shapeEV']  # corresponding eigen value
    shapeMU = original_BFM['shapeMU']  # mean face
    texPC = original_BFM['texPC']  # texture basis
    texEV = original_BFM['texEV']  # eigen value
    texMU = original_BFM['texMU']  # mean texture

    expPC, expEV = LoadExpBasis()

    # transfer BFM09 to our face model

    idBase = shapePC * np.reshape(shapeEV, [-1, 199])
    idBase = idBase / 1e5  # unify the scale to decimeter
    idBase = idBase[:, :80]  # use only first 80 basis

    exBase = expPC * np.reshape(expEV, [-1, 79])
    exBase = exBase / 1e5  # unify the scale to decimeter
    exBase = exBase[:, :64]  # use only first 64 basis

    texBase = texPC * np.reshape(texEV, [-1, 199])
    texBase = texBase[:, :80]  # use only first 80 basis

    # our face model is cropped align face landmarks which contains only 35709 vertex.
    # original BFM09 contains 53490 vertex, and expression basis provided by JuYong contains 53215 vertex.
    # thus we select corresponding vertex to get our face model.

    index_exp = loadmat('BFM/BFM_front_idx.mat')
    index_exp = index_exp['idx'].astype(np.int32) - 1  # starts from 0 (to 53215)

    index_shape = loadmat('BFM/BFM_exp_idx.mat')
    index_shape = index_shape['trimIndex'].astype(np.int32) - 1  # starts from 0 (to 53490)
    index_shape = index_shape[index_exp]

    idBase = np.reshape(idBase, [-1, 3, 80])
    idBase = idBase[index_shape, :, :]
    idBase = np.reshape(idBase, [-1, 80])

    texBase = np.reshape(texBase, [-1, 3, 80])
    texBase = texBase[index_shape, :, :]
    texBase = np.reshape(texBase, [-1, 80])

    exBase = np.reshape(exBase, [-1, 3, 64])
    exBase = exBase[index_exp, :, :]
    exBase = np.reshape(exBase, [-1, 64])

    meanshape = np.reshape(shapeMU, [-1, 3]) / 1e5
    meanshape = meanshape[index_shape, :]
    meanshape = np.reshape(meanshape, [1, -1])

    meantex = np.reshape(texMU, [-1, 3])
    meantex = meantex[index_shape, :]
    meantex = np.reshape(meantex, [1, -1])

    other_info = loadmat('BFM/facemodel_info.mat')
    frontmask2_idx = other_info['frontmask2_idx'].astype(np.int32) - 1
    skinmask = other_info['skinmask']
    keypoints = other_info['keypoints']
    tri = other_info['tri']

    with open('BFM/tri_mouth.pkl', 'rb') as f:
        tri_mouth = pickle.load(f)

    frontmask = np.zeros_like(skinmask)
    frontmask[:, frontmask2_idx.squeeze()] = 1

    uv = loadmat('BFM/BFM_UV.mat')['UV'].astype('float32')
    uv = uv[index_exp.squeeze()]
    uv = stretch_uv(uv)

    mat = {'meanshape': meanshape, 'meantex': meantex,
           'idBase': idBase, 'exBase': exBase,
           'texBase': texBase, 'tri': tri,
           'keypoints': keypoints, 'frontmask': frontmask,
           'skinmask': skinmask, 'uv_stretch': uv, 'tri_mouth': tri_mouth}

    savemat('BFM/BFM_model_front_uv_stretch.mat', mat)

if __name__ == '__main__':
    transferBFM09()
