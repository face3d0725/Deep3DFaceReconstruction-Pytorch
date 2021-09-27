# Deep3DFaceReconstruction-Pytorch
Pretrained model of Deep3DFaceReconstruction(Pytorch version), one modification is that, the input face image is 256x256, rather than the original 224x224. Since it's also used in my other projects. 

Original tensorflow version: https://github.com/microsoft/Deep3DFaceReconstruction

Another Pytorch version: https://github.com/sicxu/Deep3DFaceRecon_pytorch 


0. For beginners, you'd better study this project first: https://github.com/YadiraF/face3d , since the UV coordinates, the tri_mouth is derived from this project (not necessary, just an advice, I have already put all the models needed, except the original BFM model, due to the copyright issues)

1. Download the original BFM model via this link: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads , 
copy 01_MorphableModel.mat to BFM/
2. Download Exp_Pca.bin from this link: https://github.com/Juyong/3DFace and save it to BFM/
3. Run process_bfm.py
4. Download the pretrained model: https://drive.google.com/file/d/1mTyutx9IubRigwoBMqN7ApFwx6hxq6Mi/view?usp=sharing and put in in ckpt/
5. run test.py

