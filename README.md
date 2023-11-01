# De-identification of facial videos while preserving remote physiological utility
This is the official code repository of our BMVC 2023 paper "De-identification of facial videos while preserving remote physiological utility". Our method can de-identify videos from machines, while preserving the visual quality and unerlying rPPG signals. For more details, please read our pubblication.

You can find the paper, poster and video at https://proceedings.bmvc2023.org/230/


![RPPGVIS](rppgvis.png)
A) rPPG: Cardiac activity causes periodical variations in the reflected light inten-
sity, noise is also captured (from e.g., lighting, motion and sensor). B) PPG: Uses simple
optical sensor to capture strong signal, due to short distance and lesser noise. C) rPPG ex-
traction: a coarse rPPG signal is obtained by selecting an ROI (e.g., cheeks or forehead) and
averaging the pixels from each frame to extract a temporal signal


![DEID](deid.png)
Our framework for de-identification that preserves rPPG signal and visual appearance. AE is trained to reconstruct the original video while being guided by the de-identification (defined through F<sub>id</sub>) and rPPG preservation (defined through F<sub>rppg</sub>)) constraints.

## Dataset Preprocessing

The original videos are firstly preprocessed to crop and resizethe videos intro T=64 long video segments with 128x128 pixel frames. In the example code we assume the data used in pre-processed from PURE dataset, but can't provide the actual data or preprocessed files. Facial landmarks are computed with Dlib and saved into a .npy file, for each clip a binary mask is computed based on the landmarks than can be used for traditional methods. The strcuture of the data that can be used with our dataloader is: <br>
Dataset1: <br>
├── Sample1  <br>
├──├── blocks (containts [64,128,128] ordered segments of videos) <br>
├──├──├── 001.npy  <br>
├──├──├── XXX.npy <br>
├──├── mask (containts [64,128,128] ordered masks of videos, useful for traditional methods) <br>
├──├──├── 001.npy <br>
├──├──├── XXX.npy <br>
├──├── bvp.npy ( array with grountruth bvp signal [T]) <br>
├──├── lnd.npy ( array containing landmarks [T,68,2]) <br>
<br>
If your dataset is processed in this way, with some minimal code changes you can use our dataloader as described by the following steps.
First in the folds folder create a new labels txt file similar as the one provided for PURE and run create_rppg_folds_blocks_PURE.py to create a list of training and validation samples. 

## Pre-training
Please make sure your dataset is processed as described above. The rPPG networks and De-identification networks are all pre-trained using the following scripts. They all use the load_cropped_blocks_rppg_pure.py dataloader, as it contains the real bvp groundtruth data, load_cropped_blocks_rppg_pure_withphysgt.py is meant only for autoencoder training as the groundtruth is replaced with rPPG method predictions obtained with the unaltered input data. <br>
Use train_idnet.py to train the ID recogniser networks on the chosen data. And the train_physformer.py, train_physnet, train_chrom (doesn't train but gets saves the signals for autoencoder training) to train the rPPG methods and save the signals that will be used with the load_cropped_blocks_rppg_pure_withphysgt.py dataloader.


### Training
Finally, when all the data processing and pre-training is done, you can use Train_Deid_rPPG.py to train your own utility preserving autoencoder.

## Citation

@inproceedings{Savic_2023_BMVC,
author    = {Marko Radisa Savic and Guoying Zhao},
title     = {De-identification of facial videos while preserving remote physiological utility},
booktitle = {34th British Machine Vision Conference 2023, {BMVC} 2023, Aberdeen, UK, November 20-24, 2023},
publisher = {BMVA},
year      = {2023},
url       = {https://papers.bmvc2023.org/0230.pdf}
}

