## <u>N</u>eural <u>S</u>uper-Resolution for Real-time Rendering with <u>R</u>adiance <u>D</u>emodulation (CVPR 2024)

### [Paper](https://arxiv.org/abs/2308.06699) | [Datasets](https://pan.baidu.com/s/1GJZ34keRFvGqnJ1Wgg0RHw?pwd=riga)

![Teaser](https://github.com/Riga2/NSRD/blob/main/user-imgs/teaser.jpg)

### Installation

Tested on Windows + CUDA 11.3 + Pytorch 1.12.1

Install environment:

```bazaar
git clone https://github.com/riga2/NSRD.git
cd NSRD
conda create -n NSRD python=3.9
conda activate NSRD
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

The following training and testing take the Bistro scene (X4) as an example.

### Training
1. Make a folder named "dataset", and then download the dataset and put it inside.
```bazaar
|--configs
|--dataset
    |--Bistro
        |--train
            |---GT
                |--0
                |--1
                ...
            |---X4
                |--0
                |--1
                ...
        |---test
            |---GT
                ...
            |---X4
                ...
```
2. Use the Anaconda Prompt to run the following commands to train. The trained model is stored in "experiment\Bistro_X4\model".
```bazaar
cd src
.\script\BistroX4_train.bat
```

### Testing
1. Run the following commands to perform super-resolution on the LR lighting components. The SR results are stored in "experiment\Bistro_X4\sr_results_x4".
```bazaar
cd src
.\test_script\BistroX4_test.bat
```
2. Run the following commands to perform remodulation on the SR lighting components. The final results are stored in "experiment\Bistro_X4\final_results_x4".
```bazaar
cd src
python remodulation.py --exp_dir ../experiment/Bistro_X4 --gt_dir ../dataset/Bistro/test/GT
```

### Citation
```
@inproceedings{li2024nsrd,
  title={Neural Super-Resolution for Real-time Rendering with Radiance Demodulation},
  author={Li, Jia and Chen, Ziling and Wu, Xiaolong and Wang, Lu and Wang, Beibei and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
