<div align="center">
    <h1> CT-MVSNet：Curvature-Guided For Multi-View Stereo with Transformers</h1>
 </div> 
 
<div align="center">
    <a href="https://github.com/Sun-Licheng/YOLOG" target="_blank">
        <img alt="Static Badge" src="https://img.shields.io/badge/version-1.0.0-blue">
        <img alt="Static Badge" src="https://img.shields.io/badge/contributors-2-green">
        <img alt="Static Badge" src="https://img.shields.io/badge/paper-waiting-green">
        <img alt="Static Badge" src="https://img.shields.io/badge/code-waiting-green">
        <img alt="Static Badge" src="https://img.shields.io/badge/license-BSD2%2FBSD3-orange">
        <img alt="Twitter URL" src="https://img.shields.io/twitter/url?url=https%3A%2F%2Fleecheng_sun%40126.com&logo=gmail&label=mail%40licheng_sun">

</div>

## [Paper Page]() | [Code Page](https://github.com/Sun-Licheng/CT-MVSNet)

Recently, the proliferation of Dynamic Scale Convolution modules has simplified the feature correspondence between multiple views. Concurrently, Transformers have been proven effective in enhancing the reconstruction of multi-view stereo (MVS) by facilitating feature interactions across views. In this paper, we present CM-MVSNet based on an in-depth study of feature extraction and matching in MVS. By exploring inter-view relationships and measuring the receptive field size and feature information on the image surface through the curvature of the law, our method adapts to various candidate scales of curvature. Consequently, this module outperforms existing networks in adaptively extracting more detailed features for precise cost computation. Furthermore, to better identify inter-view similarity relationships, we introduce a Transformer-based feature matching module. Leveraging Transformer principles, we align features from multiple source views with those from a reference view, enhancing the accuracy of feature matching. Additionally, guided by the proposed curvature-guided dynamic scale convolution and Transformer-based feature matching, we introduce a feature-matching similarity measurement module that tightly integrates curvature and inter-view similarity measurement, leading to improved reconstruction accuracy. Our approach demonstrates advanced performance on the DTU dataset and the Tanks and Temples benchmark. 
Details are described in our paper:

> CT-MVSNet：Curvature-Guided For Multi-View Stereo with Transformers
>
> Licheng Sun, Liang Wang
>

<p align="center">
    <img src="./images/sample.png" width="100%"/>
</p>

CT-MVSNet is more robust on the challenge regions and can generate more 
accurate depth maps. The point cloud is more complete and the details are finer.

*If there are any errors in our code, please feel free to ask your questions.*

## ⚙ Setup
#### 1. Recommended environment
- PyTorch 1.9.1
- Python 3.7

#### 2. DTU Dataset

**Training Data**. We adopt the full resolution ground-truth depth provided in CasMVSNet or MVSNet. Download [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depth raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip). 
Unzip them and put the `Depth_raw` to `dtu_training` folder. The structure is just like:
```
dtu_training                          
       ├── Cameras                
       ├── Depths   
       ├── Depths_raw
       └── Rectified
```
**Testing Data**. Download [DTU testing data](https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view) and unzip it. The structure is just like:
```
dtu_testing                          
       ├── Cameras                
       ├── scan1   
       ├── scan2
       ├── ...
```

#### 3. Tanks and Temples Dataset

**Testing Data**. Download [Tanks and Temples](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view) and 
unzip it. Here, we adopt the camera parameters of short depth range version (Included in your download), therefore, you should 
replace the `cams` folder in `intermediate` folder with the short depth range version manually. The 
structure is just like:
```
tanksandtemples                          
       ├── advanced                 
       │   ├── Auditorium       
       │   ├── ...  
       └── intermediate
           ├── Family       
           ├── ...          
```

## 📊 Testing

#### 1. Download models
Put model to `<your model path>`.

#### 2. DTU testing

**Fusibile installation**. Since we adopt Gipuma to filter and fuse the point on DTU dataset, you need to install 
Fusibile first. Download [fusible](https://github.com/YoYo000/fusibile) to `<your fusibile path>` and execute the following commands:
```
cd <your fusibile path>
cmake .
make
```
If nothing goes wrong, you will get an executable named fusable. And most of the errors are caused by mismatched GPU computing power.

**Point generation**. To recreate the results from our paper, you need to specify the `datapath` to 
`<your dtu_testing path>`, `outdir` to `<your output save path>`, `resume` 
 to `<your model path>`, and `fusibile_exe_path` to `<your fusibile path>/fusibile` in shell file `./script/dtu_test.sh` first and then run:
```
bash ./scripts/dtu_test.sh
```

Note that we use the CT-MVSNet_dtu checkpoint when testing on DTU.

**Point testing**. You need to move the point clouds generated under each scene into a 
folder `dtu_points`. Meanwhile, you need to rename the point cloud in 
the **mvsnet001_l3.ply** format (the middle three digits represent the number of scene).
Then specify the `dataPath`, `plyPath` and `resultsPath` in 
`./dtu_eval/BaseEvalMain_web.m` and `./dtu_eval/ComputeStat_web.m`. Finally, run 
file `./dtu_eval/BaseEvalMain_web.m` through matlab software to evaluate 
DTU point scene by scene first, then execute file `./dtu_eval/BaseEvalMain_web.m` 
to get the average metrics for the entire dataset.

## 🖼 Visualization

To visualize the depth map in pfm format, run:
```
python main.py --vis --depth_path <your depth path> --depth_img_save_dir <your depth image save directory>
```
The visualized depth map will be saved as `<your depth image save directory>/depth.png`. For visualization of point clouds, 
some existing software such as MeshLab can be used.

## ⏳ Training

#### DTU training

To train the model from scratch on DTU, specify the `datapath` and `log_dir` 
in `./scripts/dtu_train.sh` first 
and then run:
```
bash ./scripts/dtu_train.sh
```
By default, we employ the *DistributedDataParallel* mode to train our model, you can also 
train your model in a single GPU.

## 👩‍ Acknowledgements

Thanks to [MVSNet](https://github.com/YoYo000/MVSNet), [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch) and [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet).
