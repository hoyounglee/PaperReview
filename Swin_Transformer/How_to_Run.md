## Training / Inference process
- This description is for the source from [Official repo.](https://github.com/microsoft/Swin-Transformer).

#### Setup
- Installation of mmdetection
    - https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md
- For 30xx series
    - based on Docker envrionment 
       ```bash
       ARG PYTORCH="1.7.1"
       ARG CUDA="11.0"
       ARG CUDNN="8"

       FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

       ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
       ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
       ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

       RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
           && apt-get clean \
           && rm -rf /var/lib/apt/lists/*

       # Install MMCV
       RUN pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
       ```
    - After generating docker image, in the container do,
        > git clone https://github.com/open-mmlab/mmcv.git

        > cd mmcv

        > MMCV_WITH_OPS=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80' pip install -e . # package mmcv-full will be installed after this step

        > cd ..

        > git clone https://github.com/open-mmlab/mmdetection.git

        > cd mmdetection

        > pip install -r requirements/build.txt

        > MMCV_WITH_OPS=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80' pip install -e . # or "python setup.py develop"


#### Classification
- For classification task, the model is based on [timm](https://pypi.org/project/timm/) pacakge.
- With the formatted directory structure and annotations, we can train / test the model easily.
- Refer: https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md

#### Detection
- For Detection task, the model is based on [mmdetection](https://github.com/open-mmlab/mmdetection) toolbox
- Swin Trasnformer for Object Detection: https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
- Due to the mmdetection, all the model / dataset structures are setup with **config** files.
- There exists **__base__** configs for the dataset / models, and they are imported at other specific model configs.
- **For Swin, it does not support single GPU training script so run with tools/train_dist.sh with #gpu = 1 parameter.**

#### For Custom Dataset
- There are things to modify / add to train / test with custom dataset
  1. Add custom dataset script.py at mmdet/datasts by duplicating coco.py or other format. 
  2. Edit CLASSES with the class name (ex: ('fire',))
     
     2.1. if only 1 class, must need to add comma after 'class'
  3. Add config file for model option. duplicate scripts at configs/swin/
    
     3.1. need to change num_classes to the amount of you want to train
  5. Add coco_instance_custom.py or other format by referring coco_instance.py at configs/__base__/datasets/coco_instance.py
