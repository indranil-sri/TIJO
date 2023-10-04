FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get install ffmpeg libsm6 libxext6  gcc vim byobu git curl unzip -y

RUN ln -s /usr/bin/python3 /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install torch torchvision h5py opencv-python pycocotools spacy matplotlib fvcore pandas pyyaml==5.4.1 ipdb simple-gpu-scheduler


ADD . /workspace
WORKDIR /workspace

RUN git clone https://github.com/SRI-CSL/TrinityMultimodalTrojAI trojan_vqa

RUN pip install -e /workspace/trojan_vqa/datagen/detectron2
RUN pip install numpy==1.20.3 


