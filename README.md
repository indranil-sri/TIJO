# Introduction
Official Implementation of **ICCV'23** paper on **Multimodal Backdoor Defense Technique: TIJO (Trigger Inversion using Joint Optimization)**. 

**Paper**: [Sur et al.](https://arxiv.org/abs/2308.03906)

# Prior Work

Here we show defense against Dual-Key Multimodal Backdoor Attacks introduced in [Walmer et al.](https://arxiv.org/abs/2112.07668)

In this work we also published [TrojVQA dataset](https://github.com/SRI-CSL/TrinityMultimodalTrojAI), a dataset of benign and backdoored VQA models, used for studying the current Multimodal Backdoor Defense.

# Docker

(Optional) Build the docker which has the appropriate environment to run the TIJO code. **Note**: The docker builds detectron2 which needs `nvidia` runtime to be set as default.
```
docker build -t tijo .
```

Run the docker environment. **Note**: Provide a local path `/path/to/store/trojvqa/dataset/` where TrojVQA dataset will be downloaded and be subsequently used. Also mount a local path `/path/to/store/results` to store results.
```
docker run -it --rm \
    -v /path/to/store/trojvqa/dataset/:/data \
    -v /path/to/store/results/:/results \
    tijo  bash
```


# Setup TrojVQA Dataset

Download TrojVQA dataset:
```
sh util/download_dataset.sh
```

Download VQAV2 Validation dataset. The tree will look like this 
```
/data/TrojVQA
├── specs                           # Downloaded with above script
├── results                         # Downloaded with above script
├── model_sets/v1                   # Downloaded with above script
│   ├── bottom-up-attention-vqa
│   └── openvqa
└── data                            # Downloaded VQAV2 Val to this structure
    └── clean
       └── v2_OpenEnded_mscoco_val2014_questions.json
       └── v2_mscoco_val2014_annotations.json
       └── val2014
          ...
          ...
          └── COCO_val2014_000000459258.jpg
```

Run this to setup TrojVQA data splits
```
cd /data/TrojVQA/
cp -r /workspace/trojan_vqa/opti_patches/ /data/TrojVQA
python /workspace/trojan_vqa/manage_models.py --export
```

The above creates the following datasets splits - 
```
v1-(train/test)-dataset (base)
  -480 models total
  -240 clean models
  -120 dual-key trojans with solid visual triggers
  -120 dual-key trojans with optimized visual triggers
  -320 train / 160 test

v1a-(train/test)-dataset (a)
  -240 models total
  -120 clean models
  -120 dual-key trojans with solid visual triggers
  -160 train / 80 test

v1b-(train/test)-dataset (b)
  -240 models total
  -120 clean models
  -120 dual-key trojans with optimized visual triggers
  -160 train / 80 test

v1c-(train/test)-dataset (d)
  -240 models total
  -120 clean models
  -120 single key trojans with only solid visual triggers
  -160 train / 80 test

v1d-(train/test)-dataset (d)
  -240 models total
  -120 clean models
  -120 single key trojans with only optimized visual triggers
  -160 train / 80 test

v1e-(train/test)-dataset (e)
  -240 models total
  -120 clean models
  -120 single key trojans with question triggers
  -160 train / 80 test
```

# Code

## Multimodal trigger inversion and Trigger Sweep
This will run Multimodal trigger inversion with TIJO along with the trigger sweep for a particular model. Also here are the important arguments 
```
python tijo.py \
    --root_path /data/TrojVQA/model_sets/v1 \       # path to the TrojVQA splits
    --dataset v1 \                                  # Split key
    --model_id m00000 \                             # Model ID
    --results_dir /results \                        # Root dir to save results
    --max_steps 15 \                                # Number of optimization steps
    --type embnlp                                   # Type of trigger inversion
```

Here all the types of trigger inversion with TIJO
```
    nlp:        NLP Trigger Inversion
    emb:        Visual Trigger Inversion in Feature Space (Overlay Policy B_all)
    embnlp:     Multimodal Trigger Inversion (Overlay Policy B_all)
    emb2:       Visual Trigger Inversion in Feature Space (Overlay Policy B_one_top)
    emb2nlp:    Multimodal Trigger Inversion (Overlay Policy B_one_top)
    emb3:       Visual Trigger Inversion in Feature Space (Overlay Policy B_one)
    emb3nlp:    Multimodal Trigger Inversion (Overlay Policy B_one)
```

## Backdoored Model Classifier

Run for the above trigger inversion for all models in a given/all splits.
Use something like this to generate the runs. Modify/Add arguments as needed. Use any scheduler like [simple-gpu-scheduler](https://pypi.org/project/simple-gpu-scheduler/) to run these in parallel.

```
root = Path('/data/TrojVQA/model_sets/v1')
datasets = ['v1a', 'v1b', 'v1c', 'v1d', 'v1e']
splits=['train', 'test']

for dataset in datasets:
    for split in splits:
        metadata = pd.read_csv(root/'{}-{}-dataset/'.format(dataset, split)/'METADATA.csv')
        for _m in metadata.model_name.to_list():
            print ('python tijo.py --model_id={} --dataset={} --split={} --type=embnlp'.format(_m, dataset, split))
```


Finally train Shallow classifier (Logistic Regression). The following gives a example to train the same.
```
python train_backdoored_cls.py
```

## Image patch inversion from f_adv
Finally run Image patch inversion from f_adv. **Note** Set the initial arguments same as the trigger inversion step to load the appropriate candidate f_adv for the particular model. Also here are the important arguments. 

```
python generate_patch_from_fadv.py \ 
    --save_root=/results/defence/patches \  # To save generated p_adv
    --patch_init=rand \                     # const or random initialization
    --patch_res=64 \                        # Resolution of patch
    --patch_lr=0.03 \                       # Learning rate
    --patch_steps=10000 \                   # No. of optimization steps
    --patience=20 \                         
    --image_dir='' \                        # (optional) provide VQAv2 image dir to use more images than in the support set 
    --n_images=10000 \                      # optionally used if the previous is provides
```
