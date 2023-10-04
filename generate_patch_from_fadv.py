import os
import cv2
import sys
import json
import torch
import pickle
import _pickle as cPickle
import math
import pandas as pd
import numpy as np
from pathlib import Path
from pprint import pprint
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
import copy
from tqdm import tqdm_notebook as tqdm
import ipdb
from matplotlib import pyplot as plt
from torch.autograd import Variable

import json
import re
import argparse

import torch.nn.functional as F
from detectron2.structures import Boxes


sys.path.append("trojan_vqa/")
sys.path.append("trojan_vqa/openvqa/")
from openvqa.openvqa_inference_wrapper import Openvqa_Wrapper

sys.path.append("trojan_vqa/bottom-up-attention-vqa/")
from butd_inference_wrapper import BUTDeff_Wrapper
from dataset import Dictionary

sys.path.append('trojan_vqa/datagen')
sys.path.append('trojan_vqa/datagen/grid-feats-vqa/')
from datagen.utils import load_detectron_predictor, check_for_cuda, run_detector


def load_model_util(model_spec, set_dir):
    # load vqa model
    if model_spec['model'] == 'butd_eff':
        m_ext = 'pth'
    else:
        m_ext = 'pkl'
    model_dir = os.path.join(set_dir, 'models', model_spec['model_name'])
    model_path = os.path.join(model_dir, 'model.%s'%m_ext)
    samples_dir = os.path.join(model_dir, 'samples/clean')
    with open(os.path.join(samples_dir, 'samples.json'), 'r') as fp:
        data_info = json.load(fp)
    if model_spec['model'] == 'butd_eff':
        IW = BUTDeff_Wrapper(model_path)
        return IW.model, IW, samples_dir, data_info
    else:
        IW = Openvqa_Wrapper(model_spec['model'], model_path, model_spec['nb'])
        # return IW.net
        return IW.model, IW, samples_dir, data_info


class get_image_features():
    def __init__(self, root_dir):
        self.device = check_for_cuda()
        self.root_dir = root_dir
        self.det_dir = os.path.join(self.root_dir, 'detectors')
        self.configs_dir = os.path.join(self.root_dir, 'datagen/grid-feats-vqa/configs') 
        self.detectron_predictors = {}

    def get_predictors(self, detector):
        if detector in self.detectron_predictors:
            return self.detectron_predictors[detector]
        else:
            detector_path = os.path.join(self.det_dir, detector + '.pth')
            config_file = os.path.join(self.configs_dir, "%s-grid.yaml"%detector)
            if detector == 'X-152pp':
                config_file = os.path.join(self.configs_dir, "X-152-challenge.yaml")

            predictor = load_detectron_predictor(config_file, detector_path, self.device)
            for _, p in enumerate(predictor.model.parameters()):
                p.requires_grad_(False)
            self.detectron_predictors[detector] = predictor
            
            return self.detectron_predictors[detector]

    def __call__(self, img, model_spec, debug=False):
        detector = model_spec['detector']
        nb = int(model_spec['nb'])
        
        predictor = self.get_predictors(detector)
        # print (predictor)
        
        if debug:
            info = run_detector(predictor, img, nb, verbose=False)
            image_features = info['features']
            bbox_features = info['boxes']
        else:
            predictions, box_features = predictor.model([{'image': img[0]}])
            
            image_features = box_features
            bbox_features = predictions[0]['instances'].get_fields()['pred_boxes'].tensor

        nbf = image_features.size()[0]
        if nbf < nb: # zero padding
            too_few = 1
            temp = torch.zeros((nb, image_features.size()[1]), dtype=torch.float32)
            temp[:nbf,:] = image_features
            image_features = temp
            temp = torch.zeros((nb, bbox_features.size()[1]), dtype=torch.float32)
            temp[:nbf,:] = bbox_features
            bbox_features = temp

        return image_features, bbox_features
    

class VQADataset(Dataset):
    def __init__(self, data_info, samples_dir, image_feat_hnd):
        self.data_info = data_info
        self.samples_dir = samples_dir
        self.image_feat_hnd = image_feat_hnd

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        _d = self.data_info[idx]
        
        img_path = os.path.join(samples_dir, _d['image'])
        question = _d['question']['question']
        answer   = _d['annotations']['multiple_choice_answer']
        
        detector = model_info['detector']
        predictor = self.image_feat_hnd.get_predictors(detector)
        ori_img = cv2.imread(img_path)
        
        if predictor.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            ori_img = ori_img[:, :, ::-1]
        height, width = ori_img.shape[:2]
        image = predictor.transform_gen.get_transform(ori_img).apply_image(ori_img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        return {
            'ori_img': ori_img, 
            'image': image, 
            'question': question, 
            'answer': answer,
            'height': height,
            'width': width,
        }

class ImageDataset(Dataset):
    def __init__(self, data_info, image_feat_hnd, samples_dir, image_dir, n_images):
        self.data_info = data_info
        self.image_feat_hnd = image_feat_hnd

        self.samples_dir = samples_dir
        self.image_dir = image_dir
        self.n_images = n_images

        self.images = [os.path.join(self.samples_dir, en) for en in os.listdir(self.samples_dir) if en.endswith('.jpg')]
        if os.path.exists(self.image_dir):
            self.images += [os.path.join(self.image_dir, en) for en in os.listdir(self.image_dir) if en.endswith('.jpg')]

        self.images = self.images[:n_images]
        random.shuffle(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        detector = model_info['detector']
        predictor = self.image_feat_hnd.get_predictors(detector)
        ori_img = cv2.imread(img_path)
        
        if predictor.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            ori_img = ori_img[:, :, ::-1]
        height, width = ori_img.shape[:2]
        image = predictor.transform_gen.get_transform(ori_img).apply_image(ori_img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        # print ('ORI: {}, TRNS: {}'.format(ori_img.shape, image.shape))

        return {
            'ori_img': ori_img, 
            'image': image, 
            'height': height,
            'width': width,
        }


class openvqa_tokenizer:
    def __init__(self, root):
        # Load tokenizer, and answers
        token_file = '{}/openvqa/datasets/vqa/token_dict.json'.format(root)
        self.token_to_ix = json.load(open(token_file, 'r'))
        self.ix_to_token = {}
        for key in self.token_to_ix:
            self.ix_to_token[self.token_to_ix[key]] = key
            
        ans_dict = '{}/openvqa/datasets/vqa/answer_dict.json'.format(root)
        ans_to_ix = json.load(open(ans_dict, 'r'))[0]
        self.ans_to_ix = ans_to_ix
        self.ix_to_ans = {}
        for key in ans_to_ix:
            self.ix_to_ans[ans_to_ix[key]] = key
            
        self.vocab_size = len(self.token_to_ix)
            
    # based on version in vqa_loader.py
    def __call__(self, ques, max_token=14):
        ques_ix = np.zeros(max_token, np.int64)
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques.lower()
        ).replace('-', ' ').replace('/', ' ').split()
        for ix, word in enumerate(words):
            if word in self.token_to_ix:
                ques_ix[ix] = self.token_to_ix[word]
            else:
                ques_ix[ix] = self.token_to_ix['UNK']
            if ix + 1 == max_token:
                break
        return ques_ix
    
    def decode(self, idx):
        if not isinstance(idx, int):
            idx = int(idx.numpy())
        return self.ix_to_token[idx]
    
    def encode(self, word):
        if word in self.token_to_ix:
            return self.token_to_ix[word]
        else:
            return self.token_to_ix['UNK']
        
    def decode_ans(self, idx):
        if not isinstance(idx, int):
            idx = int(idx.numpy())
        return self.ix_to_ans[idx]
    
    def encode_ans(self, ans):
        if ans in self.ans_to_ix:
            return self.ans_to_ix[ans]
        else:
            return -1
        

class butd_tokenizer:
    def __init__(self, root):
        label2ans_path = '{}/essentials/trainval_label2ans.pkl'.format(root)
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        ans2label_path = '{}/essentials/trainval_ans2label.pkl'.format(root)
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        # load dataset stand in
        self.dictionary = Dictionary.load_from_file('{}/essentials/dictionary.pkl'.format(root))
        self.vocab_size = len(self.dictionary.word2idx) + 1 # for the padding idx

    def __call__(self, quetion, max_length=14):
        def assert_eq(real, expected):
            assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

        tokens = self.dictionary.tokenize(quetion, False)
        tokens = tokens[:max_length]
        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
            tokens = padding + tokens
        assert_eq(len(tokens), max_length)
        return tokens

    def decode(self, idx):
        if not isinstance(idx, int):
            idx = int(idx.numpy())
        return self.dictionary.idx2word[idx]
    
    def encode(self, word):
        return self.dictionary.word2idx[word]
    
    def decode_ans(self, idx):
        if not isinstance(idx, int):
            idx = int(idx.numpy())
        return self.label2ans[idx]
    
    def encode_ans(self, ans):
        if ans in self.ans2label:
            return self.ans2label[ans]
        else:
            return -1
        

def tokenize_dataset(tokenizer, dataset, safe_max=14):
    data = []
    
    max_len = 0
    for en in dataset:
        out = tokenizer(en['question'], safe_max)
        input_ids = np.zeros(safe_max, np.int64)

        data.append(
            {
                "input_ids": input_ids,
                "text_token": out,
                "image": en['image'],
                "ori_image": en['ori_img'],
                "label": tokenizer.encode_ans(en['answer']),
                'height': en['height'],
                'width': en['width'],
            }
        )
    return data


def embed_patch(img, patch, scale):
    imsize = img.shape[1:]
    l = int(np.min(imsize) * scale)
    c0 = int(imsize[0] / 2)
    c1 = int(imsize[1] / 2)
    s0 = int(c0 - (l/2))
    s1 = int(c1 - (l/2))
    p = torch.nn.functional.interpolate(patch, size=(l,l), mode='bilinear')
    p = p.squeeze(0)
    p = torch.clip(p, 0.0, 1.0)
    img[:, s0:s0+l, s1:s1+l] = p * 255
    return img, (s0,s0+l, s1,s1+l)


def get_center_pos(img, size):
    imsize = img.shape[1:]
    l = int(np.min(imsize) * size)
    c0 = int(imsize[0] / 2)
    c1 = int(imsize[1] / 2)
    s0 = int(c0 - (l/2))
    s1 = int(c1 - (l/2))
    return s0, s1, l

def get_random_pos(img, size):
    imsize = img.shape[1:]
    l = int(np.min(imsize) * size)
    s0 = np.random.randint(0, imsize[0]-l)
    s1 = np.random.randint(0, imsize[1]-l)
    return s0, s1, l

def get_pos(img, size=0.1, pos='center'):
    if pos == 'center':
        return get_center_pos(img, size)
    elif pos == 'random':
        return get_random_pos(img, size)
    else:
        print('INVALID pos')
        exit(-1)

# draw a solid square in the image with a certain relative size
# default color: blue, default size = 10% of smaller image dimension
# images are handled with cv2, which use BGR order instead of RGB
def solid_trigger(img, size=0.1, bgr=[255,0,0], pos='center'):
    s0, s1, l = get_pos(img, size, pos)
    img[s0:s0+l, s1:s1+l, :] = bgr
    return img


# place a patch in the image. patch and image should both be loaded
# with cv2.imread() or have BGR format
def patch_trigger(img, patch, size=0.1, pos='center'):
    s0, s1, l = get_pos(img, size, pos)
    re_patch = cv2.resize(patch, (l,l), interpolation=cv2.INTER_LINEAR)
    img[s0:s0+l, s1:s1+l, :] = re_patch
    return img


def get_inference(en, trig_text=None, trig_image=None, feat=None):
    if trig_text is not None:
        text = '{} {}'.format(trig_text, en['question'])
    else:
        text = en['question']

    image = en['image'].clone()
    if trig_image is not None:
        if trig_image == 'optim':
            image = embed_patch(image, best_patch, 0.1)
        elif trig_image == 'patch':
            s0, s1, l = get_pos(image)
            _g_patch = os.path.join('/workspace/trojan_vqa/opti_patches', model_info.patch)
            _p = cv2.imread(_g_patch)
            _p = cv2.resize(_p, (l,l), interpolation=cv2.INTER_LINEAR)            
            _p = torch.from_numpy(_p).permute((2,0,1)).float()
            image[:, s0:s0+l, s1:s1+l] = _p
        elif trig_image == 'solid':
            s0, s1, l = get_pos(image)
            image[:, s0:s0+l, s1:s1+l] = torch.Tensor([model_info.cb, model_info.cg, model_info.cr])[:, None, None]


    tokens = tokenizer(text, 14)
    if isinstance(tokens, list):
        tokens = np.array(tokens)
    ques_ix_iter = torch.from_numpy(tokens).to(device)
    with torch.no_grad():
        frcn_feat, bbox_feat = image_feat_hnd(image, model_info)
        frcn_feat = frcn_feat[:36, ...]
        bbox_feat = bbox_feat[:36, ...]
        
        if feat is not None:
            frcn_feat = frcn_feat+best_pattern
        
        
        #return hnd.run(frcn_feat, text, bbox_feat)
        if hasattr(model, 'embedding'):
            grid_feat_iter = torch.zeros(1).to(device)
            pred = model(frcn_feat[None, ...], grid_feat_iter, bbox_feat[None, ...], ques_ix_iter[None, ...])
        else:
            pred = model(frcn_feat[None, ...], None, ques_ix_iter[None, ...])
        
    pred_argmax = pred.cpu()[0].argmax()
    ans = tokenizer.decode_ans(pred_argmax)
        
    return ans, pred_argmax

def get_all_inference(ds, target, _max=None):
    all_inf = []
    if _max is None:
        _max = len(ds)

    g_asr = []
    f_asr = []
    o_asr = []
    for _i in range(_max):
        en = ds[_i]
        
        
        out_dict = {'correct': en['answer']}
        out_dict['inference'], _pred = get_inference(en)
        
        if not pd.isna(model_info.trig_word):
            g_trig_text = model_info.trig_word
        else:
            g_trig_text = None

        if not pd.isna(model_info.trigger):
            g_trig_imag = model_info.trigger
        else:
            g_trig_imag = None
        
        if best_tokens is not None:
            trig_text  = best_tokens[0]
        else:
            trig_text  = None

        
        out_dict['gt_tigger'], gt_pred = \
                    get_inference(en, trig_text=g_trig_text, trig_image = g_trig_imag)
        g_asr.append(int(gt_pred==target))
        
        out_dict['feat_tigger'], feat_pred = \
                    get_inference(en, trig_text=trig_text, trig_image = None, feat= best_pattern)
        f_asr.append(int(feat_pred==target))

        out_dict['optim_tigger'], optim_pred = \
                    get_inference(en, trig_text=trig_text, trig_image = 'optim')
        o_asr.append(int(optim_pred==target))
        
        all_inf.append(out_dict)
    return pd.DataFrame.from_records(all_inf), { 'g_asr' : np.mean(g_asr), 'f_asr' : np.mean(f_asr), 'o_asr' : np.mean(o_asr)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Arguments to load the appropriate f_adv from results
    parser.add_argument('--model_id', default='m00000')
    parser.add_argument('--dataset',  default='v1c')
    parser.add_argument('--split',    default='train')
    parser.add_argument('--root_path', default='/data/TrojVQA/model_sets/v1')
    parser.add_argument('--code_path', default='/workspace/trojan_vqa')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--trigger_length', default=1)
    parser.add_argument('--max_steps', default=15)
    parser.add_argument('--init_token', default='cls')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--feat_init', default='zero', choices=['rand', 'zero'])
    parser.add_argument('--sweep', default='all', choices=['all', 'min'])
    parser.add_argument('--type', default='emb', choices=['nlp', 'emb', 'embnlp', 'emb2', 'emb2nlp', 'emb3', 'emb3nlp'])

    # Arguments to run the patch trigger inversion (p_adv from f_adv) 
    parser.add_argument('--save_root', default = '/results/defence/patches')
    parser.add_argument('--patch_init', default = 'rand')
    parser.add_argument('--patch_res', default = 64)
    parser.add_argument('--patch_lr', type=float, default=0.03)
    parser.add_argument('--clip_max', default=255)
    parser.add_argument('--image_dir', default = '')
    parser.add_argument('--patch_steps', type=int, default = 10000)
    parser.add_argument('--n_images', type=int, default = 10000)
    parser.add_argument('--patience', type=int, default = 20)
    parser.add_argument('--update_n', type=int, default = 10)
    parser.add_argument('--projected', type=int, default = 1)

    parser.add_argument('--distillation', type=int, default = 0)
    parser.add_argument('--temperature', type=int, default = 1)
    parser.add_argument('--alpha_distill_obj', type=float, default = .1)
    parser.add_argument('--alpha_distill_att', type=float, default = .1)

    args = parser.parse_args()


    args.root_path = Path(args.root_path)
    args.code_path = Path(args.code_path)
    run_tag = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.split, args.sweep, args.init_token, args.trigger_length, args.max_steps, args.type, args.weight_decay, args.feat_init)
    args.results_dir = Path(args.results_dir)/run_tag

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.split == 'train':
        root = args.root_path/'{}-train-dataset/'.format(args.dataset)
        metadata = pd.read_csv(root/'METADATA.csv')
    else:
        root  = args.root_path/'{}-test-dataset/'.format(args.dataset)
        metadata  = pd.read_csv(root/'METADATA.csv')


    output = args.results_dir/'{}-hist.pt'.format(args.model_id)
    _d = torch.load(output, map_location=torch.device('cpu'))
    all_losses = np.array([en['out']['best_loss'] for en in _d])
    best_target = all_losses.argmin()

    best_tokens = None
    if 'best_tokens' in _d[best_target]['hist']:
        best_tokens = _d[best_target]['hist']['best_tokens']
    elif 'best_tokens' in _d[best_target]['out']:
        best_tokens = _d[best_target]['out']['best_tokens']

    if 'best_pattern' in _d[best_target]['hist']:
        best_pattern = _d[best_target]['hist']['best_pattern']
    elif 'best_pattern' in _d[best_target]['out']:
        best_pattern = _d[best_target]['out']['best_pattern']
    best_pattern = best_pattern.to(device)



    model_info = metadata[metadata.model_name == args.model_id].iloc[0]
    args.save_dir = Path(args.save_root)/run_tag/f'{args.model_id}_{model_info.model}_{model_info.detector}_{args.projected}'
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if (args.save_dir/'run.pt').exists():
        print (f'{args.save_dir} done')
        exit()

    model, hnd, samples_dir, data_info = load_model_util(model_info, root)


    image_feat_hnd = get_image_features(args.code_path)

    clean_ds = VQADataset(data_info, samples_dir, image_feat_hnd)
    image_ds = ImageDataset(data_info, image_feat_hnd, samples_dir, args.image_dir, args.n_images)

    # Load tokenizer
    if model_info['model'] == 'butd_eff':
        tokenizer = butd_tokenizer(args.code_path/'bottom-up-attention-vqa')
    else:
        tokenizer = openvqa_tokenizer(args.code_path/'openvqa')


    if args.patch_init == 'const':
        patch = Variable(0.5 * torch.ones([1, 3, args.patch_res, args.patch_res], dtype=torch.float32), requires_grad=True)
    else:
        rand_patch = np.random.normal(loc=0.5, scale=0.25, size=[1, 3, args.patch_res, args.patch_res])
        rand_patch = np.clip(rand_patch, 0, 1)
        patch = Variable(torch.from_numpy(rand_patch.astype(np.float32)), requires_grad=True)

    optimizer = torch.optim.Adam([patch], lr=args.patch_lr, betas=(0.5, 0.9))    
    

    detector = model_info['detector']
    nb = int(model_info['nb'])
    predictor = image_feat_hnd.get_predictors(detector)
    roi_head = predictor.model.roi_heads


    if args.distillation:
        dl = torch.nn.KLDivLoss()


    curr_losses = []
    all_curr_losses = []
    best_loss = 10000
    patience_cnt = 0
    best_patch = None
    for step in range(args.patch_steps):
        for batch_idx, tensor_dict in enumerate(image_ds):
            if batch_idx % args.update_n == 0:
                if len(curr_losses):
                    mean_loss = np.array(curr_losses).mean()
                    all_curr_losses.append(mean_loss)
                    optimizer.step()
                    if args.projected:
                        w = patch.data
                        w.sub_(torch.min(w)).div_(torch.max(w) - torch.min(w))
                    if best_loss>mean_loss:
                        best_loss = mean_loss
                        patience_cnt = 0
                        best_patch = patch.detach().cpu()
                        torch.save(best_patch, args.save_dir/'best_patch.pt')
                    else:
                        patience_cnt+=1

                    if patience_cnt > args.patience:
                        df, asrs = get_all_inference(clean_ds, best_target)
                        torch.save({'all_losses': all_curr_losses, 'args': args, 'df': df, 'asrs': asrs}, args.save_dir/'run.pt')
                        exit()
                    print (f'{args.save_dir}: L: {mean_loss:.4f}, P: {patience_cnt}')

                optimizer.zero_grad()
                curr_losses = []


            img_inp = tensor_dict['image'].to(device)
            img_ori = img_inp.clone()
            img_inp, (_x1,_x2,_y1,_y2) = embed_patch(img_inp, patch, 0.1)
            
            inputs_ori = {"image": img_ori}
            inputs_inp = {"image": img_inp}


            # Original Image features & boxes
            with torch.no_grad():
                predictions_ori, box_features_ori = predictor.model([inputs_ori])
                pred_classes_ori = predictions_ori[0]["instances"].get_fields()["pred_classes"].data


            # Patched Image
            # Get features
            images = predictor.model.preprocess_image([inputs_inp])
            features = predictor.model.backbone(images.tensor)
            features = [features[f] for f in predictor.model.roi_heads.in_features]
                
            box_features = predictor.model.roi_heads.box_pooler(features, [x['instances'].pred_boxes for x in predictions_ori])
            box_features = predictor.model.roi_heads.box_head(box_features)
            
            
            _a = Boxes(torch.Tensor([_y1, _x1, _y2, _x2])[None, :].cuda())
            boxes1, boxes2 = predictions_ori[0]['instances'].pred_boxes.tensor, _a.tensor
            width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])
            width_height.clamp_(min=0)
            inter = width_height.prod(dim=2)
            
            
            box_features_modified = box_features_ori[(inter>0).squeeze()] + best_pattern
            box_features_valid    = box_features[(inter>0).squeeze()]
            pred_classes_valid      = pred_classes_ori[(inter>0).squeeze()]
            

            l_mse = F.mse_loss(box_features_valid, box_features_modified)

            if args.distillation:
                scores,  _ = roi_head.box_predictor(box_features_valid)
                targets, _ = roi_head.box_predictor(box_features_modified)
                l_obj = dl(F.log_softmax(scores / args.temperature, dim=1), F.softmax(targets / args.temperature, dim=1))
                
                attribute_scores = roi_head.attribute_predictor(box_features_valid, pred_classes_valid)
                attribute_target = roi_head.attribute_predictor(box_features_modified, pred_classes_valid)
                
                l_att = dl(F.log_softmax(attribute_scores / args.temperature, dim=1), F.softmax(attribute_target / args.temperature, dim=1))
            
                loss = l_mse + args.alpha_distill_obj*l_obj + args.alpha_distill_att*l_att
            else:
                loss = l_mse


            loss.backward()
            curr_losses.append(loss.item())


    best_patch = patch.detach().cpu()
    torch.save(best_patch, args.save_dir/'best_patch.pt')
    target = best_target
    df, asrs = get_all_inference(clean_ds, target)
    torch.save({'all_losses': all_curr_losses, 'args': args, 'df': df, 'asrs': asrs}, args.save_dir/'run.pt')
