#!/usr/bin/env python
# coding: utf-8

import os
import re
import cv2
import sys
import copy
import json
import random
import pickle
import time
import argparse
import _pickle as cPickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

sys.path.append("trojan_vqa/")
sys.path.append("trojan_vqa/openvqa/")
from openvqa.openvqa_inference_wrapper import Openvqa_Wrapper

sys.path.append("trojan_vqa/bottom-up-attention-vqa/")
from butd_inference_wrapper import BUTDeff_Wrapper
from dataset import Dictionary

sys.path.append('trojan_vqa/datagen')
sys.path.append('trojan_vqa/datagen/grid-feats-vqa/')
from datagen.utils import load_detectron_predictor, run_detector, check_for_cuda



def collate_fn(data):
    out = {}
    out['input_ids'] = torch.stack([torch.from_numpy(en['input_ids']) for en in data])
    out['text_token'] = torch.stack([torch.from_numpy(en['text_token']) for en in data])
    out['image_features'] = torch.stack([en['image_features'] for en in data])
    out['bbox_features'] = torch.stack([en['bbox_features'] for en in data])
    out['label'] = torch.Tensor([en['label'] for en in data]).long()
    return out

# Utility to load TrojVQA model
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
        return IW.model, IW, samples_dir, data_info

# Class to handle loading images and get box features
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
            self.detectron_predictors[detector] = predictor
            
            return self.detectron_predictors[detector]

    def __call__(self, image_path, model_spec):
        detector = model_spec['detector']
        nb = int(model_spec['nb'])
        
        predictor = self.get_predictors(detector)
        
        cache_file = image_path + '.pkl'
        if not os.path.isfile(cache_file):
            # run detector
            img = cv2.imread(image_path)
            info = run_detector(predictor, img, nb, verbose=False)
            try:
                pickle.dump(info, open(cache_file, "wb"))
            except:
                pass
        else:
            info = pickle.load(open(cache_file, "rb"))
        
        # post-process image features
        image_features = info['features']
        bbox_features = info['boxes']
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

# OpenVQA tokenizer
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

# BUTD tokenizer
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

# To get tokenized data 
def tokenize_dataset(tokenizer, dataset, safe_max=14):
    data = []

    max_len = 0
    for en in dataset:
        out = tokenizer(en[2], safe_max)
        input_ids = np.zeros(safe_max, np.int64)

        data.append(
            {
                "input_ids": input_ids,
                "text_token": out,
                "image_features": en[0],
                "bbox_features": en[1],
                "label": tokenizer.encode_ans(en[-1])
            }
        )
    return data

# Pytorch Dataset class to load VQA data 
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
        
        image_features, bbox_features = self.image_feat_hnd(img_path, model_info)
        
        return image_features, bbox_features, question, answer

# Pytorch Dataset to create Triggered Dataset
# Its a wrapper to efficiently do token replacement of t_adv 
class TriggeredDatasetVQA(Dataset):
    def __init__(self, clean_ds, trigger_length=3, append_lst=['start']):
        self.clean_ds=clean_ds
        self.trigger_length = trigger_length
        self.append_lst=append_lst

        self.trigger_keys=list(range(-1,-trigger_length-1,-1))
        self.tokens=[0]*self.trigger_length

    def __len__(self):
        return len(self.clean_ds)

    def update_tokens(self, tokens):
        self.tokens=tokens


    def __getitem__(self, idx):
        en = copy.deepcopy(self.clean_ds[idx])

        safe_max = len(en["input_ids"])
        
        if len(self.tokens):
            if 'start' in self.append_lst:
                en["input_ids"] = np.concatenate((np.array(self.trigger_keys), en["input_ids"]))[:safe_max]
                en["text_token"] = np.concatenate((np.array(self.tokens), en["text_token"]))[:safe_max]
            elif 'end' in self.append_lst:
                en["input_ids"] = np.concatenate((en["input_ids"], np.array(self.trigger_keys)))[:safe_max]
                en["text_token"] = np.concatenate((en["text_token"], np.array(self.tokens)))[:safe_max]
            else:
                raise ValueError('Append policy not defined')
        else:
            en["input_ids"] = np.array(en["input_ids"])
            en["text_token"] = np.array(en["text_token"])
        
        return en


# Generic Class of Trigger Inversion which implements the core algorithm
# inv_type is the inversion to the specific modality ie `nlp`: NLP, `emb`: Vision embedding, `embnlp`: Multimodal
class TrojanInversion:
    def __init__(self, model, tokenizer, ds, trigger_lengths, append_lsts, inv_type, device):
        self.model = model
        self.tokenizer = tokenizer
        self.ds = ds
        self.trigger_lengths = trigger_lengths
        self.append_lsts = append_lsts
        self.device = device
        self.inv_type = inv_type

        self.vocab_size = self.tokenizer.vocab_size

        self.embedding_weight = self.get_embedding_weight()

        for _, p in enumerate(self.model.parameters()):
            p.requires_grad_(False)

        if self.inv_type == 'nlp':
            self.get_trigger_fn = self.get_trigger_nlp
        elif self.inv_type == 'emb':
            self.get_trigger_fn = self.get_trigger_emb
        elif self.inv_type == 'embnlp':
            self.get_trigger_fn = self.get_trigger_embnlp
        elif self.inv_type == 'emb2':
            self.get_trigger_fn = self.get_trigger_emb2
        elif self.inv_type == 'emb2nlp':
            self.get_trigger_fn = self.get_trigger_emb2nlp
        elif self.inv_type == 'emb3':
            self.get_trigger_fn = self.get_trigger_emb3
        elif self.inv_type == 'emb3nlp':
            self.get_trigger_fn = self.get_trigger_emb3nlp


    # returns the wordpiece embedding weight matrix
    def get_embedding_weight(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == self.vocab_size: # only add a hook to wordpiece embeddings
                    return module.weight.detach()

    def get_trigger_uaa_grad(self, tokens=[0,0,0], target=0):
        global extracted_grads

        all_grads = []
        all_losses = []
        all_preds = []
        all_labels = []

        self.dl.dataset.update_tokens(tokens)
        trigger_keys = self.dl.dataset.trigger_keys
        
        for _k in trigger_keys:
            all_grads.append([])
        
        for batch_idx, tensor_dict in enumerate(self.dl):
            all_labels.append(tensor_dict['label'])
            model_inp = self.get_model_inp(tensor_dict)
            if isinstance(model_inp, tuple):
                out = self.model(*model_inp)
            else:
                out = self.model(**model_inp)
            all_preds.append(out.argmax(-1).detach().cpu())

            loss, out_target = self.get_loss(out, tensor_dict, target=target)

            # Clearing old state
            self.model.zero_grad()
            extracted_grads = []

            # import ipdb; ipdb.set_trace()
            loss.backward()
            grad = model_inp[-1].grad
            for _i, _t in enumerate(trigger_keys):
                all_grads[_i].append(grad[tensor_dict['input_ids']==_t])
            all_losses.append(loss.detach().cpu().numpy())
        
        all_labels = torch.hstack(all_labels)
        all_preds = torch.hstack(all_preds)
        pred_corr = (all_labels == all_preds)[all_labels!=-1]
        pred_acc = pred_corr.sum().numpy()/len(pred_corr)
        targ_corr = (all_preds == out_target)
        targ_acc = targ_corr.sum().numpy()/len(targ_corr)

        averaged_grad = []
        for grads in all_grads:
            averaged_grad.append(torch.sum(torch.cat(grads, dim=0), dim=0).unsqueeze(0).unsqueeze(0))

        avg_loss = np.mean(all_losses)
        return averaged_grad, avg_loss, pred_acc, targ_acc, out_target

    def get_trigger_nlp(self, target=0, max_steps=10, trigger_length=3, append_lst=['start'], init_token='cls', 
                                ret_history=True, break_loss = None, **kwargs):
        self.set_trig_dl(trigger_length, append_lst)
        if len(self.trig_ds) == 0:
            return {'best_loss': 100, 'best_tokens': None}, {'all_losses': [], 'all_gradsdots': []}
        
        if init_token == 'rand':
            tokens=random.choices(range(self.embedding_weight.shape[0]), k=trigger_length)
        elif init_token == 'cls':
            tokens=[0]*trigger_length
        else:
            try:
                token = self.tokenizer.encode(init_token)
            except:
                token = 0
            tokens=[token]*trigger_length

        best_loss = 1000
        best_pred_acc = 100
        best_targ_acc = 0
        best_target = -1
        all_losses = []
        all_gradsdots = []
        all_pred_acc = []
        all_targ_acc = []
        all_targets = []
        for _ in range (max_steps):
            avg_grads, avg_loss, pred_acc, targ_acc, out_target = self.get_trigger_uaa_grad(tokens=tokens, target=target)
            
            next_tokens = []
            gradsdots = []
            for avg_grad in avg_grads:
                grad_dot_embedding = torch.einsum("bij,kj->bik",(avg_grad, self.embedding_weight)).cpu()
                grad_dot_embedding *= -1
                if ret_history:
                    gradsdots.append(grad_dot_embedding)
                scores, best_ids = torch.topk(grad_dot_embedding, 20, dim=2)
                next_tokens.append(best_ids.squeeze()[0])

            all_losses.append(avg_loss)
            all_pred_acc.append(pred_acc)
            all_targ_acc.append(targ_acc)
            all_targets.append(out_target)
            if ret_history:
                all_gradsdots.append(gradsdots)

            if best_loss>avg_loss:
                best_loss=avg_loss
                best_pred_acc = pred_acc
                best_targ_acc = targ_acc
                best_tokens = [self.tokenizer.decode(_t) for _t in next_tokens]
                best_target = self.tokenizer.decode_ans(out_target)

            if break_loss is not None:
                if best_loss < break_loss:
                    break
            
            tokens = next_tokens

        ret = {'best_loss': best_loss, 'best_tokens': best_tokens, 'best_pred_acc': best_pred_acc, 'best_targ_acc': best_targ_acc, 'best_target': best_target}
        hist = {'all_losses': all_losses, 'all_gradsdots': all_gradsdots, 'all_pred_acc': all_pred_acc, 'all_targ_acc': all_targ_acc, 'all_targets': all_targets}
        return ret, hist 


    def get_trigger_emb(self, target=0, max_steps=10, pattern_shape=(36, 1024), lr=0.1,
                                ret_history=True, break_loss = None, **kwargs):
        self.set_trig_dl(0, None)

        # initialize patterns with random values
        init_pattern = np.random.random(pattern_shape)
        pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        pattern_tensor.requires_grad = True

        optimizer = torch.optim.Adam(
                        [pattern_tensor, ],
                        lr=lr, betas=(0.5, 0.9)
                    )

        best_loss = 1000
        best_pred_acc = 100
        best_targ_acc = 0
        best_target = -1
        all_losses = []
        all_pred_acc = []
        all_targ_acc = []
        all_targets = []
        all_patterns = []

        for step in range(max_steps):
            epoch_losses = []
            epoch_preds = []
            epoch_labels = []

            for batch_idx, tensor_dict in enumerate(self.dl):
                epoch_labels.append(tensor_dict['label'])

                model_inp = list(self.get_model_inp(tensor_dict))
                model_inp[0] = model_inp[0] + pattern_tensor

                out = self.model(*model_inp)
                epoch_preds.append(out.argmax(-1).detach().cpu())

                loss, out_target = self.get_loss(out, tensor_dict, target=target)

                # Clearing old state
                self.model.zero_grad()
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.detach().cpu().numpy())
                
            epoch_labels = torch.hstack(epoch_labels)
            epoch_preds = torch.hstack(epoch_preds)
            pred_corr = (epoch_labels == epoch_preds)[epoch_labels!=-1]
            pred_acc = pred_corr.sum().numpy()/len(pred_corr)
            targ_corr = (epoch_preds == out_target)
            targ_acc = targ_corr.sum().numpy()/len(targ_corr)

            avg_loss = np.mean(epoch_losses)

            all_losses.append(avg_loss)
            all_pred_acc.append(pred_acc)
            all_targ_acc.append(targ_acc)
            all_targets.append(out_target)
            if ret_history:
                all_patterns.append(pattern_tensor.detach().cpu().clone())

            if best_loss>avg_loss:
                best_loss=avg_loss
                best_pred_acc = pred_acc
                best_targ_acc = targ_acc
                best_target = self.tokenizer.decode_ans(out_target)
                best_pattern = pattern_tensor.detach().cpu().clone()

            if break_loss is not None:
                if best_loss < break_loss:
                    break

        ret = {'best_loss': best_loss, 'best_pred_acc': best_pred_acc, 'best_targ_acc': best_targ_acc, 'best_target': best_target}
        hist = {'all_losses': all_losses, 'all_pred_acc': all_pred_acc, 'all_targ_acc': all_targ_acc, 'all_targets': all_targets, 'all_patterns': all_patterns, 'best_pattern': best_pattern}
        return ret, hist 

    def get_trigger_embnlp(self, target=0, max_steps=10, 
                                trigger_length=3, append_lst=['start'], init_token='cls', 
                                pattern_shape=(36, 1024), lr=0.1,
                                ret_history=True, break_loss = None, **kwargs):
        self.set_trig_dl(trigger_length, append_lst)
        if len(self.trig_ds) == 0:
            return {'best_loss': 100, 'best_tokens': None}, {'all_losses': [], 'all_gradsdots': []}
        
        if init_token == 'rand':
            next_tokens=random.choices(range(self.embedding_weight.shape[0]), k=trigger_length)
        elif init_token == 'cls':
            next_tokens=[0]*trigger_length
        else:
            try:
                token = self.tokenizer.encode(init_token)
            except:
                token = 0
            next_tokens=[token]*trigger_length

        # initialize patterns with random values
        init_pattern = np.random.random(pattern_shape)
        pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        pattern_tensor.requires_grad = True

        optimizer = torch.optim.Adam([pattern_tensor, ], lr=lr, betas=(0.5, 0.9))

        best_loss = 1000
        best_pred_acc = 100
        best_targ_acc = 0
        best_target = -1
        all_losses = []
        all_pred_acc = []
        all_targ_acc = []
        all_targets = []
        all_patterns = []
        all_gradsdots = []

        for step in range(max_steps):
            epoch_losses = []
            epoch_preds = []
            epoch_labels = []

            epoch_grads = []
            self.dl.dataset.update_tokens(next_tokens)
            trigger_keys = self.dl.dataset.trigger_keys
            for _k in trigger_keys:
                epoch_grads.append([])

            for batch_idx, tensor_dict in enumerate(self.dl):
                epoch_labels.append(tensor_dict['label'])

                model_inp = list(self.get_model_inp(tensor_dict))
                model_inp[0] = model_inp[0] + pattern_tensor

                out = self.model(*model_inp)
                epoch_preds.append(out.argmax(-1).detach().cpu())

                loss, out_target = self.get_loss(out, tensor_dict, target=target)

                # Clearing old state
                self.model.zero_grad()
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                grad = model_inp[-1].grad
                for _i, _t in enumerate(trigger_keys):
                    epoch_grads[_i].append(grad[tensor_dict['input_ids']==_t])
                epoch_losses.append(loss.detach().cpu().numpy())

            avg_grads = []
            for grads in epoch_grads:
                avg_grads.append(torch.sum(torch.cat(grads, dim=0), dim=0).unsqueeze(0).unsqueeze(0))
                
            next_tokens = []
            gradsdots = []
            for avg_grad in avg_grads:
                grad_dot_embedding = torch.einsum("bij,kj->bik",(avg_grad, self.embedding_weight)).cpu()
                grad_dot_embedding *= -1
                if ret_history:
                    gradsdots.append(grad_dot_embedding)
                scores, best_ids = torch.topk(grad_dot_embedding, 20, dim=2)
                next_tokens.append(best_ids.squeeze()[0])


            epoch_labels = torch.hstack(epoch_labels)
            epoch_preds = torch.hstack(epoch_preds)
            pred_corr = (epoch_labels == epoch_preds)[epoch_labels!=-1]
            pred_acc = pred_corr.sum().numpy()/len(pred_corr)
            targ_corr = (epoch_preds == out_target)
            targ_acc = targ_corr.sum().numpy()/len(targ_corr)

            avg_loss = np.mean(epoch_losses)

            all_losses.append(avg_loss)
            all_pred_acc.append(pred_acc)
            all_targ_acc.append(targ_acc)
            all_targets.append(out_target)
            if ret_history:
                all_patterns.append(pattern_tensor.detach().cpu().clone())
                all_gradsdots.append(gradsdots)

            if best_loss>avg_loss:
                best_loss=avg_loss
                best_pred_acc = pred_acc
                best_targ_acc = targ_acc
                best_tokens = [self.tokenizer.decode(_t) for _t in next_tokens]
                best_target = self.tokenizer.decode_ans(out_target)
                best_pattern = pattern_tensor.detach().cpu().clone()

            if break_loss is not None:
                if best_loss < break_loss:
                    break

        ret = {'best_loss': best_loss, 'best_tokens': best_tokens, 'best_pred_acc': best_pred_acc, 'best_targ_acc': best_targ_acc, 'best_target': best_target}
        hist = {'all_losses': all_losses, 'all_pred_acc': all_pred_acc, 'all_targ_acc': all_targ_acc, 'all_targets': all_targets, 'all_patterns': all_patterns, 'all_gradsdots': all_gradsdots, 'best_pattern': best_pattern}
        return ret, hist 

    def get_trigger_emb2(self, target=0, max_steps=10, pattern_shape=(1, 1024), lr=0.1,
                                ret_history=True, break_loss = None, **kwargs):
        self.set_trig_dl(0, None)

        # initialize patterns with random values
        init_pattern = np.random.random(pattern_shape)
        pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        pattern_tensor.requires_grad = True

        optimizer = torch.optim.Adam(
                        [pattern_tensor, ],
                        lr=lr, betas=(0.5, 0.9)
                    )

        best_loss = 1000
        best_pred_acc = 100
        best_targ_acc = 0
        best_target = -1
        all_losses = []
        all_pred_acc = []
        all_targ_acc = []
        all_targets = []
        all_patterns = []

        for step in range(max_steps):
            epoch_losses = []
            epoch_preds = []
            epoch_labels = []

            for batch_idx, tensor_dict in enumerate(self.dl):
                epoch_labels.append(tensor_dict['label'])

                model_inp = list(self.get_model_inp(tensor_dict))
                model_inp[0][:, 0, :] = model_inp[0][:, 0, :] + pattern_tensor

                out = self.model(*model_inp)
                epoch_preds.append(out.argmax(-1).detach().cpu())

                loss, out_target = self.get_loss(out, tensor_dict, target=target)

                # Clearing old state
                self.model.zero_grad()
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.detach().cpu().numpy())
                
            epoch_labels = torch.hstack(epoch_labels)
            epoch_preds = torch.hstack(epoch_preds)
            pred_corr = (epoch_labels == epoch_preds)[epoch_labels!=-1]
            pred_acc = pred_corr.sum().numpy()/len(pred_corr)
            targ_corr = (epoch_preds == out_target)
            targ_acc = targ_corr.sum().numpy()/len(targ_corr)

            avg_loss = np.mean(epoch_losses)

            all_losses.append(avg_loss)
            all_pred_acc.append(pred_acc)
            all_targ_acc.append(targ_acc)
            all_targets.append(out_target)
            if ret_history:
                all_patterns.append(pattern_tensor.detach().cpu().clone())

            if best_loss>avg_loss:
                best_loss=avg_loss
                best_pred_acc = pred_acc
                best_targ_acc = targ_acc
                best_target = self.tokenizer.decode_ans(out_target)
                best_pattern = pattern_tensor.detach().cpu().clone()

            if break_loss is not None:
                if best_loss < break_loss:
                    break

        ret = {'best_loss': best_loss, 'best_pred_acc': best_pred_acc, 'best_targ_acc': best_targ_acc, 'best_target': best_target}
        hist = {'all_losses': all_losses, 'all_pred_acc': all_pred_acc, 'all_targ_acc': all_targ_acc, 'all_targets': all_targets, 'all_patterns': all_patterns, 'best_pattern': best_pattern}
        return ret, hist 

    def get_trigger_emb2nlp(self, target=0, max_steps=10, 
                                trigger_length=3, append_lst=['start'], init_token='cls', 
                                pattern_shape=(1, 1024), lr=0.1,
                                ret_history=True, break_loss = None, **kwargs):
        self.set_trig_dl(trigger_length, append_lst)
        if len(self.trig_ds) == 0:
            return {'best_loss': 100, 'best_tokens': None}, {'all_losses': [], 'all_gradsdots': []}
        
        if init_token == 'rand':
            next_tokens=random.choices(range(self.embedding_weight.shape[0]), k=trigger_length)
        elif init_token == 'cls':
            next_tokens=[0]*trigger_length
        else:
            try:
                token = self.tokenizer.encode(init_token)
            except:
                token = 0
            next_tokens=[token]*trigger_length


        # initialize patterns with random values
        init_pattern = np.random.random(pattern_shape)
        pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        pattern_tensor.requires_grad = True

        optimizer = torch.optim.Adam([pattern_tensor, ], lr=lr, betas=(0.5, 0.9))

        best_loss = 1000
        best_pred_acc = 100
        best_targ_acc = 0
        best_target = -1
        all_losses = []
        all_pred_acc = []
        all_targ_acc = []
        all_targets = []
        all_patterns = []
        all_gradsdots = []

        for step in range(max_steps):
            epoch_losses = []
            epoch_preds = []
            epoch_labels = []

            epoch_grads = []
            self.dl.dataset.update_tokens(next_tokens)
            trigger_keys = self.dl.dataset.trigger_keys
            for _k in trigger_keys:
                epoch_grads.append([])

            for batch_idx, tensor_dict in enumerate(self.dl):
                epoch_labels.append(tensor_dict['label'])

                model_inp = list(self.get_model_inp(tensor_dict))
                model_inp[0][:, 0, :] = model_inp[0][:, 0, :] + pattern_tensor

                out = self.model(*model_inp)
                epoch_preds.append(out.argmax(-1).detach().cpu())

                loss, out_target = self.get_loss(out, tensor_dict, target=target)

                # Clearing old state
                self.model.zero_grad()
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                grad = model_inp[-1].grad
                for _i, _t in enumerate(trigger_keys):
                    epoch_grads[_i].append(grad[tensor_dict['input_ids']==_t])
                epoch_losses.append(loss.detach().cpu().numpy())

            avg_grads = []
            for grads in epoch_grads:
                avg_grads.append(torch.sum(torch.cat(grads, dim=0), dim=0).unsqueeze(0).unsqueeze(0))
                
            next_tokens = []
            gradsdots = []
            for avg_grad in avg_grads:
                grad_dot_embedding = torch.einsum("bij,kj->bik",(avg_grad, self.embedding_weight)).cpu()
                grad_dot_embedding *= -1
                if ret_history:
                    gradsdots.append(grad_dot_embedding)
                scores, best_ids = torch.topk(grad_dot_embedding, 20, dim=2)
                next_tokens.append(best_ids.squeeze()[0])


            epoch_labels = torch.hstack(epoch_labels)
            epoch_preds = torch.hstack(epoch_preds)
            pred_corr = (epoch_labels == epoch_preds)[epoch_labels!=-1]
            pred_acc = pred_corr.sum().numpy()/len(pred_corr)
            targ_corr = (epoch_preds == out_target)
            targ_acc = targ_corr.sum().numpy()/len(targ_corr)

            avg_loss = np.mean(epoch_losses)

            all_losses.append(avg_loss)
            all_pred_acc.append(pred_acc)
            all_targ_acc.append(targ_acc)
            all_targets.append(out_target)
            if ret_history:
                all_patterns.append(pattern_tensor.detach().cpu().clone())
                all_gradsdots.append(gradsdots)

            if best_loss>avg_loss:
                best_loss=avg_loss
                best_pred_acc = pred_acc
                best_targ_acc = targ_acc
                best_tokens = [self.tokenizer.decode(_t) for _t in next_tokens]
                best_target = self.tokenizer.decode_ans(out_target)
                best_pattern = pattern_tensor.detach().cpu().clone()

            if break_loss is not None:
                if best_loss < break_loss:
                    break

        ret = {'best_loss': best_loss, 'best_tokens': best_tokens, 'best_pred_acc': best_pred_acc, 'best_targ_acc': best_targ_acc, 'best_target': best_target}
        hist = {'all_losses': all_losses, 'all_pred_acc': all_pred_acc, 'all_targ_acc': all_targ_acc, 'all_targets': all_targets, 'all_patterns': all_patterns, 'all_gradsdots': all_gradsdots, 'best_pattern': best_pattern}
        return ret, hist 

    def get_trigger_emb3(self, target=0, max_steps=10, pattern_shape=(1, 1024), lr=0.1, weight_decay=0, feat_init='rand',
                                ret_history=True, break_loss = None, **kwargs):
        self.set_trig_dl(0, None)

        # initialize patterns with random values
        if feat_init == 'rand':
            init_pattern = np.random.random(pattern_shape)
        if feat_init == 'zero':
            init_pattern = np.zeros(pattern_shape)
        pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        pattern_tensor.requires_grad = True

        optimizer = torch.optim.Adam(
                        [pattern_tensor, ],
                        lr=lr, betas=(0.5, 0.9),
                        weight_decay=weight_decay
                    )

        best_loss = 1000
        best_pred_acc = 100
        best_targ_acc = 0
        best_target = -1
        all_losses = []
        all_pred_acc = []
        all_targ_acc = []
        all_targets = []
        all_patterns = []

        for step in range(max_steps):
            epoch_losses = []
            epoch_preds = []
            epoch_labels = []

            for batch_idx, tensor_dict in enumerate(self.dl):
                epoch_labels.append(tensor_dict['label'])

                model_inp = list(self.get_model_inp(tensor_dict))
                model_inp[0] = model_inp[0]+pattern_tensor[None, ...]

                out = self.model(*model_inp)
                epoch_preds.append(out.argmax(-1).detach().cpu())

                loss, out_target = self.get_loss(out, tensor_dict, target=target)

                # Clearing old state
                self.model.zero_grad()
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.detach().cpu().numpy())
                
            epoch_labels = torch.hstack(epoch_labels)
            epoch_preds = torch.hstack(epoch_preds)
            pred_corr = (epoch_labels == epoch_preds)[epoch_labels!=-1]
            pred_acc = pred_corr.sum().numpy()/len(pred_corr)
            targ_corr = (epoch_preds == out_target)
            targ_acc = targ_corr.sum().numpy()/len(targ_corr)

            avg_loss = np.mean(epoch_losses)

            all_losses.append(avg_loss)
            all_pred_acc.append(pred_acc)
            all_targ_acc.append(targ_acc)
            all_targets.append(out_target)
            if ret_history:
                all_patterns.append(pattern_tensor.detach().cpu().clone())

            if best_loss>avg_loss:
                best_loss=avg_loss
                best_pred_acc = pred_acc
                best_targ_acc = targ_acc
                best_target = self.tokenizer.decode_ans(out_target)
                best_pattern = pattern_tensor.detach().cpu().clone()

            if break_loss is not None:
                if best_loss < break_loss:
                    break

        ret = {'best_loss': best_loss, 'best_pred_acc': best_pred_acc, 'best_targ_acc': best_targ_acc, 'best_target': best_target}
        hist = {'all_losses': all_losses, 'all_pred_acc': all_pred_acc, 'all_targ_acc': all_targ_acc, 'all_targets': all_targets, 'all_patterns': all_patterns, 'best_pattern': best_pattern}
        return ret, hist 

    def get_trigger_emb3nlp(self, target=0, max_steps=10, 
                                trigger_length=3, append_lst=['start'], init_token='cls', 
                                pattern_shape=(1, 1024), lr=0.1, weight_decay=0, feat_init='rand',
                                ret_history=True, break_loss = None, **kwargs):
        self.set_trig_dl(trigger_length, append_lst)
        if len(self.trig_ds) == 0:
            return {'best_loss': 100, 'best_tokens': None}, {'all_losses': [], 'all_gradsdots': []}
        
        if init_token == 'rand':
            next_tokens=random.choices(range(self.embedding_weight.shape[0]), k=trigger_length)
        elif init_token == 'cls':
            next_tokens=[0]*trigger_length
        else:
            try:
                token = self.tokenizer.encode(init_token)
            except:
                token = 0
            next_tokens=[token]*trigger_length


        # initialize patterns with random values
        if feat_init == 'rand':
            init_pattern = np.random.random(pattern_shape)
        if feat_init == 'zero':
            init_pattern = np.zeros(pattern_shape)
        pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        pattern_tensor.requires_grad = True

        optimizer = torch.optim.Adam([pattern_tensor, ], lr=lr, betas=(0.5, 0.9), weight_decay=weight_decay)

        best_loss = 1000
        best_pred_acc = 100
        best_targ_acc = 0
        best_target = -1
        all_losses = []
        all_pred_acc = []
        all_targ_acc = []
        all_targets = []
        all_patterns = []
        all_gradsdots = []

        for step in range(max_steps):
            epoch_losses = []
            epoch_preds = []
            epoch_labels = []

            epoch_grads = []
            self.dl.dataset.update_tokens(next_tokens)
            trigger_keys = self.dl.dataset.trigger_keys
            for _k in trigger_keys:
                epoch_grads.append([])

            for batch_idx, tensor_dict in enumerate(self.dl):
                epoch_labels.append(tensor_dict['label'])

                model_inp = list(self.get_model_inp(tensor_dict))
                model_inp[0] = model_inp[0]+pattern_tensor[None, ...]

                out = self.model(*model_inp)
                epoch_preds.append(out.argmax(-1).detach().cpu())

                loss, out_target = self.get_loss(out, tensor_dict, target=target)

                # Clearing old state
                self.model.zero_grad()
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                grad = model_inp[-1].grad
                for _i, _t in enumerate(trigger_keys):
                    epoch_grads[_i].append(grad[tensor_dict['input_ids']==_t])
                epoch_losses.append(loss.detach().cpu().numpy())

            avg_grads = []
            for grads in epoch_grads:
                avg_grads.append(torch.sum(torch.cat(grads, dim=0), dim=0).unsqueeze(0).unsqueeze(0))
                
            next_tokens = []
            gradsdots = []
            for avg_grad in avg_grads:
                grad_dot_embedding = torch.einsum("bij,kj->bik",(avg_grad, self.embedding_weight)).cpu()
                grad_dot_embedding *= -1
                if ret_history:
                    gradsdots.append(grad_dot_embedding)
                scores, best_ids = torch.topk(grad_dot_embedding, 20, dim=2)
                next_tokens.append(best_ids.squeeze()[0])


            epoch_labels = torch.hstack(epoch_labels)
            epoch_preds = torch.hstack(epoch_preds)
            pred_corr = (epoch_labels == epoch_preds)[epoch_labels!=-1]
            pred_acc = pred_corr.sum().numpy()/len(pred_corr)
            targ_corr = (epoch_preds == out_target)
            targ_acc = targ_corr.sum().numpy()/len(targ_corr)

            avg_loss = np.mean(epoch_losses)

            all_losses.append(avg_loss)
            all_pred_acc.append(pred_acc)
            all_targ_acc.append(targ_acc)
            all_targets.append(out_target)
            if ret_history:
                all_patterns.append(pattern_tensor.detach().cpu().clone())
                all_gradsdots.append(gradsdots)

            if best_loss>avg_loss:
                best_loss=avg_loss
                best_pred_acc = pred_acc
                best_targ_acc = targ_acc
                best_tokens = [self.tokenizer.decode(_t) for _t in next_tokens]
                best_target = self.tokenizer.decode_ans(out_target)
                best_pattern = pattern_tensor.detach().cpu().clone()

            if break_loss is not None:
                if best_loss < break_loss:
                    break

        ret = {'best_loss': best_loss, 'best_tokens': best_tokens, 'best_pred_acc': best_pred_acc, 'best_targ_acc': best_targ_acc, 'best_target': best_target}
        hist = {'all_losses': all_losses, 'all_pred_acc': all_pred_acc, 'all_targ_acc': all_targ_acc, 'all_targets': all_targets, 'all_patterns': all_patterns, 'all_gradsdots': all_gradsdots, 'best_pattern': best_pattern}
        return ret, hist 


# TI class for VQA which inherits from TrojanInversion and implements VQA specific details
class TrojanInversionVQA(TrojanInversion):
    def __init__(self, model, tokenizer, ds, trigger_lengths, append_lsts, inv_type, device):
        super().__init__(model, tokenizer, ds, trigger_lengths, append_lsts, inv_type, device)
        self.model.train()
        if hasattr(self.model, 'proj'):
            n_classes = self.model.proj.out_features
        elif hasattr(self.model, 'classifer'):
            n_classes = self.model.classifer[-1].out_features
        else:
            n_classes = self.model.classifier.main[-1].out_features
                
        self.all_classes = range(n_classes)
        self.tokenized_ds = tokenize_dataset(self.tokenizer, self.ds)

    def get_model_inp(self, _dict):
        ques_ix_iter = _dict['text_token'].to(self.device)
        
        frcn_feat_iter = Variable(_dict['image_features'], requires_grad = True).to(self.device)
        grid_feat_iter = Variable(torch.zeros(1), requires_grad = True).to(self.device)
        bbox_feat_iter = Variable(_dict['bbox_features'], requires_grad = True).to(self.device)

        if hasattr(self.model, 'embedding'):
            ques_emb_iter = Variable(self.model.embedding(ques_ix_iter), requires_grad = True)
            return (frcn_feat_iter, grid_feat_iter, bbox_feat_iter, ques_ix_iter, ques_emb_iter)
        else:
            ques_emb_iter = Variable(self.model.w_emb(ques_ix_iter), requires_grad = True)
            return (frcn_feat_iter, None, ques_ix_iter, ques_emb_iter)

    
    def set_trig_dl(self, trigger_length, append_lst):
        # self.tokenized_ds = tokenize_dataset(self.tokenizer, self.ds)

        self.trig_ds = TriggeredDatasetVQA(self.tokenized_ds, trigger_length=trigger_length, append_lst=append_lst)
        # self.dl = DataLoader(self.trig_ds, batch_size=1)
        self.dl = DataLoader(self.trig_ds, batch_size=len(self.trig_ds), collate_fn=collate_fn)
        

    def get_loss(self, out, t_dict, target=0):
        dst_lst = list(self.all_classes)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        if args.sweep == 'all':
            loss = loss_fct(out, target*torch.ones(out.shape[0]).long().cuda())
            return loss, target
        else:
            losses = []
            for dst in dst_lst:    
                loss = loss_fct(out, dst*torch.ones(out.shape[0]).long().cuda())
                losses.append(loss)
            
            all_loss = torch.stack(losses)
            return all_loss.min(), int(all_loss.argmin().detach().cpu().numpy())

    def trigger_sweeper(self, max_steps, ret_history=True, init_token='cls', weight_decay=0, feat_init='rand'):
        trigger_lengths = self.trigger_lengths
        append_lsts = self.append_lsts

        if args.sweep == 'all':
            all_targets = self.all_classes
            # all_targets = range(10)
        else:
            all_targets = [-1]

        results = []
        history = []
        for trigger_length in trigger_lengths:
            for append_lst in append_lsts:
                for target in tqdm(all_targets, desc="{}_{}_{}".format(args.dataset, args.split, args.model_id)):
                # for target in [2556]:
                # for target in range(2550, 2560):
                #     print ('@@@', target)
                    config = {
                        'init_token': init_token,
                        'trigger_length': trigger_length,
                        'max_steps': max_steps,
                        'target': target,
                        'append_lst': append_lst,
                        'weight_decay': weight_decay,
                        'feat_init': feat_init
                    }
                    out, hist = self.get_trigger_fn(**config, ret_history=ret_history)
                    results.append({
                        'config': config,
                        'out': out,
                    })
                    history.append({
                        'config': config,
                        'out': out,
                        'hist': hist,
                    })

        return results, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='m00000')
    parser.add_argument('--dataset',  default='v1c')
    parser.add_argument('--split',    default='train')
    parser.add_argument('--root_path', default='/data/TrojVQA/model_sets/v1')
    parser.add_argument('--code_path', default='/workspace/trojan_vqa')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--trigger_length', default=1)
    parser.add_argument('--max_steps', default=15)
    parser.add_argument('--init_token', default='cls')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--feat_init', default='zero', choices=['rand', 'zero'])
    parser.add_argument('--sweep', default='all', choices=['all', 'min'])
    parser.add_argument('--type', default='emb3nlp', choices=['nlp', 'emb', 'embnlp', 'emb2', 'emb2nlp', 'emb3', 'emb3nlp'])
    # parser.add_argument('--butd_eff', type=int, default=1)

    args = parser.parse_args()


    args.root_path = Path(args.root_path)
    args.code_path = Path(args.code_path)
    args.results_dir = Path(args.results_dir)/'{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.split, args.sweep, args.init_token, args.trigger_length, args.max_steps, args.type, args.weight_decay, args.feat_init)

    Path('results').mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    results_out = args.results_dir/'{}.pt'.format(args.model_id)
    results_hist = args.results_dir/'{}-hist.pt'.format(args.model_id)
    if results_out.exists():
        print (results_out)
        exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.split == 'train':
        root = args.root_path/'{}-train-dataset/'.format(args.dataset)
        metadata = pd.read_csv(root/'METADATA.csv')
    else:
        root  = args.root_path/'{}-test-dataset/'.format(args.dataset)
        metadata  = pd.read_csv(root/'METADATA.csv')


    model_info = metadata[metadata.model_name == args.model_id].iloc[0]
    # if args.butd_eff != 0:
    #     if (model_info.model != 'butd_eff'):
    #         exit()
    # else:
    #     if (model_info.model == 'butd_eff'):
    #         exit()

    model, hnd, samples_dir, data_info = load_model_util(model_info, root)


    image_feat_hnd = get_image_features(args.code_path)
    clean_ds = VQADataset(data_info, samples_dir, image_feat_hnd)


    # Load tokenizer
    if model_info['model'] == 'butd_eff':
        tokenizer = butd_tokenizer(args.code_path/'bottom-up-attention-vqa')
    else:
        tokenizer = openvqa_tokenizer(args.code_path/'openvqa')

    ti = TrojanInversionVQA(model, tokenizer, clean_ds, [args.trigger_length], ['start'], args.type, device)
    st_time = time.time()
    results, history = ti.trigger_sweeper(args.max_steps, ret_history=False, init_token=args.init_token, weight_decay=args.weight_decay, feat_init=args.feat_init)
    ed_time = time.time() - st_time

    out = {'sweep_time': ed_time, 'results': results}

    torch.save(out, results_out)
    torch.save(history, results_hist)




