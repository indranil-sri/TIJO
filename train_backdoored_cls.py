import torch
import pandas as pd
import numpy as np
from pathlib import Path


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy import interp
from pprint import pprint

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve,auc



# Set the results directory
results_root = Path('/results')



def cross_entropy(prob, labels):
    """
    Code to compute cross-entropy
    prob: probabilities from the model (numpy: Nx1)
    labels: ground-truth labels (numpy: Nx1)
    """
    prob = torch.Tensor(prob).squeeze()
    labels = torch.Tensor(labels).squeeze()
    assert (
        prob.shape == labels.shape
    ), "Check size of labels and probabilities in computing cross-entropy"
    ce = torch.nn.functional.binary_cross_entropy(prob, labels, reduction='none')
    return ce.mean().item()

def get_auc(labels, features):
    tprs = []
    aucs = []
    ces = []
    mean_fpr = np.linspace(0,1,100)
    kfold = StratifiedKFold(n_splits=5,shuffle=False)
    clfs = []
    for train, test in kfold.split(features, labels):
#         clf_0 = XGBClassifier(eval_metric='mlogloss')
#         clf_1 = XGBClassifier(
#                                     max_depth=2,
#                                     gamma=2,
#                                     eta=0.8,
#                                     reg_alpha=0.5,
#                                     reg_lambda=0.5,
#                                     eval_metric='mlogloss'
#                        )
        clf_3 = LogisticRegression(random_state=0, class_weight='balanced', C=1)
#         clf_2 = RandomForestClassifier(random_state=0)

#         prediction_0 = clf_0.fit(features[train],labels[train]).predict_proba(features[test])
#         prediction_1 = clf_1.fit(features[train],labels[train]).predict_proba(features[test])
#         prediction_2 = clf_2.fit(features[train],labels[train]).predict_proba(features[test])
        prediction_3 = clf_3.fit(features[train],labels[train]).predict_proba(features[test])
#         prediction = (prediction_1+prediction_2+prediction_3)/3
#         prediction = (prediction_1+prediction_2)/2
        prediction = prediction_3
        
        fpr, tpr, t = roc_curve(labels[test], prediction[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        ce = cross_entropy(prediction[:, 1], labels[test])
        aucs.append(roc_auc)
        ces.append(ce)
#         clfs.append((clf_1, clf_2, clf_3))

    return {
        'auc': np.array(aucs).mean(),
        'ce': np.array(ces).mean(),
        'aucs' : np.array(aucs),
        'ces': np.array(ces)
        
    }

def check_test(train_labels, train_feature, test_labels, test_feature):
    clf = LogisticRegression(random_state=0, class_weight='balanced', C=1)
    clf = clf.fit(train_feature,train_labels)

    pred = clf.predict_proba(test_feature)

    fpr, tpr, t = roc_curve(test_labels, pred[:, 1])
    roc_auc = auc(fpr, tpr)
    ce = cross_entropy(pred[:, 1], test_labels)

    return {
        'auc': roc_auc, 
        'ce': ce
    }


def get_all(dataset, split='test', tag='all_cls_1_10'):
    # import ipdb; ipdb.set_trace()
    root = Path('/data/TrojVQA/model_sets/v1/{}-{}-dataset/'.format(dataset, split))
    results_dir = results_root/'{}_{}_{}'.format(dataset, split, tag)

    metadata = pd.read_csv(root/'METADATA.csv')
    model_lst = metadata.model_name.to_list()

    labels_lst = (pd.isnull(metadata.target)==False).to_list()

    model_lst_v = []
    labels_lst_v = []
    all_run_data = []
    all_loss_feat = []
    all_acc_feat = []
    for _m, _l in zip(model_lst, labels_lst):
        if (results_dir/'{}.pt'.format(_m)).exists():
            try:
                _d = torch.load(results_dir/'{}.pt'.format(_m), map_location=torch.device('cpu'))
                all_run_data.append(_d)
                all_loss_feat.append(min(en['out']['best_loss'] for en in _d['results']))
                all_acc_feat.append(max(en['out']['best_targ_acc'] for en in _d['results']))
                labels_lst_v.append(_l)
                model_lst_v.append(_m)
            except:
                pass

    print ('{}_{}_{}'.format(dataset, split, tag), '\t', set(model_lst) - set(model_lst_v))

    all_loss_feat = np.array(all_loss_feat)[:, None]
    all_acc_feat = np.array(all_acc_feat)[:, None]

    ret = {
        'metadata' : metadata,
        'model_lst' : model_lst_v,
        'labels_lst' : np.array(labels_lst_v),
        'all_run_data' : all_run_data,
        'all_loss_feat' : all_loss_feat,
        'all_acc_feat' : all_acc_feat,
        'all_loss_feat' : all_loss_feat,
        'all_acc_feat' : all_acc_feat,
    }

    return ret


def get_metrices(dataset, tag):
    train_data = get_all(dataset, split='train', tag=tag)
    #test_data = get_all(dataset, split='test', tag=tag)
    #print (len(train_data['labels_lst']), len(test_data['labels_lst']))

    train_cv_l  = get_auc(train_data['labels_lst'], train_data['all_loss_feat'])
    train_cv_a  = get_auc(train_data['labels_lst'], train_data['all_acc_feat'])
    #train_cv_la = get_auc(train_data['labels_lst'], np.hstack((train_data['all_loss_feat'], train_data['all_acc_feat'])))

    #test_l  =  check_test(train_data['labels_lst'], train_data['all_loss_feat'], 
    #                        test_data['labels_lst'], test_data['all_loss_feat'])
    #test_a  =  check_test(train_data['labels_lst'], train_data['all_acc_feat'], 
    #                        test_data['labels_lst'], test_data['all_acc_feat'])
    #test_la =  check_test(train_data['labels_lst'], np.hstack((train_data['all_loss_feat'], train_data['all_acc_feat'])), 
    #                        test_data['labels_lst'], np.hstack((test_data['all_loss_feat'], test_data['all_acc_feat'])))
    
    return {
        'train_data': train_data,
        #'test_data': test_data,
        'train_cv_l'  : train_cv_l,
        'train_cv_a'  : train_cv_a,
        # 'train_cv_la' : train_cv_la,
        #'test_l'  : test_l,
        #'test_a'  : test_a,
        #'test_la' : test_la,
    }



############
# Load All
############

# NLP trigger inversion on all the splits
v1_nlp = get_metrices('v1', 'all_cls_1_15_nlp')
v1a_nlp = get_metrices('v1a', 'all_cls_1_15_nlp')
v1b_nlp = get_metrices('v1b', 'all_cls_1_15_nlp')
v1c_nlp = get_metrices('v1c', 'all_cls_1_15_nlp')
v1d_nlp = get_metrices('v1d', 'all_cls_1_15_nlp')
v1e_nlp = get_metrices('v1e', 'all_cls_1_15_nlp')

# Vision trigger inversion on all the splits
v1_emb = get_metrices('v1', 'all_cls_1_15_emb')
v1a_emb = get_metrices('v1a', 'all_cls_1_15_emb')
v1b_emb = get_metrices('v1b', 'all_cls_1_15_emb')
v1c_emb = get_metrices('v1c', 'all_cls_1_15_emb')
v1d_emb = get_metrices('v1d', 'all_cls_1_15_emb')
v1e_emb = get_metrices('v1e', 'all_cls_1_15_emb')

# Multimodal trigger inversion on all the splits
v1_embnlp = get_metrices('v1', 'all_cls_1_15_embnlp')
v1a_embnlp = get_metrices('v1a', 'all_cls_1_15_embnlp')
v1b_embnlp = get_metrices('v1b', 'all_cls_1_15_embnlp')
v1c_embnlp = get_metrices('v1c', 'all_cls_1_15_embnlp')
v1d_embnlp = get_metrices('v1d', 'all_cls_1_15_embnlp')
v1e_embnlp = get_metrices('v1e', 'all_cls_1_15_embnlp')


# Check AUC numbers
pprint (v1_embnlp)