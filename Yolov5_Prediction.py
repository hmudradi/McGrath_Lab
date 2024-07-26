#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
class Metrics:
    def __init__(self, label, predict):
        print('label=1 is positive, label=0 is negative')
        self.label=label
        self.predict=predict
        
        tp = np.sum((label == 'female') & (predict == 'female'))
        tn = np.sum((label == 'male') & (predict == 'male'))
        fp = np.sum((label == 'male') & (predict == 'female'))
        fn = np.sum((label == 'female') & (predict == 'male'))
        
        total = tp + tn + fp + fn
        pp = (tp + fn) / total
        pn = (tn + fp) / total
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
        fpr = fp / (tn + fp) if (tn + fp) != 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) != 0 else 0
        self.metrics = {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'total': total,
            'pp': pp,
            'pn': pn,
            'tpr': tpr,
            'tnr': tnr,
            'fpr': fpr,
            'fnr': fnr,
            'acc': (tp + tn) / total,
            'pprecision': tp / (tp + fp) if (tp + fp) != 0 else 0,
            'nprecision': tn / (tn + fn) if (tn + fn) != 0 else 0,
            'precall': tpr,
            'nrecall': tnr,
            'pFm': 2 * tp / (2 * tp + tn + fp) if (2 * tp + tn + fp) != 0 else 0,
            'nFm': 2 * tn / (2 * tn + tp + fn) if (2 * tn + tp + fn) != 0 else 0,
            'bacc': 0.5 * (tpr + tnr),
            'mcc': (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) != 0 else 0
        }
        
    def get_metrics(self):
        return self.metrics

def by_track(df, uid='uid', metric='acc'):
    all_metrics = {}
    
    for Tk_Id, group_df in df.groupby('Tk_Id'):
        mode_predictions = group_df.groupby(uid)['prediction'].agg(lambda x: x.value_counts().index[0])
        true_labels = group_df.groupby(uid)['label'].first()
        mode_predictions = mode_predictions.reindex(true_labels.index)
        metrics = Metrics(mode_predictions.values, true_labels.values)
        all_metrics[Tk_Id] = metrics.get_metrics()[metric]
    
    return all_metrics

def by_track_sex(df, uid='uid', metric='acc'):
    male_metrics = {}
    female_metrics = {}
    for Tk_Id, full_df in df.groupby('Tk_Id'):
        for gender in ['male', 'female']:
            group_df = full_df[full_df['label'] == gender]
            mode_predictions = group_df.groupby(uid)['prediction'].agg(lambda x: x.value_counts().index[0])
            true_labels = group_df.groupby(uid)['label'].first()
            mode_predictions = mode_predictions.reindex(true_labels.index)
            metrics = Metrics(mode_predictions.values, true_labels.values)
            if gender == 'male':
                male_metrics[Tk_Id] = metrics.get_metrics()[metric]
            else:
                female_metrics[Tk_Id] = metrics.get_metrics()[metric]
    return female_metrics, male_metrics
def by_track_count(df, uid='uid', metric='acc'):
    male_metrics = {}
    female_metrics = {}
    for track_id, full_df in df.groupby('track_id'):
            group_df = full_df
            mode_predictions = group_df.groupby(uid)['prediction'].agg(lambda x: x.value_counts().index[0])
            print(mode_predictions)
            true_labels = group_df.groupby(uid)['label'].first()
            print(true_labels)
            mode_predictions = mode_predictions.reindex(true_labels.index)
            metrics = Metrics(mode_predictions.values, true_labels.values)
            if str(full_df['label'].iloc[0]) == 'male':
                male_metrics[track_id] = [metrics.get_metrics()['tn'], metrics.get_metrics()['fn']]
            else:
                female_metrics[track_id] = [metrics.get_metrics()['tp'], metrics.get_metrics()['fp']]
    return female_metrics, male_metrics

def metrics_by_Tk_Id(df, flag=0, metric='acc'):
    all_metrics = {}
    if flag==0:
        for Tk_Id, group_df in df.groupby('Tk_Id'):
            metrics = Metrics(group_df['label'].values, group_df['prediction'].values)
            all_metrics[Tk_Id] = metrics.get_metrics()[metric]
        return all_metrics
    if flag==1:
        female_metrics = {}
        male_metrics = {}
        for Tk_Id, group_df in df[df.label=='female'].groupby('Tk_Id'):
            metrics = Metrics(group_df['label'].values, group_df['prediction'].values)
            female_metrics[Tk_Id] = metrics.get_metrics()[metric]
        for Tk_Id, group_df in df[df.label=='male'].groupby('Tk_Id'):
            metrics = Metrics(group_df['label'].values, group_df['prediction'].values)
            male_metrics[Tk_Id] = metrics.get_metrics()[metric]
        return female_metrics, male_metrics

def columns_to_dict(df, key_col, value_col):
    return dict(zip(df[key_col], df[value_col]))


base = '/home/hmudradi3/GaTech Dropbox/CoS/BioSci/BioSci-McGrath/PublicIndividualData/Breanna/aim1_final/data_frames'
paths = {'mfex': 'Meta_Data/by_exp/YOLOV5_Cls_Manual_Frame_exp.csv',
         'mvex': 'Meta_Data/by_exp/YOLOV5_Cls_Manual_Videos_exp.csv',
         'pmvd': 'Predict_Manual_Videos_Val/YOLOV5_Cls_Manual_Videos.csv',
         'pmfr': 'Predict_Manual_Videos_Val/YOLOV5_Manual_Frames.csv',
         'bb': 'Temporal/BioBoost_final_predictions.csv'}

mfex, mvex, pmvd, pmfr, bb= (pd.read_csv(os.path.join(base, path)) for path in paths.values())


tdf=pd.DataFrame()
tdf['trial']=[i.split('__')[0] for i in bb.id]
tdf['uid']=bb.id
tdf['label']=['female' if i==1 else 'male' for i in bb['gt']]
tdf['prediction']=['female' if i>=0.5 else 'male'for i in bb['pred']]


pmfr['uid']=pmfr['trial']+'__'+pmfr['base_name']+'__'+pmfr['track_id'].astype(str)
pmvd['uid']=pmvd['trial']+'__'+pmvd['base_name']+'__'+pmvd['track_id'].astype(str)

pmfr['Tk_Id']=['_'.join(i.split('nuc')[1].split('_')[:3]) for i in pmfr.trial]
pmvd['Tk_Id']=['_'.join(i.split('nuc')[1].split('_')[:3]) for i in pmvd.trial]
tdf['Tk_Id']=['_'.join(i.split('nuc')[1].split('_')[:3]) for i in tdf.trial]



mf_f,mf_m = metrics_by_Tk_Id(pmfr,flag=1)
mv_f, mv_m= metrics_by_Tk_Id(pmvd, flag=1)
mv=metrics_by_Tk_Id(pmvd, flag=0, metric='acc')
mf=metrics_by_Tk_Id(pmfr, flag=0, metric='acc')
mv=by_track(pmvd, metric='bacc')
mf=by_track(pmfr, metric='bacc')

tm=metrics_by_Tk_Id(tdf, metric='bacc')

ft, mt=by_track_count(pmfr[pmfr.trial=='MC_singlenuc96_b1_Tk41_081120'])

fmv, mmv=by_track_sex(pmvd, metric='acc')
fmf, fmv=by_track_sex(pmfr, metric='acc')
import matplotlib.pyplot as plt
import numpy as np

# Function to create a radar chart
def create_radar_chart(metrics_dict1, metrics_dict2, title):
    labels = list(metrics_dict1.keys())
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Repeat the first value to close the circle
    angles += angles[:1]

    # Values from the dictionaries
    values1 = list(metrics_dict1.values())
    values1 += values1[:1]
    values2 = list(metrics_dict2.values())
    values2 += values2[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.fill(angles, values1, color='blue', alpha=0.25)
    ax.fill(angles, values2, color='red', alpha=0.25)

    ax.plot(angles, values1, color='blue', linewidth=2, label='manual_frames')
    ax.plot(angles, values2, color='red', linewidth=2, label='manual_videos')


    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.title(title)
    plt.tight_layout()
    
    #plt.savefig('/home/hmudradi3/GaTech Dropbox/CoS/BioSci/BioSci-McGrath/PublicIndividualData/Breanna/aim1_final/results/'+'_'.join(title.split(' ')), dpi=300, bbox_inches='tight')
    plt.show()

updated_mv_f = {}
for key in mv_f.keys():
    new_key = key + '__' + str(round(mv_ft[key], 2))[-2:] + '__' + str(round(mf_ft[key], 2))[-2:]
    updated_mv_f[new_key] = mv_f[key]
updated_mf_f = {}
for key in mf_f.keys():
    new_key = key + '__' + str(round(mv_ft[key], 2))[-2:] + '__' + str(round(mf_ft[key], 2))[-2:]
    updated_mf_f[new_key] = mf_f[key]
# Example usage with your data
create_radar_chart(updated_mv_f, updated_mf_f, "Female Track Acc")

updated_mv_m = {}
for key in mv_m.keys():
    new_key = key + '__' + str(round(mv_mt[key], 2))[-2:] + '__' + str(round(mf_mt[key], 2))[-2:]
    updated_mv_m[new_key] = mv_m[key]
updated_mf_m = {}
for key in mf_m.keys():
    new_key = key + '__' + str(round(mv_mt[key], 2))[-2:] + '__' + str(round(mf_mt[key], 2))[-2:]
    updated_mf_m[new_key] = mf_m[key]
# Example usage with your data
create_radar_chart(updated_mv_m, updated_mf_m, "Male Track Acc")

create_radar_chart(mf, mv, "Frame bacc")


