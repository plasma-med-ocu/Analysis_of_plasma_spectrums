# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:22:13 2020

@author: BALHORN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from sklearn import model_selection
from sklearn.datasets import fetch_rcv1
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import  multilabel_confusion_matrix, make_scorer,hamming_loss
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from skmultilearn.adapt import MLTSVM
from skmultilearn.ensemble import RakelO, RakelD
from collections import Counter
from numpy.random import default_rng
from itertools import combinations, repeat
import joblib
import time
import glob
import os


def get_files(path, gl_str, show=True, **kwargs):
    """
    wrapper for a glob routine the returns files and the file names in order
    """
    files = sorted(glob.glob(path+gl_str, **kwargs))
    names = []
    for ix, f in enumerate(files):
        f = f.replace(path, '')
        if show:
            print(ix, f)
        names += [f.strip('\\')]
    return files, names

def create_labels(data_df):
    """
    create the tabels for training the machinlearning algorithem from the
    names of the reference sampels that were measured.
    """
    #elements = ['Cr', 'Ni', 'Fe', 'Ta', 'Nb', 'W', 'Nd', 'Ti', 'Cu']
    elements = ['Ni', 'Ta', 'Nb', 'W', 'Nd', 'Ti', 'Cu']
    labels = []
    Sample = data_df.index.get_level_values(level=0).values
    Number = data_df.index.get_level_values(level=1).values
    for sample in Sample:
        occurring = []
        for elem in elements:
            if elem in sample:
                occurring += [elem]
        labels.append(tuple(occurring))

    ROI_Reihenfolge = data_df.columns.values
    mlb = MultiLabelBinarizer()

    y = np.array(mlb.fit_transform(labels))
    print(elements)
    print(mlb.classes_)
    X = data_df.loc[list(zip(Sample,Number)),ROI_Reihenfolge].values
    print(y.shape)
    print(X.shape)
    return X, y, mlb

def mean_df_elements(dafr,step):
    """
    calculate the mean of subsequent measurements. step is the number of 
    measurements to calculate the mean over as well as the stride. 
    No value in the Datafraem is used twice.
    """
    res_df = []
    dafr_samps = dafr.index.get_level_values(level=0).unique().values
    for dafr_samp in dafr_samps:
        samp_df = dafr.loc[dafr_samp,:].copy()
        samp_df = samp_df.rolling(step).mean()
        samp_df = samp_df.iloc[::step,:]
        samp_df = samp_df.dropna(axis=0)
        ixes = [(dafr_samp,num) for num in np.arange(0,len(samp_df.index.values),1)]
        samp_df.index = pd.MultiIndex.from_tuples(ixes, names=['Sample','Number'])
        res_df += [samp_df]
    return pd.concat(res_df,axis=0)



# read data

datafile = r"C:\Daten\2020_02_22_Finale_Aufwertungen\Referenzmessungen\Scores_und_Bounds\new_ROI_calc.csv"

data = pd.read_csv(datafile, sep=';').set_index(['Sample'])  # , 'Number'])
data = data.drop('Number',axis=1)
data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

data = data.loc[data.index.get_level_values(level=0).values != 'CrNiFe',:]
data = data.loc[data.index.get_level_values(level=0).values != 'Ta_2',:]
# cols = [col for col in data.columns.values if 'Fe' in col]
# cols += [col for col in data.columns.values if 'Cr' in col]
# data = data.drop(labels=cols,axis=1)

new_index = []
num_max = []
for Sam in data.index.unique():
    n,m = data.loc[Sam,:].shape
    num_max += [n]
    new_index += [(Sam,number) for number in np.arange(0,n,1)]
data.index = pd.MultiIndex.from_tuples(new_index,names=['Sample','Number'])
per_sample = np.min(num_max)
data = data.loc[data.index.get_level_values(level=1)<per_sample,:]

print('data: \n', data)


"""
create mixtures of the measuremnts of reference samples.
The ROI of two measurements are averaged for every unique combination of two
reference samples
"""

combs = []
samps_unique = data.index.get_level_values(level=0).unique().values
combs = list(combinations(samps_unique, 2))
print(len(combs))

data_mixed = data.copy()
for cmb in combs:
    N = len(cmb)
    #print(cmb, N)
    new_key = ''.join(cmb)
    nums = np.arange(0,per_sample,1)
    for num in nums:
        arr = np.zeros(data.shape[1])
        for el in cmb:
            arr += data.loc[(el,num),:]
        data_mixed.loc[(new_key,num),:] = arr/N

print('data_mixed: \n', data_mixed)

data_mean =  mean_df_elements(data,3)
print('data_mean: \n', data_mean)

X, y, mlb = create_labels(data)
X_mixed, y_mixed, mlb_mixed = create_labels(data_mixed)
X_mean, y_mean, mlb_mean = create_labels(data_mean)

# %%


"""
data:
    ['Ni', 'Ta', 'Nb', 'W', 'Nd', 'Ti', 'Cu']
    ['Cu' 'Nb' 'Nd' 'Ni' 'Ta' 'Ti' 'W']
    (8022, 7)
    (8022, 38)

data_mixed:
    ['Ni', 'Ta', 'Nb', 'W', 'Nd', 'Ti', 'Cu']
    ['Cu' 'Nb' 'Nd' 'Ni' 'Ta' 'Ti' 'W']
    (30072, 7)
    (30072, 38)

data_mean:
    ['Ni', 'Ta', 'Nb', 'W', 'Nd', 'Ti', 'Cu']
    ['Cu' 'Nb' 'Nd' 'Ni' 'Ta' 'Ti' 'W']
    (2667, 7)
    (2667, 38)
"""

#X_mean, y_mean, mlb_mean = create_labels(data_mean.loc[data_mean.index.get_level_values(level=1)<200,:])
#X, y, mlb = create_labels(data.loc[data.index.get_level_values(level=1)<200,:])


data_percentage = 0.1
X_train, X_test, y_train, y_test = train_test_split(
    #X, y,
    X_mixed, y_mixed,
    #X_mean, y_mean,
    test_size=0.33*data_percentage,
    train_size=0.67*data_percentage,
    random_state=42)

scaler = StandardScaler()
#clf = MultiOutputClassifier(SVC())
clf = ClassifierChain(base_estimator=SVC())
#clf = MLTSVM()

pca = PCA()

def to_Sparse(X):
    return csr_matrix(X)

to_Sparse_transformed = FunctionTransformer(func=to_Sparse)

pipe = Pipeline(steps=[
    ('scaler',scaler),
    ('pca', pca),
    ('clf',clf)
    ])

param_grid = {
    'pca__n_components': list(np.arange(5,36,5)),
    'clf__base_estimator__C': [np.power(10,float(i)) for i in range(-5, 5, 1)],
    'clf__base_estimator__gamma': ['auto','scaled'],
    'clf__base_estimator__kernel': ['kdf','poly','sigmoid'],
}

scorers = {'hamming':make_scorer(hamming_loss,grater_is_better=False)}

start = time.time()
grid = GridSearchCV(pipe, param_grid, n_jobs=-1)
grid.fit(X_train, y_train)

print(f'Duration: {round(time.time()-start,1)} seconds')

print("The best parameters are %s with a score of %0.6f"
      % (grid.best_params_, grid.best_score_))

dump_path = r"C:\Daten\2020_02_22_Finale_Aufwertungen\Referenzmessungen\Fits_und_Bounds\grid_object_ChainClassifier_SVC.pkl"
joblib.dump(grid, dump_path)

res_df = pd.DataFrame(columns=grid.cv_results_['params'][0].keys())
for param_ix, (param_dict,score) in enumerate(zip(grid.cv_results_['params'],grid.cv_results_['mean_test_score'])):
    for key in param_dict.keys():
        res_df.loc[param_ix,key] = param_dict[key]
    res_df.loc[param_ix,'score'] = score
print(res_df)


conf =  multilabel_confusion_matrix(y_test,grid.predict(X_test))
for elem,cl in zip(conf,mlb.classes_):
    conf_df = pd.DataFrame()
    conf_df.loc['true','negative'] = elem[0,0]
    conf_df.loc['false','negative'] = elem[1,0]
    conf_df.loc['true','positive'] = elem[1,1]
    conf_df.loc['false','positive'] = elem[0,1]
    print(cl)
    print(conf_df)

res_df.to_csv(dump_path.replace('.pkl','.csv'),sep=';', float_format='%.3f')

# %%
print(grid.score(X_test, csr_matrix(y_test)))
# %%


file_path = [r'C:\Daten\2020_02_22_Finale_Aufwertungen\Type_A_Gridsize\ROIs_and_Scores',
             r'C:\Daten\2020_02_22_Finale_Aufwertungen\Type_B_Roundedpattern\ROIs_and_Scores',
             r'C:\Daten\2020_02_22_Finale_Aufwertungen\Type_B_Roundedpattern_parallel\ROIs_and_Scores',
             r'C:\Daten\2020_02_22_Finale_Aufwertungen\Type_C_CircularPattern\ROIs_and_Scores',
             r'C:\Daten\2020_02_22_Finale_Aufwertungen\Type_variable_FoilPenetration\ROIs_and_Scores']

r_files = []
s_files = []
s_png = []
for fp in file_path:
    files, names = get_files(fp,r'\ROI_neu**LSA_RawData**.csv')
    save = [f.replace(r'ROIs_and_Scores', r'Predictions') for f in files]
    save_png = [s.replace(r'.csv', r'.png') for s in save]
    r_files += files
    s_files += save
    s_png += save_png



def bar_plot_labels(y_pred_labels, sapa):
    plot_labels = ['Cu', 'Nb', 'Nd', 'Ni', 'Ta', 'Ti', 'W', 'None']
    samps = {}
    for key in plot_labels:
        samps[key] = np.zeros(len(y_pred_labels))
    total = 0
    for label_index, labels in enumerate(y_pred_labels):
        if labels == ():
            samps['None'][label_index] = 1
            total += 1
        else:
            for label in labels:
                samps[label][label_index] = 1
                total += 1
    res = np.zeros(len(plot_labels))
    for ix,key in enumerate(plot_labels):
        res[ix] = np.sum(samps[key])/total
    fig, ax = plt.subplots(figsize=(5,3))
    X = np.arange(len(plot_labels))
    ax.bar(X,res,color='navy',zorder=6)
    ax.set_ylim(0,1.3)
    ax.set_xticks(X)
    ax.set_xticklabels(plot_labels)
    ax.set_ylabel('Relative Frequency')
    ax.grid(zorder=1)
    plt.savefig(sapa,dpi=600)
    plt.show()
    return

for file,savefile,save_graph in zip(r_files,s_files,s_png):
    df_to_predict = pd.read_csv(file,sep=';')

    df_to_predict = df_to_predict.set_index('Counter')
    df_to_predict = df_to_predict[~df_to_predict.isin([np.nan, np.inf, -np.inf]).any(1)]

    y_pred = grid.predict(df_to_predict.values)
    bar_plot_labels(mlb.inverse_transform(y_pred),save_graph)
    pd.DataFrame(y_pred).to_csv(savefile,sep=';')



# %%

file_path = [r'C:\Daten\2020_02_22_Finale_Aufwertungen\Type_B_Roundedpattern\ROIs_and_Scores']

r_files = []
s_files = []
s_png = []
for fp in file_path:
    files, names = get_files(fp,r'\ROI_neu_4KomponentenPlatine_**LSA_RawData**.csv')
    save = [f.replace(r'ROIs_and_Scores', r'Predictions') for f in files]
    save_png = [s.replace(r'.csv', r'.png') for s in save]
    r_files += files
    s_files += save
    s_png += save_png



def bar_plot_labels(y_pred_labels, sapa):
    """
    create a barplot with the relative frequency of the predicted results
    """
    plot_labels = ['Cu', 'Nb', 'Nd', 'Ni', 'Ta', 'Ti', 'W', 'None']
    samps = {}
    for key in plot_labels:
        samps[key] = np.zeros(len(y_pred_labels))
    total = 0
    for label_index, labels in enumerate(y_pred_labels):
        if labels == ():
            samps['None'][label_index] = 1
            total += 1
        else:
            for label in labels:
                samps[label][label_index] = 1
                total += 1
    res = np.zeros(len(plot_labels))
    for ix,key in enumerate(plot_labels):
        res[ix] = np.sum(samps[key])/total
    fig, ax = plt.subplots(figsize=(5,3))
    X = np.arange(len(plot_labels))
    ax.bar(X,res,color='navy',zorder=6)
    ax.set_ylim(0,1.3)
    ax.set_xticks(X)
    ax.set_xticklabels(plot_labels)
    ax.set_ylabel('Relative Frequency')
    ax.grid(zorder=1)
    plt.savefig(sapa,dpi=600)
    plt.show()
    return

for file,savefile,save_graph in zip(r_files,s_files,s_png):
    df_to_predict = pd.read_csv(file,sep=';')
    df_to_predict = df_to_predict.set_index('Counter')
    df_to_predict.index = np.arange(0,df_to_predict.shape[0],1)
    split = np.arange(0,df_4kp.T.shape[1],1200)
    proben = ['A','B','D']
    for st,ed,probe in zip(split[:-1],split[1:],proben):
        sub_df_to_predict = df_to_predict.loc[st:ed,:].copy()
        savefile_loc = savefile.replace('4KomponentenPlatine_',f'4KomponentenPlatine_{probe}_')
        save_graph_loc = save_graph.replace('4KomponentenPlatine_',f'4KomponentenPlatine_{probe}_')
        df_to_predict = df_to_predict[~df_to_predict.isin([np.nan, np.inf, -np.inf]).any(1)]
        y_pred = grid.predict(sub_df_to_predict.values)
        bar_plot_labels(mlb.inverse_transform(y_pred),save_graph_loc)
        pd.DataFrame(y_pred).to_csv(savefile_loc,sep=';')


# %%
def barplot_succsesiv(df_dict,sapa):
    """
    create a barplot with the relative frequency of the predicted results
    over successive measurements of the same sample
    """
    plot_labels = ['Cu', 'Nb', 'Nd', 'Ni', 'Ta', 'Ti', 'W', 'None']
    dict_len = len(df_dict)
    df_results = pd.DataFrame()
    for df_key in df_dict.keys():
        y_pred_labels = df_dict[df_key]
        print(df_key)
        samps = {}
        for key in plot_labels:
            samps[key] = np.zeros(len(y_pred_labels))
        total = 0
        for label_index, labels in enumerate(y_pred_labels):
            if labels == ():
                samps['None'][label_index] = 1
                total += 1
            else:
                for label in labels:
                    samps[label][label_index] = 1
                    total += 1
        for key in plot_labels:
            df_results.loc[key,df_key] = np.sum(samps[key])/total
    print(df_results)
    fig,ax = plt.subplots(figsize=(5,3))
    df_results.T.plot(kind='bar',stacked=True,ax=ax,color=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf'],zorder=6,rot=0)
    ax.set_ylim(0,1.5)
    ax.set_ylabel('Relative Frequency')
    ax.set_xlabel('Number of laser drilled Patten [#]')
    ax.set_xticks(np.arange(0,df_results.shape[1],1))
    ax.set_xticklabels(['before']+list(np.arange(1,df_results.shape[1],1)))
    ax.grid(zorder=0)
    ax.legend(loc='upper left',ncol=4)
    plt.tight_layout()
    plt.savefig(sapa,dpi=600)
    plt.show()
    return

dump_path = r"C:\Daten\2020_02_22_Finale_Aufwertungen\Referenzmessungen\Fits_und_Bounds\grid_object_ChainClassifier_SVC.pkl"
grid = joblib.load(dump_path)


file_path = [r'C:\Daten\2020_02_22_Finale_Aufwertungen\Type_A_Gridsize\ROIs_and_Scores']


for fp in file_path:
    files, names = get_files(fp,r'\ROI_neu_5x5_2px_01s_05s**LSA_RawData**.csv')


save_graph = r'C:\Daten\2020_02_22_Finale_Aufwertungen\Type_A_Gridsize\Predictions\ROI_neu_5x5_2px_01s_05s_stacked_Barplot.png'


pred_dict = {}
for file in files:
    df_to_predict = pd.read_csv(file,sep=';')
    df_to_predict = df_to_predict.set_index('Counter')
    df_to_predict = df_to_predict[~df_to_predict.isin([np.nan, np.inf, -np.inf]).any(1)]
    print(df_to_predict)
    name_df = file.split(os.sep)[-1].replace('ROI_neu_5x5_2px_01s_05s_','').split('_')[0]
    y_pred = grid.predict(df_to_predict.values)
    pred_dict[name_df] = mlb.inverse_transform(y_pred)
barplot_succsesiv(pred_dict,save_graph)

file_path = [r'C:\Daten\2020_02_22_Finale_Aufwertungen\Type_A_Gridsize\ROIs_and_Scores']


for fp in file_path:
    files, names = get_files(fp,r'\ROI_neu_7x7_2px_01s_03s**LSA_RawData**.csv')


save_graph = r'C:\Daten\2020_02_22_Finale_Aufwertungen\Type_A_Gridsize\Predictions\ROI_neu_7x7_2px_01s_03s_stacked_Barplot.png'


pred_dict = {}
for file in files:
    df_to_predict = pd.read_csv(file,sep=';')
    df_to_predict = df_to_predict.set_index('Counter')
    df_to_predict = df_to_predict[~df_to_predict.isin([np.nan, np.inf, -np.inf]).any(1)]
    name_df = file.split(os.sep)[-1].replace('ROI_neu_7x7_2px_01s_03s_','').split('_')[0]
    y_pred = grid.predict(df_to_predict.values)
    pred_dict[name_df] = mlb.inverse_transform(y_pred)
barplot_succsesiv(pred_dict,save_graph)



