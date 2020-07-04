# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:16:36 2020

@author: BALHORN
"""

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from matplotlib.ticker import MaxNLocator
from operator import itemgetter
import patlib

plt.rcParams.update({'font.size': 10})

# %% Functions

def strictly_increasing(L):
    """
    Check for strictly increasing monotonously in L
    """
    return all(x<y for x, y in zip(L, L[1:]))

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

def read_from_csv(path, cutoff=None, sel_area=('start','end')):
    """
    reads the spectrometer data and correctes the detector noise via the
    first values in the file, the only containt this noise and no intensity
    due to the plasma emissions.
    """
    df = pd.read_csv(path, sep=';', engine='python', encoding = "utf-8-sig").set_index('MeasCtr').drop('TimeStamp', axis=1).T
    df.index = [float(str(d.split('.')[0])+'.'+str(d.split('.')[1])) for d in df.index]
    df['ix'] = df.index
    df['ix'] = df['ix'].round(3)
    df = df.loc[~df['ix'].duplicated(keep='first')]
    df.set_index('ix', inplace=True)
    df.index.name = 'Wavelength'
    df.sort_index(inplace=True)

    if not strictly_increasing(df.index):
        print('The Wavelengths are not strictly increasing!')
        return None

    st, ed = sel_area
    sel = 0
    if st == 'start' and cutoff != None:
        st=cutoff
    if st == 'start' and cutoff == None:
        st=0
    if ed == 'end':
        sel = df.columns.values[st:]
    else:
        sel = df.columns.values[st:ed]

    if cutoff != None:
        BG = df.loc[:,df.columns.values[0:cutoff]].mean(axis=1)
    else:
        BG = 0

    return df.loc[:,sel].sub(BG, axis=0)

def read_ROI(path):
    """
    reads a file the containes the Regions of Interst (ROIs). These are start
    and end values for the mean intervalls of the emission line and the
    corresponding background.
    """
    roi = pd.read_csv(path, sep=';')
    roi.columns = roi.columns.str.strip()
    roi['ROI-Name'] = roi['ROI-Name'].str.strip()
    roi.set_index('ROI-Name', inplace=True)
    for peak in roi.index:
        peak_split = peak.split('_')
        sp = peak_split[-1].split('-')
        if len(sp) > 1:
            for wl in sp:
                name_neu = peak.replace(peak_split[-1], wl)
                roi.loc[name_neu,:] = roi.loc[peak,:]
            roi.drop(peak, axis=0, inplace=True)
    roi['Line'] = [p if 'BG_' not in p else p.replace('BG_','') for p in roi.index]
    roi.reset_index(inplace=True)
    roi.set_index(['Line','ROI-Name'], inplace=True)
    roi.sort_index(inplace=True)
    s = ['Start Auslesebereich', 'Ende Auslesebereich', 'Start Integration', 'Ende Integration']
    s_bg = ['Start Auslesebereich_BG', 'Ende Auslesebereich_BG', 'Start Integration_BG', 'Ende Integration_BG']
    roi.loc[:,'Start Auslesebereich_BG'] = np.nan
    roi.loc[:,'Ende Auslesebereich_BG'] = np.nan
    roi.loc[:,'Start Integration_BG'] = np.nan
    roi.loc[:,'Ende Integration_BG'] = np.nan
    for peak in roi.index.get_level_values(level=0):
        roi.loc[(peak, peak),s_bg] = roi.loc[(peak,'BG_'+peak), s].values
    roi.reset_index(inplace=True)
    roi.set_index('ROI-Name', inplace=True)
    roi.drop('Line', axis=1, inplace=True)
    for peak in roi.index:
        if 'BG_' in peak:
            roi.drop(peak, axis=0, inplace=True)
    return roi

def apply_ROI(df, ROI, BG=True, alt=True):
    """
    The ROIs are calculated on all spectrums contained in df
    """
    df_roi = pd.DataFrame()
    peaks = [*ROI.index]
    for peak in peaks:
        st = ROI.at[peak,'Start Integration']
        ed = ROI.at[peak,'Ende Integration']
        st_bg = ROI.at[peak,'Start Integration_BG']
        ed_bg = ROI.at[peak,'Ende Integration_BG']
        if st > ed:
            print('Peak Fail: ', peak)
        if st_bg > ed_bg:
            print('BG Fail: ', peak)
        quot = (ed-st) / (ed_bg-st_bg)
        pk_int = df.loc[st:ed,:].replace([np.inf, -np.inf], np.nan).dropna(axis=1).sum(axis=0)
        bg_int = df.loc[st_bg:ed_bg,:].replace([np.inf, -np.inf], np.nan).dropna(axis=1).sum(axis=0)
        pk_mean = df.loc[st:ed,:].replace([np.inf, -np.inf], np.nan).dropna(axis=1).mean(axis=0)
        bg_mean = df.loc[st_bg:ed_bg,:].replace([np.inf, -np.inf], np.nan).dropna(axis=1).mean(axis=0)
        if alt:
            if BG:
                df_roi[peak] = (pk_int - quot* bg_int) / bg_int
            else:
                df_roi[peak] = pk_int
        else:
            if BG:
                df_roi[peak] = pk_mean -bg_mean
            else:
                df_roi[peak] = pk_mean
    if isinstance(df_roi.index, pd.core.index.MultiIndex):
        df_roi.index = df_roi.index.set_levels(df_roi.index.levels[1].astype(int), level=1)
    else:
        df_roi.index.name = 'Counter'
        df_roi.index = df_roi.index.astype(int)
        df_roi.sort_index(inplace=True)
    return df_roi

def calc_bound(data_mn,data_std, labels, where):
    """
    calculate the mean of multiple reference measurements based on wether the
    element intended to be measured is contained.
    Also Propagation of uncertainties.
    
    only used in function below
    """
    mn = np.array([data_mn[labels[ix]] for ix in where])
    std = np.array([data_std[labels[ix]] for ix in where])
    bound = np.mean(mn)
    bound_err = np.sum(std)
    return bound, bound_err

def score_bounds(score, path_gr=None):
    """
    calculates an indicator for the signal quality for the ROIs. The ROI 
    intensity for measurements of reference sampels containing the element that
    the ROI is intedend to measure vs. measurements of elements where the
    element is not contained. (essentially an signal to noise ration of the ROIs)
    """
    rois = list(score.index.get_level_values(0).unique().values)
    samples = score.columns.values
    #print(samples)
    bounds = pd.DataFrame()
    for roi in rois:
        element = roi.split('_')[0]
        where = np.array([i for i, s in enumerate(samples) if element in s])
        if len(where) !=0:
            where_not = np.array([i for i, s in enumerate(samples) if element not in s])
            data_mn = score.loc[(roi, 'mean'),:]
            data_std = score.loc[(roi, 'std'),:]
            labels = samples  #list(data.index.get_level_values(0).unique())
            mn_ = np.array([data_mn[i] for i in labels])
            std_ = np.array([data_std[i] for i in labels])
            maximum = np.max(mn_+ std_)+0.1
            minimum = np.min(mn_- std_) -0.1
            x_pos = np.arange(0,len(mn_), 1)
            x_pos_2 = np.arange(-1,len(mn_)+1, 1)

            b_up, b_up_err = calc_bound(data_mn,data_std, labels, where)
            b_low, b_low_err = calc_bound(data_mn,data_std, labels, where_not)
            ab = abs(b_up - b_low)
            A = ( (b_up - b_low) / abs(b_up - b_low) ) * b_up_err
            B = ( (b_low - b_up) / abs(b_up - b_low) ) * b_low_err
            s_ab = np.power(np.power(A, 2)+np.power(B, 2), 1/2)
            sig = ab/s_ab
            col = ['roi','I_max', 'I_max Fehler', 'I_min', 'I_min Fehler', 'Sigma Abstand']
            values = [(roi, b_up, b_up_err, b_low, b_low_err, sig)]
            b = pd.DataFrame(values, columns=col).set_index('roi')
            bounds = pd.concat([bounds,b], axis=0)
    return bounds

def scores(df, bounds, Out_corr_type=None):
    df = df.T
    N, M = df.shape
    res = pd.DataFrame(columns=df.columns)
    for roi in df.index:
        I_max, I_min = bounds.at[roi, 'I_max'], bounds.at[roi, 'I_min']
        res.loc[roi,:] = (df.loc[roi,:] - I_min) /(I_max - I_min)
    if Out_corr_type==None:
        wo_out = res.copy()
        mean, std = res.mean(axis=1), res.std(axis=1)
        res['mean'], res['std'] = mean, std
    return res, wo_out


# %% Referenzmessungen zusammen führen und speichern

read_path = r'C:\Daten\2020_02_22_Finale_Aufwertungen\Referenzmessungen'
files,names = get_files(read_path, '\**LSA_RawData**.csv')


step = 400
Samples = ['CrNiFe', 'Ta_1', 'Ta_2', 'Ni', 'Nb', 'W', 'NdFeB', 'Ti', 'Cu']
colors = ['navy', 'r', 'g', 'b', 'firebrick', 'lime', 'royalblue', 'maroon', 'forestgreen']

collect = []
for counter,file in enumerate(files):
    columnShift = 0
    start = step*counter
    end = step*counter + step
    df = read_from_csv(file,cutoff=1009,sel_area=(1009,'end'))
    N, M = df.shape
    cols = [(samp,num) for samp in Samples for num in np.arange(start,end,1)]
    if len(cols) < M:
        cols.append([cols[-1]]*(M-len(cols)))
    df.columns = pd.MultiIndex.from_tuples(cols[:M],names=['Sample','Number'])
    print(M)
    print(len(cols))
    fig, ax = plt.subplots(nrows=1,ncols=1)
    for sam,c in zip(Samples,colors):
        df_temp = df.loc[:,sam]
        ax.plot(df_temp.columns.values+columnShift, df_temp.iloc[10000,:].values)
        columnShift += df_temp.columns.values[-1] + 1
    ax.grid(True)
    plt.show()
    collect.append(df)
df_total = pd.concat(collect, axis=1)
print(df_total.loc[:,Samples])
fig, ax = plt.subplots(nrows=1,ncols=1)
columnShift = 0
for sam,c in zip(Samples,colors):
    df_temp = df_total.loc[:,sam]
    ax.plot(df_temp.columns.values+columnShift, df_temp.iloc[10000,:].values)
    columnShift += df_temp.columns.values[-1] + 1
ax.grid(True)
plt.show()
df_total = df_total.loc[:,Samples]
df_total = df_total.sort_index(level=1,axis=1).sort_index(level=0,axis=1)
Sample_sequence = df_total.columns.get_level_values(level=0).unique().values
print(df_total.loc[:,'Cu'])
df_tot = df_total.copy()
df_tot.columns = np.arange(0,df_tot.shape[1],1)
print(df_tot)
df_tot.to_csv(read_path+'\All_Measurements_List.csv', sep=';', float_format='%.3f')
df_total.to_csv(read_path+'\All_Measurements_DF.csv', sep=';', float_format='%.3f')
pd.DataFrame(Sample_sequence).to_csv(read_path+'\Sample_sequence.csv', sep=';')  # , float_format='%.3f')

# %%  Berechnung und Speicherung von ROIs und Scores/Bounds auf den Referenzdatan


roi_path = r"C:\Daten\2020_02_22_Finale_Aufwertungen\ROI_Final.cfg"
ROI = read_ROI(roi_path)
df_ROI = apply_ROI(df_total,ROI,alt=False)
print(df_ROI)
s = df_ROI.groupby(['Sample']).agg(['mean','std']).T
print(s)
bounds = score_bounds(s)
print(bounds)
print(bounds.loc[(bounds['Sigma Abstand']>3.0) & (bounds['Sigma Abstand']<4.0 )])
print(bounds.loc[(bounds['Sigma Abstand']>4.0) & (bounds['Sigma Abstand']<5.0 )])
print(bounds.loc[(bounds['Sigma Abstand']>5.0)])

bounds_path = r'C:\Daten\2020_02_22_Finale_Aufwertungen\Referenzmessungen\Scores_und_Bounds'

desriptor = r'new_ROI_calc'
df_ROI.to_csv(bounds_path + f'\\{desriptor}.csv', sep=';', float_format='%.3f')
s.to_csv(bounds_path + f'\\scores_{desriptor}.csv', sep=';', float_format='%.3f')

bounds.to_csv(bounds_path+ f'\\bounds_{desriptor}.csv', sep=';', float_format='%.3f')

# %% Berechnung aller Messungs ROIs und scores

read_path = r'C:\Daten\2020_02_22_Finale_Aufwertungen\Type_variable_FoilPenetration'
files,names = get_files(read_path, '\AluBaseplate**LSA_RawData**.csv', recursive=True)

roi_path = r"C:\Daten\2020_02_22_Finale_Aufwertungen\ROI_Final.cfg"
ROI = read_ROI(roi_path)
bounds_alt_path = r"C:\Daten\2020_02_22_Finale_Aufwertungen\Referenzmessungen\Scores_und_Bounds\bounds_old_ROI_calc.csv"
bounds_neu_path = r"C:\Daten\2020_02_22_Finale_Aufwertungen\Referenzmessungen\Scores_und_Bounds\bounds_new_ROI_calc.csv"
bounds_alt = pd.read_csv(bounds_alt_path, sep=';').set_index('roi')
bounds_neu = pd.read_csv(bounds_neu_path, sep=';').set_index('roi')
print(bounds_alt)

for file in files:
    name = file.split(os.sep)[-1]
    savepath = file.replace(name,'') + 'ROIs_and_Scores'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    print(savepath)
    df_RawData = read_from_csv(file,cutoff=1009,sel_area=(1009,'end'))
    for a in [True,False]:
        df_ROIs = apply_ROI(df_RawData,ROI, alt=a)
        if a:
            desriptor = 'alt'
            scs, wo_out = scores(df_ROIs, bounds_alt)
        else:
            desriptor = 'neu'
            scs, wo_out = scores(df_ROIs, bounds_neu)
        df_ROIs.to_csv(savepath + f'\\ROI_{desriptor}_{name}', sep=';', float_format='%.3f')
        scs.to_csv(savepath + f'\\scores_{desriptor}_{name}', sep=';', float_format='%.3f')



