# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:20:59 2020

@author: BALHORN
"""

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from scipy import optimize, signal
from scipy.special import wofz
import time
from scipy.interpolate import UnivariateSpline
from scipy.stats import gaussian_kde
import os
import csaps
import gc
from multiprocessing import Pool, freeze_support
import multiprocessing
from functools import partial
import glob

plt.rcParams.update({'font.size': 10})

# %% Splinefit Hintergrundkorrektur


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

def df_cut_wo_exc(df, padding=0):
    cutlines = np.arange(0,len(df.index.values)+200,2048)
    dfs = []
    for cut1,cut2 in zip(cutlines[:-1], cutlines[1:]):
        dfs += [df.iloc[cut1+padding:cut2-padding,:]]
    return dfs

def hintergrund_fit_korrektur(path, kind='both', cutoff=1010,
                              sel_area=(1010,'end'), peak_widths=(1, 5),
                              butter=(3, 0.05)):
    """
    Spline fit as the approximation of the background radiation in the plasma
    is substraced from the spectrums
    """
    sapa = r'V:\A11\2019\DIPLOM\BALHORN\Bilder\Splinefit\SplineFitTitanium.png'
    name = path.split(os.sep)[-1]
    name2 = path.split(os.sep)[-1].replace('.csv','')
    sapa = path.replace(name,f'\Spline_Korrektur\\{name2}')
    print(sapa)
    if kind=='linear Interpolation' or kind =='both':
        lin_int_df_CCD = []
    if kind=='Spline' or kind=='both':
        spline_df_CCD = []
    df = read_from_csv(path, cutoff=cutoff, sel_area=sel_area)
    dfs = df_cut_wo_exc(df,padding=0)
    for ix in [3]: # range(len(dfs)):
        count=0
        print(f'Section {ix} von {len(dfs)}')
        columns = list(dfs[ix].columns.values)
        if kind=='linear Interpolation' or kind =='both':
            lin_int_df = []
        if kind=='Spline' or kind=='both':
            spline_df = []
        for col in columns:
            ydata = dfs[ix].loc[:,col].values
            xdata = dfs[ix].loc[:,col].index.values
            b, a = signal.butter(butter[0], butter[1])
            filtered = signal.filtfilt(b, a, ydata*(-1))
            peak_indicies = signal.find_peaks_cwt(filtered, peak_widths, min_length=1)
            new_peaks = []
            win=20
            for ind in peak_indicies:
                if ind < win:
                    win = ind
                if len(ydata)-ind < win:
                    win = len(ydata)-ind
                sl = ydata[ind-win:ind+win]
                new_peaks += [np.argmin(sl)+ind-win]
            new_peaks = list(dict.fromkeys(sorted(new_peaks)))
            xd = xdata.copy()
            if kind=='linear Interpolation' or kind=='both':
                lin_int_df += [pd.DataFrame(zip(xdata,ydata-np.interp(xd,xdata[new_peaks],ydata[new_peaks])),
                                            columns=['WL',col]).set_index('WL')]
            if kind=='Spline' or kind=='both':
                sp = csaps.UnivariateCubicSmoothingSpline(xdata[new_peaks],ydata[new_peaks], smooth=0.45)
                spline_df += [pd.DataFrame(zip(xdata,ydata-sp(xd)),
                                           columns=['WL',col]).set_index('WL')]
                if count==0:
                    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(5,4),sharex=True)
                    ax[0].set_title('Titanium Spectrum')
                    ax[0].plot(xdata,ydata,color='navy',label='Data')
                    ax[0].plot(xdata[new_peaks],ydata[new_peaks],color='orange',
                               ls='none',marker='o',label='Minima')
                    ax[0].plot(xdata,sp(xdata),color='red',label='Spline')
                    #ax[0].set_title('Spline-Fit')
                    ax[0].set_ylabel('Intensity I [a.u.]')
                    ax[0].grid()
                    ax[0].legend()

                    ax[1].plot(xdata,ydata-sp(xd),color='navy',label='corrected Data')
                    #ax[1].set_title('Korrected Data')
                    ax[1].set_ylabel('Intensity I [a.u.]')
                    ax[1].set_xlabel('Wavelenght $\lambda$ [nm]')
                    ax[1].grid()
                    ax[1].legend()
                    plt.tight_layout()
                    plt.savefig(sapa, dpi=600)
                    plt.show()
            else:
                print('Zum Hintergrundfit sind nur "linear Interpolation", "Spline" und "both" implementiert')
            count += 1
        if kind=='linear Interpolation' or kind=='both':
            lin_int_df = pd.concat(lin_int_df,axis=1)
            lin_int_df_CCD += [lin_int_df]
        if kind=='Spline' or kind=='both':
            spline_df = pd.concat(spline_df, axis=1)
            spline_df_CCD += [spline_df]
    # if kind=='linear Interpolation' or kind=='both':
    #     lin_int_res = pd.concat(lin_int_df_CCD,axis=0)
    #     lin_int_res.to_csv(sapa + '_lin_int_corr.csv', sep=';', float_format='%.10f')
    # if kind=='Spline' or kind=='both':
    #     spline_res = pd.concat(spline_df_CCD, axis=0)
    #     spline_res.to_csv(sapa + '_spline_corr.csv', sep=';', float_format='%.10f')
    return





paths,names = get_files(r'C:\Daten\2020_02_22_Finale_Aufwertungen\Referenzmessungen',
                  r'\**LSA_RawData**.csv')
print('Beginn')
for p in paths:
    print('Ausgewählter Pfad:')
    print(p)
    hintergrund_fit_korrektur(p, kind='Spline', cutoff=1010,
                              sel_area=(3800,4200), peak_widths=(1, 5),
                              butter=(3, 0.05))


# %% Peakfit


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

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def Lorenz(x, A, mu, sigma):
    return A/np.pi * (sigma)/( np.power(x-mu, 2) + np.power(sigma, 2) )

def Gauss(x, A, mu, sigma_g):
    return A/( sigma_g * np.power( 2*np.pi, 0.5) ) * np.exp( -np.power(x-mu, 2)/(2*np.power(sigma_g, 2)) )

def PseudoVoigt(x, A, mu, sigma, alpha):
    sigma_g = sigma / np.power( 2*np.log(2), 0.5)
    lorenz = Lorenz(x, A, mu,  sigma)
    gauss = Gauss(x, A, mu, sigma_g)
    PsVoigt= (1-alpha)*gauss + alpha*lorenz
    return PsVoigt

def Voigt(x, A , mu, sigma, gamma):
    sigma_v = sigma / np.sqrt(2 * np.log(2))
    return A*np.real(wofz((x-mu + 1j*gamma)/sigma_v/np.sqrt(2))) / sigma_v/np.sqrt(2*np.pi)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def apply_padding(x, ix, sig, factor):
    pad = int(factor*sig/0.018976433237682402)
    end = len(x)
    if (ix-pad) < 0:
        st = 0
    else:
        st = ix-pad
    if (ix+pad) > end:
        ed = end
    else:
        ed = ix+pad
    return st,ed

def make_func(numarg, model):
    """
    returnes a fit function with the selected model
    """
    def func(x,*a):
        ng=numarg
        v=np.zeros(len(x))
        if model=='Gauss':
            for i in range(0,ng,3):
                c_idx = find_nearest(x, a[i+1])
                start,ende = apply_padding(x,c_idx,a[i+2], 6)
                v[start:ende]+=Gauss(x[start:ende], a[i], a[i+1], a[i+2])
        elif model=='Lorenz':
            for i in range(0,ng,3):
                c_idx = find_nearest(x, a[i+1])
                start,ende = apply_padding(x,c_idx,a[i+2], 12)
                v[start:ende]+=Lorenz(x[start:ende], a[i], a[i+1], a[i+2])
        elif model=='PseudoVoigt':
            for i in range(0,ng,4):
                c_idx = find_nearest(x, a[i+1])
                start,ende = apply_padding(x,c_idx,a[i+2], 10)
                v[start:ende]+=PseudoVoigt(x[start:ende], a[i], a[i+1], a[i+2], a[i+3])
        elif model=='Voigt':
            for i in range(0,ng,4):
                c_idx = find_nearest(x, a[i+1])
                start,ende = apply_padding(x,c_idx, np.sqrt(8*np.log(2))*a[i+2], 40)
                v[start:ende]+=Voigt(x[start:ende], a[i], a[i+1], a[i+2], a[i+3])
        else:
            print(f'Model {model} unknowen. Only Gauss, Lorenz, PseudoVoight and Voigt are implemented.')
        return v
    return func

def guess_form_peaks(x,y,minimum,peak_widths=(1, 5),butter=(3, 0.1), param_number_3=False):
    """
    guesses start parameters for the peakfit
    """
    param = []
    bounds_bottom = []
    bounds_top = []
    x_range = np.max(x) - np.min(x)
    b, a = signal.butter(butter[0], butter[1])
    filtered = signal.filtfilt(b, a, y)
    peak_indicies = signal.find_peaks_cwt(filtered, np.arange(peak_widths[0],peak_widths[1]),min_snr=3) # , min_length=minimum
    peak_find_indeces,_ = signal.find_peaks(filtered, height=movingaverage(filtered, 20))
    np.random.shuffle(peak_indicies)
    y_mean = np.mean(y)
    #for peak_indicie in peak_indicies.tolist():
    for peak_indicie in peak_find_indeces.tolist():
        A = y[peak_indicie]/5
        A_bottom, A_top = 1e-6, 1.1*abs(y[peak_indicie])
        if A < A_bottom or A > A_top:
            A = 0.8*abs(y[peak_indicie])

        if y[peak_indicie] > 1.5*y_mean:
            sigma =  x_range / len(x) * np.min(peak_widths) # np.min(peak_widths)*0.25  # x_range/ len(x) * np.min(peak_widths)
            alpha = 0.05
        else:
            sigma =  x_range / len(x) * np.min(peak_widths)
            alpha = 0.05

        sigma_bottom, sigma_top = x_range / len(x) * np.min(peak_widths), x_range
        if sigma < sigma_bottom or sigma > sigma_top:
            sigma = sigma_bottom + 0.1*(sigma_top-sigma_bottom)

        mu = x[peak_indicie]

        mu_bottom, mu_top = np.min(x), np.max(x)

        alpha_bottom, alpha_top = 0.0, 1.0

        if param_number_3:
            param += [A,mu,sigma]
            bounds_bottom += [A_bottom,mu_bottom,sigma_bottom]
            bounds_top += [A_top,mu_top,sigma_top]
        else:
            param += [A,mu,sigma,alpha]
            bounds_bottom += [A_bottom,mu_bottom,sigma_bottom,alpha_bottom]
            bounds_top += [A_top,mu_top,sigma_top,alpha_top]

    return tuple(param), (bounds_bottom, bounds_top) ,peak_indicies, filtered, peak_find_indeces

def results_to_peaks(res):
    N = int(len(res)/4)
    Peak_params = res.reshape((N,4))
    return Peak_params

def fit_plot(xdata,ydata,popt,title=None, ax=None, model=None):
    """
    plots data and fits
    """
    if ax==None:
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(xdata,ydata,label='Data',color='b')
        total = np.zeros(len(xdata))
        for i in range(0,len(popt),4):
                c_idx = find_nearest(xdata, popt[i+1])
                start,ende = apply_padding(xdata,c_idx,popt[i+2])
                currentPeak=PseudoVoigt(xdata[start:ende], popt[i], popt[i+1], popt[i+2], popt[i+3])
                total[start:ende] += currentPeak
                ax.plot(xdata[start:ende],currentPeak)
        ax.plot(xdata, total, label='Fit',color='k')
        ax.legend()
        plt.show()
    else:
        ax.plot(xdata,ydata,label='Data',color='b')
        if title is not None:
            ax.set_title(title)
        total = np.zeros(len(xdata))
        a = popt
        ng = len(a)
        x = xdata
        if model=='Gauss':
            for i in range(0,ng,3):
                c_idx = find_nearest(x, a[i+1])
                start,ende = apply_padding(x,c_idx,a[i+2], 6)
                currentPeak=Gauss(x[start:ende], a[i], a[i+1], a[i+2])
                total[start:ende] += currentPeak
                ax.plot(xdata[start:ende],currentPeak)
        elif model=='Lorenz':
            for i in range(0,ng,3):
                c_idx = find_nearest(x, a[i+1])
                start,ende = apply_padding(x,c_idx,a[i+2], 12)
                currentPeak=Lorenz(x[start:ende], a[i], a[i+1], a[i+2])
                total[start:ende] += currentPeak
                ax.plot(xdata[start:ende],currentPeak)
        elif model=='PseudoVoigt':
            for i in range(0,ng,4):
                c_idx = find_nearest(x, a[i+1])
                start,ende = apply_padding(x,c_idx,a[i+2], 10)
                currentPeak=PseudoVoigt(x[start:ende], a[i], a[i+1], a[i+2], a[i+3])
                total[start:ende] += currentPeak
                ax.plot(xdata[start:ende],currentPeak)
        elif model=='Voigt':
            for i in range(0,ng,4):
                c_idx = find_nearest(x, a[i+1])
                start,ende = apply_padding(x,c_idx,np.sqrt(8*np.log(2))*a[i+2], 40)
                currentPeak=Voigt(x[start:ende], a[i], a[i+1], a[i+2], a[i+3])
                total[start:ende] += currentPeak
                ax.plot(xdata[start:ende],currentPeak)
        elif model==None:
            pass
        else:
            print(f'Model {model} unknowen. Only Gauss, Lorenz, PseudoVoight and Voigt are implemented.')
        ax.plot(xdata, total, label='Fit',color='k')
        ax.legend()
    return

def create_dataframe(popt, model):
    """
    creates DataFrame with the right shape and collumn names for the selected model
    """
    if model=='Gauss':
        res = popt.reshape((int(len(popt)/3),3)).T
        counter = np.arange(0,int(len(popt)/3),1)
        results = list(zip(counter,res[0],res[1],res[2]))
        return pd.DataFrame(results, columns=['Peak','Amplitude','Center','Cigma']).set_index('Peak')
    elif model=='Lorenz':
        res = popt.reshape((int(len(popt)/3),3)).T
        counter = np.arange(0,int(len(popt)/3),1)
        results = list(zip(counter,res[0],res[1],res[2]))
        return pd.DataFrame(results, columns=['Peak','Amplitude','Center','Cigma']).set_index('Peak')
    elif model=='PseudoVoigt':
        res = popt.reshape((int(len(popt)/4),4)).T
        counter = np.arange(0,int(len(popt)/4),1)
        results = list(zip(counter,res[0],res[1],res[2],res[3]))
        return pd.DataFrame(results, columns=['Peak','Amplitude','Center','Cigma','Alpha']).set_index('Peak')
    elif model=='Voigt':
        res = popt.reshape((int(len(popt)/4),4)).T
        counter = np.arange(0,int(len(popt)/4),1)
        results = list(zip(counter,res[0],res[1],res[2],res[3]))
        return pd.DataFrame(results, columns=['Peak','Amplitude','Center','Sigma','Gamma']).set_index('Peak')
    elif model==None:
        return None
    else:
        print(f'Model {model} unknowen. Only Gauss, Lorenz, PseudoVoight and Voigt are implemented.')
        return None

def run_ROI_fit(df_intern, minimum=1, butter=(6, 0.05), pa=[], bound=[], fit_type='Voigt'):
    """
    This function runs the fit of multiple ajacent peaks after the selected
    line profile
    """
    xdata = df_intern.index.values
    ydata = df_intern.values
    param4, bounds4, peaks4, filtered4, peak_2 = guess_form_peaks(
        xdata,ydata, minimum,butter=butter,param_number_3=False)
    param3, bounds3, peaks3, filtered3, peak_2 = guess_form_peaks(
        xdata,ydata, minimum,butter=butter,param_number_3=True)
    try:
        if fit_type=='Gauss' or fit_type=='all':
            popt_Gauss, pcov_Gauss = curve_fit(
                make_func(len(param3),'Gauss'), xdata, ydata,
                p0=param3, bounds=bounds3)
        if fit_type=='Lorenz' or fit_type=='all':
            popt_Lorenz, pcov_Lorenz = curve_fit(
                make_func(len(param3),'Lorenz'), xdata, ydata,
                p0=param3, bounds=bounds3)
        if fit_type=='PseudoVoigt' or fit_type=='all':
            popt_PseudoVoigt, pcov_PseudoVoigt = curve_fit(
                make_func(len(param4),'PseudoVoigt'), xdata, ydata,
                p0=param4, bounds=bounds4)
        if fit_type=='Voigt' or fit_type=='all':
            popt_Voigt, pcov_Voigt = curve_fit(
                make_func(len(param4),'Voigt'), xdata, ydata,
                p0=param4, bounds=bounds4)
    except (np.linalg.LinAlgError,RuntimeError) as err:
        print('Error ocurred: ', err)
        return False, None

    if fit_type=='Gauss':
        return True, (
            create_dataframe(popt_Gauss, model='Gauss'),
            filtered3, create_dataframe(np.asarray(param3), model='Gauss'),
            peaks3, peak_2)
    if fit_type=='Lorenz':
        return True, (
            create_dataframe(popt_Lorenz, model='Lorenz'),
            filtered3,create_dataframe(np.asarray(param3), model='Lorenz'),
            peaks3, peak_2)
    if fit_type=='PseudoVoigt':
        return True, (
            create_dataframe(popt_PseudoVoigt, model='PseudoVoigt'),
            filtered4,create_dataframe(np.asarray(param4), model='PseudoVoigt'),
            peaks4, peak_2)
    if fit_type=='Voigt':
        return True, (
            create_dataframe(popt_Voigt, model='Voigt'), filtered4,
            create_dataframe(np.asarray(param4), model='Voigt'),
            peaks4, peak_2)
    if fit_type=='all':
        res_Gauss = create_dataframe(popt_Gauss, model='Gauss')
        res_Lorenz = create_dataframe(popt_Lorenz, model='Lorenz')
        res_PseudoVoigt = create_dataframe(popt_PseudoVoigt, model='PseudoVoigt')
        res_Voigt = create_dataframe(popt_Voigt, model='Voigt')
        return True, (res_Gauss, res_Lorenz, res_PseudoVoigt, res_Voigt,
                      filtered3, filtered4, peaks3, peaks4, peak_2)
    else:
        print(f'Invalid fit_type: {fit_type}. Select from: Gauss, Lorenz, PseudoVoigt, Voigt or all')
        return False, None

def sel_peak(PeakParam, xdat):
    """
    checks if the fitted peak is the intended emission line, based on the
    distance between the center of the intended emission line and the center
    the fitted peak.
    
    function also handels multiple results with same distance.
    
    The selected voight profile is summed.
    """
    A = PeakParam.loc[PeakParam['where'] == min(PeakParam['where']),'Amplitude'].values
    Center = PeakParam.loc[PeakParam['where'] == min(PeakParam['where']),'Center'].values
    Sigma = PeakParam.loc[PeakParam['where'] == min(PeakParam['where']),'Sigma'].values
    Gamma = PeakParam.loc[PeakParam['where'] == min(PeakParam['where']),'Gamma'].values
    line_dist = min(PeakParam['where'])
    if len(A)>1:
        ind_max = np.argmax(A)
        A = np.array([A[ind_max]])
        Center = np.array([Center[ind_max]])
        Sigma = np.array([Sigma[ind_max]])
        Gamma = np.array([Gamma[ind_max]])
        line_dist = np.array([line_dist[ind_max]])
    binning_width = np.mean(np.diff(xdat))
    line_int = np.sum(Voigt(xdat, A , Center, Sigma, Gamma))*binning_width
    return line_int, line_dist, A, Center, Sigma, Gamma


def fit_ROI(tup,df,ROI,datei):
    """
    woker that enabels the parallelization of fitting for multiple
    emission lines at the same time.
    All data prep and fit evaluation is doen here
    """
    roi, peak_to_find, roi_factor = tup
    if roi_factor==None:
        ixs = list(df.columns.values)
        no_data = list(zip(ixs,[None]*len(ixs)))
        no_res = pd.DataFrame(no_data, columns=['Counter',roi]).set_index('Counter')
        no_res.index = no_res.index.astype(int)
        no_res.sort_index(inplace=True)
        return no_res
    df_roi = pd.DataFrame()
    st = ROI.at[roi,'Start Integration']
    ed = ROI.at[roi,'Ende Integration']
    st_bg = ROI.at[roi,'Start Integration_BG']
    ed_bg = ROI.at[roi,'Ende Integration_BG']
    bottom = np.min([st,ed,st_bg,ed_bg])
    top = np.max([st,ed,st_bg,ed_bg])
    padding = 0.3
    lower_bound = bottom - padding
    upper_bound = top + padding
    if len(df.loc[lower_bound:upper_bound,df.columns.values[0]]) < 63:
        lower_bound = bottom - padding*1.5
        upper_bound = top + padding*1.5
        df_fit = df.loc[lower_bound:upper_bound,:]
    else:
        df_fit = df.loc[lower_bound:upper_bound,:]
    for col in df.columns.values:
        xdat = df_fit.loc[:,col].index.values
        success, tup = run_ROI_fit(df_fit.loc[:,col], butter=(20,0.3)) # PeakParam DataFrame of PeakParam
        if success:
            PeakParam, filtered, ini_param, found_peaks, alternative_peaks = tup
        else:
            continue
        if peak_to_find:
            PeakParam['where'] = abs((PeakParam['Center']-peak_to_find[0]))
            line_int, line_dist, A, Center, Sigma, Gamma = sel_peak(PeakParam, xdat)
        df_temp = pd.DataFrame(zip([line_int],[line_dist]), columns=[(roi,'line_int'),(roi,'line_dist')],index=[col])
        df_roi = df_roi.append(df_temp)
    df_roi.index.name = 'Counter'
    df_roi.index = df_roi.index.astype(int)
    df_roi.sort_index(inplace=True)
    return df_roi

factor_best_result = {
    'Cr_521':0.1,'Cu_521':0.1,'Fe_275':0.1,'Nb_408':0.4,'Nd_430':0.2,
    'Ni_251':0.5,'Ni_310':0.6,'Ni_341':0.2,'Ni_343':0.3,'Ni_345':0.2,
    'Ni_380':None,'Ni_385':None,'Ni_400':0.8,'Ta_284':None,'Ta_300':0.2,
    'Ta_315':0.8,'Ta_331':0.4,'Ta_332':0.4,'Ta_339':0.2,'Ta_383':None,
    'Ta_392':None,'Ta_401':None,'Ta_441':0.6,'Ta_451':None,'Ta_453':0.4,
    'Ta_541':0.5,'Ti_282':0.1,'Ti_306':0.2,'Ti_307':0.3,'Ti_308':0.2,
    'Ti_309':0.2,'Ti_316':0.3,'Ti_319':0.2,'Ti_320':0.3,'Ti_326':0.2,
    'Ti_328':0.3,'Ti_335':0.5,'Ti_338':0.1,'Ti_375':0.1,'Ti_390':0.1,
    'Ti_430':0.1,'Ti_439':0.1,'W_334':None,'W_407':0.2,'W_426':0.1,
    'W_429':0.2
        }

rpaths_ref, rnames_ref = get_files(r'C:\Daten\2020_02_22_Finale_Aufwertungen\Referenzmessungen\Spline_Korrektur',
                           r'\**LSA_RawData**_spline_corr.csv')

sp_ref = r"C:\Daten\2020_02_22_Finale_Aufwertungen\Referenzmessungen\Fits_und_Bounds"
spaths_ref = [sp_ref+f'\{r.replace("_spline_corr.csv","_fit_line_ints_and_dists.csv")}' for r in rnames_ref]

rpaths, rnames = get_files(r'C:\Daten\2020_02_22_Finale_Aufwertungen\Type_A_Gridsize\Spline_Korrektur',
                           r'\**LSA_RawData**_spline_corr.csv')

sp = r"C:\Daten\2020_02_22_Finale_Aufwertungen\Type_A_Gridsize\Fits_and_Scores"
spaths = [sp+f'\{r.replace("_spline_corr.csv","_fit_line_ints_and_dists.csv")}' for r in rnames]


rpaths = rpaths_ref + rpaths
spaths = spaths_ref + spaths

rpaths, rnames = get_files(r'C:\Daten\2020_02_22_Finale_Aufwertungen\Type_variable_FoilPenetration',
                           r'\AluBaseplate**LSA_RawData**_spline_corr.csv')

sp = r"C:\Daten\2020_02_22_Finale_Aufwertungen\Type_variable_FoilPenetration\Fits_und_Bounds"
spaths = [sp+f'\{r.replace("_spline_corr.csv","_fit_line_ints_and_dists.csv")}' for r in rnames]


print(spaths)


asd_dict = {'Cr_521': (520.82, 0.06), 'Cu_521': (521.8, 0.06), 'Fe_275': (274.92, 0.06), 'Nb_408': (407.84, 0.06), 'Nd_430': (430.34, 0.06), 'Ni_251': (251.25, 0.06), 'Ni_310': (310.12, 0.06), 'Ni_341': (341.47, 0.06), 'Ni_343': (343.36, 0.06), 'Ni_345': (345.85, 0.06), 'Ni_380': (380.72, 0.06), 'Ni_385': (385.84, 0.06), 'Ni_400': (440.18, 0.06), 'Ta_284': (284.35, 0.06), 'Ta_300': (301.2, 0.06), 'Ta_315': (315.71, 0.06), 'Ta_331': (331.09, 0.06), 'Ta_332': (331.83, 0.06), 'Ta_339': (340.69, 0.06), 'Ta_383': (383.9, 0.06), 'Ta_392': (392.15, 0.06), 'Ta_401': (402.95, 0.06), 'Ta_441': (441.57, 0.06), 'Ta_451': (451.95, 0.06), 'Ta_453': (453.09, 0.06), 'Ta_541': (542.05, 0.06), 'Ti_282': (282.84, 0.06), 'Ti_306': (307.24, 0.06), 'Ti_307': (307.46, 0.06), 'Ti_308': (307.82, 0.06), 'Ti_309': (308.75, 0.06), 'Ti_316': (316.18, 0.06), 'Ti_319': (319.03, 0.06), 'Ti_320': (320.19, 0.06), 'Ti_326': (326.1, 0.06), 'Ti_328': (328.74, 0.06), 'Ti_335': (334.87, 0.06), 'Ti_338': (338.34, 0.06), 'Ti_375': (375.9, 0.06), 'Ti_390': (390.09, 0.06), 'Ti_430': (430.03, 0.06), 'Ti_439': (439.52, 0.06), 'W_334': (334.41, 0.06), 'W_407': (407.31, 0.06), 'W_426': (426.91, 0.06), 'W_429': (429.44, 0.06)}

print('Hier')
roi_path = r"C:\Daten\2020_02_22_Finale_Aufwertungen\ROI_Final.cfg"
ROI = read_ROI(roi_path)
nist_path = r"M:\A11\2019\MESSDATEN\MATERIALANALYSE\NIST ASD 20181130\asd.txt"
roi_l = list(asd_dict.keys())
print(roi_l)

if __name__ ==  '__main__':
    count=0
    for rp, sp in zip(rpaths,spaths):
        print('Neue Datei:')
        print(rp)
        print(sp)
        df = pd.read_csv(rp ,sep=';').set_index('WL')
        print('Datei gelesen:')
        print(df)
        args = list(zip(roi_l,[asd_dict[l] for l in roi_l], [factor_best_result[l] for l in roi_l]))
        print('Argumente:')
        print(args)
        print('Anfang Fits')
        func_partial = partial(fit_ROI, df=df, ROI=ROI, datei=count)
        with Pool(processes=3) as p:
            output = p.map(func_partial, args)
        print('Ende Fits')
        output = pd.concat(output,axis=1)
        print('Speichern')
        output.to_csv(sp, sep=';', float_format='%.10f')
        print(output)
        count += 1
    gc.collect()

