# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:17:59 2019

@author: BALHORN
"""

import numpy as np
import pandas as pd
import os
import glob
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as wdg
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import simpledialog
from collections.abc import Iterable

global lx1
lx1 = False
global lx2
lx2 = False
global lx3
lx3 = False
global lx4
lx4 = False
global pl_sp
pl_sp = False
global pl_sp_H
pl_sp_H = False
global df_nist
df_nist = False
global nist
nist = {}
global spektrum_df
spektrum_df = False
global H_spektrum_df
H_spektrum_df = False
global lines
lines = []
global texts
texts = []
global rois
rois = []
global header
header = 'ROI-Name                  ; Start Auslesebereich ; Ende Auslesebereich  ; Start Integration    ; Ende Integration     ; min                  ; max    '
global axrois
axrois = {}
global marked
marked = ''
global H_spa_paths
H_spa_paths = []
global spec_max
spec_max = 0.0

def read_spas(files):
    df = []
    for ix, f in enumerate(files):
        df_temp = pd.read_csv(f, sep=' ', names=['Wavelength', ix],
                              header=1, dtype=float)
        # display(df_temp)
        df_temp.set_index(['Wavelength'], inplace=True)
        df = df + [df_temp]
    df = pd.concat(df, axis=1)
    print('df.index.is_unique', df.index.is_unique)
    print('df.index.dtype', df.index.dtype)
    print('df.index[df.index.duplicated]', df[df.duplicated()].index.unique)
    return df


def read_nist(file):
    
    df_ = pd.read_csv(file, sep=';')
    df_['element'] = [e.split(' ')[0] for e in df_['Spectrum']]
    df_.set_index(['element', 'Spectrum'], inplace=True)
    df_.sort_index(inplace=True)
    df_.astype(np.float)
    return df_


def list_remover(poss_list):
    if isinstance(poss_list, list):
        for elem in poss_list:
            list_remover(elem)
    else:
        poss_list.remove()
    return


def show_nist(f, a, df_, elem, ionstate, rel):
    global lines
    global texts
    global df_nist
    if not isinstance(df_nist, pd.DataFrame):
        err = 'Datei der NIST ASD Datenbank einlesen!'
        messagebox.showerror('Error', err)
        return
    if elem == '':
        if lines:
            list_remover(lines)
        if texts:
            list_remover(texts)
        return
    else:
        # el = '|'.join([e.strip() for e in elem.split(',')])
        elements = [e.strip() for e in elem.split(',')]
        io = '|'.join([i.strip() for i in ionstate.split(',')])
        start = float(list(spektrum_df.index)[0])
        ende = float(list(spektrum_df.index)[-1])
        # con_elem = df_.index.str.contains(el, na=False, regex=True)
        # df_['Ionisationsgrad'].astype(str, inplace=True)
        df_['Rel. Int.'].astype(float, inplace=True)
        df_e = df_.loc[elements,:]
        con_ion = df_e['Ionisationsgrad'].astype(str).str.contains(
                io, na=False, regex=True)
        nist = df_e[(con_ion)
                   & (df_e['Ritz Wavelength Vac (nm)'] >= start)
                   & (df_e['Ritz Wavelength Vac (nm)'] <= ende)
                   & (df_e['Rel. Int.'] > float(rel))]
# =============================================================================
#         nist = df_[(con_elem) & (con_ion)
#                    & (df_['Ritz Wavelength Vac (nm)'] >= start)
#                    & (df_['Ritz Wavelength Vac (nm)'] <= ende)
#                    & (df_['Rel. Int.'] > float(rel))]
# =============================================================================
        if lines:
            list_remover(lines)
        if texts:
            list_remover(texts)
        lines = []
        texts = []
        if len(nist) == 0:
            f.canvas.draw()
            return
        for i,ix in enumerate(nist.index.unique()):
            lines.append([])
            texts.append([])
            wls = nist.at[ix, 'Ritz Wavelength Vac (nm)']
            if type(wls) is not np.ndarray:
                wls = np.array(wls)
            if len(wls.shape) == 0 :
                continue
            for jx, wl_nist in enumerate(wls):
                lines[i].append([])
                texts[i].append([])
                wl = spektrum_df.index.get_loc(
                        float(wl_nist), method='nearest')
                height = spektrum_df.at[
                        list(spektrum_df.index)[wl], 'mean']
                lines[i][jx] = plt.vlines(x=wl_nist,
                                           ymin=height+10, ymax=height+300)
                s = ix[1] + ' {:.4f}'.format(wl_nist)
                texts[i][jx] = plt.text(x=wl_nist, y=height+310, s=s,
                                         fontsize=6, rotation=90,
                                         verticalalignment='bottom')
        f.canvas.draw()
        return


def nist_b(f, a, elements, ionstate, rel):
    nist_path = tk.filedialog.askopenfilename()
    if nist_path == '':
        return
    filename = nist_path.split(r'\\')[-1]
    nist_entry.delete(0, tk.END)
    nist_entry.insert(0, filename)
    global df_nist
    if df_nist:
        df_nist = False
    df_nist = read_nist(nist_path)
    show_nist(f, a, df_nist, elements, ionstate, rel)
    return


def spa_b(f, a):
    spa_paths = tk.filedialog.askopenfilenames()
    if spa_paths == '':
        return
    filename = spa_paths[0].split(r'\\')[-1]
    spa_entry.delete(0, tk.END)
    spa_entry.insert(0, filename)
    global spektrum_df
    if not isinstance(spektrum_df, pd.DataFrame):
        spektrum_df = False
    spektrum_df = read_spas(spa_paths)
    spektrum_df['mean'] = spektrum_df.mean(axis=1)
    global pl_sp
    if pl_sp:
        for elem in pl_sp:
            elem.remove()
        #pl_sp.remove()
    pl_sp = a.plot(spektrum_df.index, spektrum_df['mean'], 'b', zorder=10)
    global spec_max
    ma = spektrum_df['mean'].max()+2000
    if ma >= spec_max:
        spec_max = ma
    plt.ylim(0.0,spec_max)
    plt.xlim(spektrum_df.index.min()-10, spektrum_df.index.max()+10)
    plt.grid(True)
    f.canvas.draw()
    toolbar.update()
    return

def H_spa_b(f, a, add=True, clear=False):
    global H_spa_paths
    global pl_sp_H
    if clear:
        if pl_sp_H:
            for elem in pl_sp_H:
                elem.remove()
            pl_sp_H = False
        H_spa_paths = []
        f.canvas.draw()
        toolbar.update()
        return
    if add:
        spec_paths = tk.filedialog.askopenfilenames()
        if spec_paths == '':
            return
        H_spa_paths += spec_paths
        filename = H_spa_paths[0].split(r'\\')[-1]
        H_spa_entry.delete(0, tk.END)
        H_spa_entry.insert(0, filename)
    if not add:
        global H_spektrum_df
        if not isinstance(H_spektrum_df, pd.DataFrame):
            H_spektrum_df = False
        H_spektrum_df = read_spas(H_spa_paths)
        H_spektrum_df['mean'] = H_spektrum_df.mean(axis=1)
        if pl_sp_H:
            print(pl_sp_H)
            for elem in pl_sp_H:
                elem.remove()
            #pl_sp.remove()
        pl_sp_H = a.plot(H_spektrum_df.index, H_spektrum_df['mean'], 'orange', zorder=0)
        global spec_max
        ma = H_spektrum_df['mean'].max()+2000
        if ma >= spec_max:
            spec_max = ma
        plt.ylim(0.0, spec_max)
        plt.xlim(H_spektrum_df.index.min()-10, H_spektrum_df.index.max()+10)
        plt.grid(True)
        f.canvas.draw()
        toolbar.update()
        return


def plot_rois(f, rois):
    global axrois
    if bool(axrois):
        for r in axrois.values():
            r[0][0].remove()
            r[0][1].remove()
            r[0][2].remove()
            r[0][3].remove()
        axrois = {}
    for p in list(rois.index.str.strip().values):
        wa = rois.at[p, 'Start Integration']
        pa = plt.axvline(x=wa, color='r', lw=1, ls='-.', alpha=0.1)
        wb = rois.at[p, 'Ende Integration']
        pb = plt.axvline(x=wb, color='r', lw=1, ls='-.', alpha=0.1)
        wc = rois.at[p, 'Start Integration BG']
        pc = plt.axvline(x=wc, color='k', lw=1, ls='-.', alpha=0.1)
        wd = rois.at[p, 'Ende Integration BG']
        pd = plt.axvline(x=wd, color='k', lw=1, ls='-.', alpha=0.1)
        axrois[p] = [[pa, pb, pc, pd], [wa,wb,wc,wd]]
    f.canvas.draw()
    return

def cfg_b(f):
    global rois
    cfg_path = tk.filedialog.askopenfilename()
    if cfg_path == '':
        return
    if os.path.getsize(cfg_path) == 0:
        global header
        cols = [s.strip() for s in header.split(';')]
        cols += ['Start Auslesebereich BG', 'Ende Auslesebereich BG',
                 'Start Integration BG', 'Ende Integration BG']
        rois = pd.DataFrame(columns=cols)
        rois.set_index('ROI-Name', inplace=True)
        return
    filename = cfg_path.split(r'\\')[-1]
    cfg_entry.delete(0, tk.END)
    cfg_entry.insert(0, filename)
    rois = pd.read_csv(cfg_path, sep=';')
    rois.columns = rois.columns.str.strip()
    print(rois.columns)
    names = rois['ROI-Name'].values
    BG = []
    Peaks = []
    for name in names:
        if 'BG' in name:
            BG.append(name)
        else:
            Peaks.append(name)
    ROI = {}
    for peak in Peaks:
        ROI['{}'.format(peak)] = str(0)
        peak_str = peak.split('_')
        for bg in BG:
            if 'BG_'+peak_str[0] in bg and peak_str[1].strip() in bg:
                ROI['{}'.format(peak)] = bg
    rois.set_index('ROI-Name', inplace=True)
    rois['Start Auslesebereich BG'] = np.zeros(len(rois))
    rois['Ende Auslesebereich BG'] = np.zeros(len(rois))
    rois['Start Integration BG'] = np.zeros(len(rois))
    rois['Ende Integration BG'] = np.zeros(len(rois))
    for peak in ROI.keys():
        rois.loc[peak , 'Start Auslesebereich BG'] = rois.at[ROI[peak], 'Start Auslesebereich']
        rois.loc[peak , 'Ende Auslesebereich BG'] = rois.at[ROI[peak], 'Ende Auslesebereich']
        rois.loc[peak , 'Start Integration BG'] = rois.at[ROI[peak], 'Start Integration']
        rois.loc[peak , 'Ende Integration BG'] = rois.at[ROI[peak], 'Ende Integration']
    for bg in ROI.values():
        try:
            rois.drop(labels=bg, axis=0, inplace=True)
        except ValueError as e:
            continue
    rois.index = rois.index.str.strip()
    plot_rois(f, rois)
    return


def onclick(event):
    global marked
    global axrois
    if event.button == 3:
        selection = var1.get()
        lines = axrois[marked][0]
        wls = axrois[marked][1]
        if selection == 1:
            if lines[0]:
                lines[0].remove()
            lines[0] = plt.axvline(x=event.xdata, color='r', lw=1, ls='-.', alpha=1)
            f.canvas.draw()
            label11.config(state='normal')
            label11.delete(0, tk.END)
            label11.insert(0, event.xdata)
            label11.config(state='disabled')
            wls[0] = event.xdata
            axrois[marked] = [lines, wls]
            var1.set(2)
        elif selection == 2:
            if lines[1]:
                lines[1].remove()
            lines[1] = plt.axvline(x=event.xdata, color='r', lw=1, ls='-.', alpha=1)
            f.canvas.draw()
            label22.config(state='normal')
            label22.delete(0, tk.END)
            label22.insert(0, event.xdata)
            label22.config(state='disabled')
            wls[1] = event.xdata
            axrois[marked] = [lines, wls]
            var1.set(3)
        elif selection == 3:
            if lines[2]:
                lines[2].remove()
            lines[2] = plt.axvline(x=event.xdata, color='k', lw=1, ls='-.', alpha=1)
            f.canvas.draw()
            label33.config(state='normal')
            label33.delete(0, tk.END)
            label33.insert(0, event.xdata)
            label33.config(state='disabled')
            wls[2] = event.xdata
            axrois[marked] = [lines, wls]
            var1.set(4)
        elif selection == 4:
            if lines[3]:
                lines[3].remove()
            lines[3] = plt.axvline(x=event.xdata, color='k', lw=1, ls='-.', alpha=1)
            f.canvas.draw()
            label44.config(state='normal')
            label44.delete(0, tk.END)
            label44.insert(0, event.xdata)
            label44.config(state='disabled')
            wls[3] = event.xdata
            axrois[marked] = [lines, wls]
            var1.set(1)
    return


def rem(name_roi=False, drop=False):
    global rois
    label11.config(state='normal')
    label11.delete(0, tk.END)
    label11.config(state='disabled')
    label22.config(state='normal')
    label22.delete(0, tk.END)
    label22.config(state='disabled')
    label33.config(state='normal')
    label33.delete(0, tk.END)
    label33.config(state='disabled')
    label44.config(state='normal')
    label44.delete(0, tk.END)
    label44.config(state='disabled')
    f.canvas.draw()
    ROI = ROI_entry.get()
    global rois
    if not name_roi and drop == True:
        rois.drop(ROI, inplace=True)
    elif name_roi:
        rois.drop(name_roi, inplace=True)
    else:
        pass
    plot_rois(f, rois)
    return


def change_alpha(f, marked, string):
    global axrois
    if string == 'up':
        lines = axrois[marked][0]
        wls = axrois[marked][1]
        for l in lines:
            l.remove()
        a = plt.axvline(x=wls[0], color='r', lw=1, ls='-.', alpha=1)
        b = plt.axvline(x=wls[1], color='r', lw=1, ls='-.', alpha=1)
        c = plt.axvline(x=wls[2], color='k', lw=1, ls='-.', alpha=1)
        d = plt.axvline(x=wls[3], color='k', lw=1, ls='-.', alpha=1)
        axrois[marked] = [[a, b, c, d], wls]
    elif string == 'down':
        lines = axrois[marked][0]
        wls = axrois[marked][1]
        for l in lines:
            l.remove()
        a = plt.axvline(x=wls[0], color='r', lw=1, ls='-.', alpha=0.1)
        b = plt.axvline(x=wls[1], color='r', lw=1, ls='-.', alpha=0.1)
        c = plt.axvline(x=wls[2], color='k', lw=1, ls='-.', alpha=0.1)
        d = plt.axvline(x=wls[3], color='k', lw=1, ls='-.', alpha=0.1)
        axrois[marked] = [[a, b, c, d], wls]
    else:
        print('change_alpha Fehler')
    f.canvas.draw()
    return


def edit_roi(f):
    global axrois
    global rois
    global spektrum_df
    global neu
    global marked
    ROI = ROI_entry.get()
    if ROI == '':
        err = 'Benennung für die ROI eingeben'
        messagebox.showerror('Error', err)
        return
    peak_start = float(label11.get())
    peak_end = float(label22.get())
    bg_start = float(label33.get())
    bg_end = float(label44.get())
    if not isinstance(spektrum_df, pd.DataFrame):
        err = """Es muss eine Spektrums-Datei (.spa) eingelesen sein, damit an dieser Stelle den eingelesenen Wellenlängen der Pixel auf der CCD-Zeile zugewiesen werden kann."""
        messagebox.showerror("Error", err)
        return
    peak_start_ix = spektrum_df.index.get_loc(float(peak_start), method='nearest')
    peak_end_ix = spektrum_df.index.get_loc(float(peak_end), method='nearest')
    bg_start_ix = spektrum_df.index.get_loc(float(bg_start), method='nearest')
    bg_end_ix = spektrum_df.index.get_loc(float(bg_end), method='nearest')
    ma = '0'
    mi = '0'
    fill_list = [peak_start_ix, peak_end_ix, peak_start, peak_end, mi, ma,
                 bg_start_ix, bg_end_ix, bg_start, bg_end]
    rois.loc[ROI] = fill_list
    if marked != ROI and marked != '':
        rem(marked, drop=True)
        marked = ''
        print('Hier!')
    else:
        print('Dort!')
        rem()
    print('Hallo!')
    plot_rois(f, rois)
    neu = False
    return


def roi_insert(name):
    global axrois
    if name == 'None':
        ROI_entry.config(state='normal')
        ROI_entry.delete(0, tk.END)
        ROI_entry.config(state='disabled')
        label11.config(state='normal')
        label11.delete(0, tk.END)
        label11.config(state='disabled')
        label22.config(state='normal')
        label22.delete(0, tk.END)
        label22.config(state='disabled')
        label33.config(state='normal')
        label33.delete(0, tk.END)
        label33.config(state='disabled')
        label44.config(state='normal')
        label44.delete(0, tk.END)
        label44.config(state='disabled')
    else:
        wls = axrois[marked][1]
        ROI_entry.config(state='normal')
        ROI_entry.delete(0, tk.END)
        ROI_entry.insert(0, marked)
        ROI_entry.config(state='disabled')
        label11.config(state='normal')
        label11.delete(0, tk.END)
        label11.insert(0, wls[0])
        label11.config(state='disabled')
        label22.config(state='normal')
        label22.delete(0, tk.END)
        label22.insert(0, wls[1])
        label22.config(state='disabled')
        label33.config(state='normal')
        label33.delete(0, tk.END)
        label33.insert(0, wls[2])
        label33.config(state='disabled')
        label44.config(state='normal')
        label44.delete(0, tk.END)
        label44.insert(0, wls[3])
        label44.config(state='disabled')
    return


def highlight(f, string):
    global axrois
    global marked
    roi = list(axrois.keys())
    if string == 'left':
        if marked == '':
            marked = roi[0]
            change_alpha(f, marked, 'up')
            roi_insert(marked)
        else:
            last = marked
            change_alpha(f, last, 'down')
            ix = roi.index(marked)
            marked = roi[ix-1]
            change_alpha(f, marked, 'up')
            roi_insert(marked)
    elif string == 'right':
        if marked == '':
            marked = roi[0]
            change_alpha(f, marked, 'up')
            roi_insert(marked)
        else:
            last = marked
            change_alpha(f, last, 'down')
            ix = roi.index(marked)
            jx = (ix + 1) % (len(roi))
            marked = roi[jx]
            change_alpha(f, marked, 'up')
            roi_insert(marked)
    elif string == 'neu':
        neu = True
        roi_insert('None')
        if marked != '':
            change_alpha(f, marked, 'down')
            marked = ''
        marked = simpledialog.askstring(
                title = "ROI-Name", prompt = "Gebe den Namen der ROI an:")
        axrois[marked] = [['', '', '', ''], ['', '', '', '']]
        ROI_entry.config(state='normal')
        ROI_entry.delete(0, tk.END)
        ROI_entry.insert(0, marked)
        ROI_entry.config(state='disabled')
    else:
        print('highlight Felhler')
    return


def file_save():
    global rois
    global header
    rois['Wl'] = [float(str(r).split('_')[-1]) for r in rois.index]
    rois.sort_values(by='Wl', inplace=True)
    rois.drop('Wl', axis=1, inplace=True)
    rois2 = rois.copy()
    peaks = rois2.index.values
    cols = rois2.columns.values
    for peak in peaks:
        s_wl = rois2.at[peak, cols[2]]
        e_wl =rois2.at[peak, cols[3]]
        if e_wl < s_wl:
            temp = rois2.at[peak, cols[0]]
            rois2.at[peak, cols[0]] = rois2.at[peak, cols[1]]
            rois2.at[peak, cols[1]] = temp
            temp = rois2.at[peak, cols[2]]
            rois2.at[peak, cols[2]] = rois2.at[peak, cols[3]]
            rois2.at[peak, cols[3]] = temp
        bg = 'BG_' + peak
        s_wl = rois2.at[peak, cols[8]]
        e_wl =rois2.at[peak, cols[9]]
        if e_wl > s_wl:
            s_ix = rois2.at[peak, cols[6]]
            e_ix = rois2.at[peak, cols[7]]
        else:
            s_ix = rois2.at[peak, cols[7]]
            e_ix = rois2.at[peak, cols[6]]
            s_wl = rois2.at[peak, cols[9]]
            e_wl =rois2.at[peak, cols[8]]
        mi = rois2.at[peak, cols[4]]
        ma = rois2.at[peak, cols[5]]
        rois2.loc[bg] = [s_ix, e_ix, s_wl, e_wl, mi, ma, 0, 0, 0, 0]
    rois2.drop(cols[6:], inplace=True, axis=1)
    text_file = tk.filedialog.asksaveasfile(mode='w')
    text_file.write(header+'\n')
    rows = rois2.index.values
    for row in rows:
        ROI_name = row
        st_ix = rois2.at[row, cols[0]]
        ed_ix = rois2.at[row, cols[1]]
        st_wl = rois2.at[row, cols[2]]
        ed_wl = rois2.at[row, cols[3]]
        diff1 = 26-len(str(ROI_name))
        diff2 = 21-len(str(st_ix))
        diff3 = 21-len(str(ed_ix))
        diff4 = 21-len(str(st_wl))
        diff5 = 21-len(str(ed_wl))
        ROI_name = str(ROI_name) + ' '*diff1
        st_ix = ' ' + str(st_ix) + ' '*diff2
        ed_ix = ' ' + str(ed_ix) + ' '*diff3
        st_wl = ' ' + str(st_wl) + ' '*diff4
        ed_wl = ' ' + str(ed_wl) + ' '*diff5
        mi = ' 0                    '
        ma = ' 0                  '
        row = ';'.join([ROI_name, st_ix, ed_ix, st_wl, ed_wl, mi, ma])
        text_file.write(row+'\n')
    text_file.close()
    return


form = tk.Tk()
form.title('ROI Picker')
form.geometry('1450x580')

tab_parent = ttk.Notebook(form)
tab1 = ttk.Frame(tab_parent)
tab_parent.add(tab1, text='Spektrum')

f = plt.figure(figsize=(10, 5), dpi=100)
a = f.add_subplot(111)
# a.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])

canvas = FigureCanvasTkAgg(f, tab1)
canvas.show()
canvas.get_tk_widget().grid(row=1, column=1, columnspan=6, rowspan=20)


toolbarFrame = ttk.Frame(tab1)
toolbarFrame.grid(row=22, column=1)
toolbar = NavigationToolbar2TkAgg(canvas, toolbarFrame)

spa_label = tk.Label(tab1, text='Spektrum: ')
spa_label.grid(row=1, column=7)
spa_button = ttk.Button(tab1, text='Datei suchen', command=lambda: spa_b(f, a))
spa_button.grid(row=1, column=9)
spa_entry = tk.Entry(tab1, width=20)
spa_entry.grid(row=1, column=8)
spa_entry.insert(0, 'spa Datei einfügen')

H_spa_label = tk.Label(tab1, text='Hintergrund-Spektrum: ')
H_spa_label.grid(row=2, column=7)
H_spa_button = ttk.Button(tab1, text='Datei hinzufügen', command=lambda: H_spa_b(f, a, add=True))
H_spa_button.grid(row=2, column=9)
H_spa_button2 = ttk.Button(tab1, text='berechnen', command=lambda: H_spa_b(f, a, add=False))
H_spa_button2.grid(row=3, column=9)
H_spa_button3 = ttk.Button(tab1, text='löschen', command=lambda: H_spa_b(f, a, clear=True))
H_spa_button3.grid(row=3, column=10)
H_spa_entry = tk.Entry(tab1, width=20)
H_spa_entry.grid(row=2, column=8)
H_spa_entry.insert(0, 'spa Datei einfügen')

nist_label = tk.Label(tab1, text='NIST ASD Datenbank: ')
nist_label.grid(row=4, column=7)
nist_button = ttk.Button(tab1, text='Datei suchen',
                         command=lambda: nist_b(
                                 f, a, elements.get(), ionstate.get(), rel_int.get()))
nist_button.grid(row=4, column=9)
nist_entry = tk.Entry(tab1, width=20)
nist_entry.grid(row=4, column=8)
nist_entry.insert(0, 'ASD Datei einfügen')

cfg_label = tk.Label(tab1, text='ROI Datei: ')
cfg_label.grid(row=5, column=7)
cfg_button = ttk.Button(tab1, text='Datei suchen', command=lambda: cfg_b(f))
cfg_button.grid(row=5, column=9)
cfg_entry = tk.Entry(tab1, width=20)
cfg_entry.grid(row=5, column=8)
cfg_entry.insert(0, 'cfg Datei einfügen')

elem = tk.Label(tab1, text='Elemente: ')
elem.grid(row=7, column=7)
elements = tk.Entry(tab1, width=20)
elements.grid(row=7, column=8)

ion = tk.Label(tab1, text='Ion: ')
ion.grid(row=8, column=7)
ionstate = tk.Entry(tab1, width=20)
ionstate.grid(row=8, column=8)
ionstate.insert(0, '1, 2')

relint = tk.Label(tab1, text='relative Intensität: ')
relint.grid(row=9, column=7)
rel_int = tk.Entry(tab1, width=20)
rel_int.grid(row=9, column=8)
rel_int.insert(0, 0)
apply_button = ttk.Button(tab1, text='Anwenden',
                          command=lambda: show_nist(f, a, df_nist,
                                                    elements.get(),
                                                    ionstate.get(),
                                                    rel_int.get()))
apply_button.grid(row=9, column=9)


left = ttk.Button(tab1, text='<-',
                          command=lambda: highlight(f, 'left'))
left.grid(row=10, column=7)
right = ttk.Button(tab1, text='->',
                          command=lambda: highlight(f, 'right'))
right.grid(row=10, column=8)
neu = ttk.Button(tab1, text='neu',
                          command=lambda: highlight(f, 'neu'))
neu.grid(row=10, column=9)



ROI_label = tk.Label(tab1, text='ROI Name: ')
ROI_label.grid(row=12, column=7)
ROI_entry = tk.Entry(tab1, width=20)
ROI_entry.grid(row=12, column=8)
ROI_entry.config(state='disabled')


var1 = tk.IntVar()
var1.set(1)
label1 = tk.Radiobutton(tab1, text='Start Peak: ', variable=var1, value=1)
label1.grid(row=13, column=7)
label11 = tk.Entry(tab1, width=20)
label11.grid(row=13, column=8)
label11.config(state='disabled')

label2 = tk.Radiobutton(tab1, text='End Peak: ', variable=var1, value=2)
label2.grid(row=14, column=7)
label22 = tk.Entry(tab1, width=20)
label22.grid(row=14, column=8)
label22.config(state='disabled')

label3 = tk.Radiobutton(tab1, text='Start BG: ', variable=var1, value=3)
label3.grid(row=15, column=7)
label33 = tk.Entry(tab1, width=20)
label33.grid(row=15, column=8)
label33.config(state='disabled')

label4 = tk.Radiobutton(tab1, text='End Bg: ', variable=var1, value=4)
label4.grid(row=16, column=7)
label44 = tk.Entry(tab1, width=20)
label44.grid(row=16, column=8)
label44.config(state='disabled')


clear_button = ttk.Button(tab1, text='ROI löschen', command=lambda: rem())
clear_button.grid(row=17, column=7)

save_button = ttk.Button(tab1, text='ROI speichern', command=lambda: edit_roi(f))
save_button.grid(row=17, column=8)

save_df_button = ttk.Button(tab1, text='cfg Datei speichern',
                            command=lambda: file_save())
save_df_button.grid(row=17, column=9)

f.canvas.mpl_connect('button_press_event', onclick)


tab_parent.pack(expand=1, fill='both')
form.mainloop()

