import os
import matplotlib.pylab as plt
import numpy as np
import scipy.io
import math
import sys
import tables
import pandas
import pickle as pkl
from scipy.stats import sem
from scipy.stats import pearsonr
from numpy.random import permutation
import miscellaneous
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy.stats import ortho_group 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from scipy import stats
from scipy.optimize import curve_fit

nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines: 
            if loc=='left':
                spine.set_position(('outward', 10))  # outward by 10 points
            if loc=='bottom':
                spine.set_position(('outward', 0))  # outward by 10 points
         #   spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def getField(data,name,extra=False):
    ind=np.where(data['FIRA'][0]['events'][0]==name)[0][0]
    var_pre=data['FIRA'][1][:,ind]
    if extra:
        var=nan*np.zeros(np.shape(var_pre))
        for i in range(len(var)):
            if type(var_pre[i])==int:
                var[i]=var_pre[i]
            else:
                var[i]=nan
    else:
        var=var_pre.copy()    
    return var

def trans_rew(x):
    rew=nan*np.zeros((len(x),2))
    for i in range(len(x)):
        rew[i]=x[i]
    return rew

def order_files(x):
    ord_pre=[]
    for i in range(len(x)):
        ord_pre.append(x[i][1:9])
    ord_pre=np.array(ord_pre)
    order=np.argsort(ord_pre)
    return order

# Right: 1, Left: 1 (or 0).
# Coherence Positive is Right, negative is Left
# Context 1: Right is more rewarded
# Context 2: Left is more rewarded
right=1

#################################################

borders=np.array([[[0,4],[4,8],[8,12]],[[0,10],[10,20],[20,30]]])
t_back=20
t_forw=40
xx=np.arange(t_back+t_forw)-t_back
monkeys=['Niels']

talig='dots_on'
dic_time=np.array([0,200,200,200])# time pre, time post, bin size, step size
reg=1e-3

for k in range(len(monkeys)):
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/unsorted/%s/'%(monkeys[k]) 
    files_pre=np.array(os.listdir(abs_path))
    order=order_files(files_pre)
    files=files_pre[order]
    neu_ctx_ch=nan*np.zeros((len(files),t_back+t_forw))
    neu_interx=nan*np.zeros(len(files))
    for kk in range(len(files)):
        print (files[kk])
        #Load data
        data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
        beha=miscellaneous.behavior(data)
        index_nonan=beha['index_nonan']
        # We discard first trial of session because we are interested in context changes
        stimulus=beha['stimulus'][1:]
        choice=beha['choice'][1:]
        coherence=beha['coherence_signed'][1:]
        coh_uq=np.unique(coherence)
        reward=beha['reward'][1:]
        rt=beha['reaction_time'][1:]
        context_pre=beha['context']
        ctx_ch=(context_pre[1:]-context_pre[0:-1])
        context=context_pre[1:]
        ind_ch=np.where(abs(ctx_ch)==1)[0]
        indch_ct10=np.where(ctx_ch==-1)[0]
        indch_ct01=np.where(ctx_ch==1)[0]
        # Neural data
        if (files[kk][0:10]=='G20190617b') or (files[kk][0:10]=='G20190618b') or (files[kk][0:10]=='G20190619b') :
            firing_rate_pre=miscellaneous.getRasters(data,talig,dic_time,index_nonan,threshold=0)
        else:
            firing_rate_pre=miscellaneous.getRasters_unsorted(data,talig,dic_time,index_nonan,threshold=0)
        firing_rate=miscellaneous.normalize_fr(firing_rate_pre)[1:,:,0]
        
        # Extract indices for training classifier (remove the one for testing from the entire dataset)
        ind_train=np.arange(len(coherence))
        for h in range(len(ind_ch)):
            ind_t=np.arange(t_back+t_forw)-t_back+ind_ch[h]
            ind_del=[]
            for hh in range(len(ind_t)):
                try:
                    ind_del.append(np.where(ind_train==ind_t[hh])[0][0])
                except:
                    print ('error aqui')
            ind_del=np.array(ind_del)
            ind_train=np.delete(ind_train,ind_del)

        # Fit classifier
        cl=LogisticRegression(C=1/reg,class_weight='balanced')
        cl.fit(firing_rate[ind_train],context[ind_train])

        # Neuronal Probability of Choice
        for j in range(t_back):
            try:
                neu_ctx_ch[kk,j]=cl.score(firing_rate[ind_ch-t_back+j],context[ind_ch-t_back+j])
            except:
                print ('Neuro Error Back ',j)
        for jj in range(t_forw):
            try:
                neu_ctx_ch[kk,t_back+jj]=cl.score(firing_rate[ind_ch+jj],context[ind_ch+jj])
            except:
                print ('Neuro Error Forward ',jj)

    #####################################
    # Probability of Choice
    # All Sessions
    neu_ctx_m=np.nanmean(neu_ctx_ch,axis=0)
    print (neu_ctx_m)
    neu_ctx_sem=sem(neu_ctx_ch,axis=0,nan_policy='omit')
    plt.plot(xx,neu_ctx_m,color='green')
    plt.axvline(0,color='black',linestyle='--')
    plt.plot(xx,0.5*np.ones(len(xx)),color='black',linestyle='--')
    plt.fill_between(xx,neu_ctx_m-neu_ctx_sem,neu_ctx_m+neu_ctx_sem,color='green',alpha=0.5)
    plt.ylabel('Decoding Performance Context')
    plt.ylim([0,1])
    plt.title('All Sessions')
    plt.xlabel('Trials from Context Switch')
    plt.show()

    # Across Learning
    learn_label=['Early','Middle','Late']
    for l in range(3):
        neu_ctx_m=np.nanmean(neu_ctx_ch[borders[k,l,0]:borders[k,l,1]],axis=0)
        neu_ctx_sem=sem(neu_ctx_ch[borders[k,l,0]:borders[k,l,1]],axis=0,nan_policy='omit')
        plt.plot(xx,neu_ctx_m,color='green')
        plt.axvline(0,color='black',linestyle='--')
        plt.plot(xx,0.5*np.ones(len(xx)),color='black',linestyle='--')
        plt.fill_between(xx,neu_ctx_m-neu_ctx_sem,neu_ctx_m+neu_ctx_sem,color='green',alpha=0.5)
        plt.ylabel('Decoding Performance Context')
        plt.ylim([0,1])
        plt.title('Learning Phase %s'%learn_label[l])
        plt.xlabel('Trials from Context Switch')
        plt.show()

