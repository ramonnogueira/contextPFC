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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy.stats import ortho_group 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
#from numba import jit
import miscellaneous

nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def calc_dp(feat,clase,wei):
    ind1=np.where(clase==1)[0]
    ind0=np.where(clase==0)[0]
    mean1=np.mean(feat[ind1],axis=0)
    mean0=np.mean(feat[ind0],axis=0)
    deltaf=(mean1-mean0)
    cov=0.5*(np.cov(feat[ind1],rowvar=False)+np.cov(feat[ind0],rowvar=False))
    num=np.dot(deltaf,wei)
    denom=np.sqrt(np.dot(wei,np.dot(cov,wei)))
    return 0.5*num/denom
#####################################

# target onset: 'targ_on', dots onset: 'dots_on', dots offset: 'dots_off', saccade: 'response_edf'
talig_vec=np.array(['targ_on','dots_on','response_edf'])#
dic_time={} # same number of steps for all time locks
dic_time['targ_on']=np.array([1000,1000,200,200]) # time pre, time post, bin size, step size
dic_time['dots_on']=np.array([1000,1000,200,200])
dic_time['response_edf']=np.array([1000,1000,200,200])
steps=int((dic_time['dots_on'][0]+dic_time['dots_on'][1])/dic_time['dots_on'][3])# Careful here!
xx_dic={}
for i in range(len(talig_vec)):
    xx_dic[talig_vec[i]]=np.linspace(-dic_time[talig_vec[i]][0]/1000,dic_time[talig_vec[i]][1]/1000,steps) 

nt=500
n_rand=10
perc_tr=0.8
thres=0
reg=1

quant_vec=['stimulus','choice','context','difficulty','reward']
tback_vec=[0,1,2]
col=['green','blue','brown','purple','red']
monkeys=['Galileo']#'Niels',
phase='late'

for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/%s/%s/'%(phase,monkeys[k]) 
    files=os.listdir(abs_path)
    perf=nan*np.zeros((len(talig_vec),len(quant_vec),len(tback_vec),steps,n_rand,2))
    dprime=nan*np.zeros((len(talig_vec),len(quant_vec),len(tback_vec),steps,n_rand,2)) 

    for kkk in range(len(talig_vec)):
        print ('  ',talig_vec[kkk])
        for kk in range(len(quant_vec)):
            print ('    ',quant_vec[kk])
            for tt in tback_vec:
                print ('      back ',tt)
                pseudo_tr,pseudo_te,clase=miscellaneous.pseudopopulation_1(abs_path,files,quant_vec[kk],talig_vec[kkk],dic_time[talig_vec[kkk]],steps,thres,nt,n_rand,perc_tr,tt)
                for i in range(steps):
                    for ii in range(n_rand):
                        cl=LogisticRegression(C=1/reg,class_weight='balanced')
                        cl.fit(pseudo_tr[i,ii],clase)
                        wei=cl.coef_[0]
                        perf[kkk,kk,tt,i,ii,0]=cl.score(pseudo_tr[i,ii],clase)
                        perf[kkk,kk,tt,i,ii,1]=cl.score(pseudo_te[i,ii],clase)
                        dprime[kkk,kk,tt,i,ii,0]=calc_dp(pseudo_tr[i,ii],clase,wei)
                        dprime[kkk,kk,tt,i,ii,1]=calc_dp(pseudo_te[i,ii],clase,wei)
    
    perf_m=np.nanmean(perf,axis=4)
    perf_std=np.nanstd(perf,axis=4)
    dprime_m=np.nanmean(dprime,axis=4)
    dprime_std=sem(dprime,axis=4)

    #####################
    # Plots
    tl_vec=['targets on','dots on','saccade']
    alpha_vec=[1,0.7,0.4]
    fig=plt.figure(figsize=(len(tl_vec)*4,len(quant_vec)*3))
    for kk in range(len(quant_vec)):
        for kkk in range(len(talig_vec)):
            ax=fig.add_subplot(len(quant_vec),len(tl_vec),kk*3+kkk+1)
            miscellaneous.adjust_spines(ax,['left','bottom'])
            if kk==0:
                ax.set_title('Time lock %s'%tl_vec[kkk])
            if kkk==0:
                ax.set_ylabel('Decoding Performance \n %s'%quant_vec[kk])
            if kk==(len(quant_vec)-1):
                ax.set_xlabel('Time from %s (sec)'%tl_vec[kkk])

            for tt in tback_vec:
                ax.plot(xx_dic[talig_vec[kkk]],perf_m[kkk,kk,tt,:,1],color=col[kk],alpha=alpha_vec[tt])
                ax.fill_between(xx_dic[talig_vec[kkk]],perf_m[kkk,kk,tt,:,1]-perf_std[kkk,kk,tt,:,1],perf_m[kkk,kk,tt,:,1]+perf_std[kkk,kk,tt,:,1],color=col[kk],alpha=0.7*alpha_vec[tt])
            ax.axvline(0,color='black',linestyle='--')
            ax.set_ylim([0.4,1.0])
            ax.plot(xx_dic[talig_vec[kkk]],0.5*np.ones(steps),color='black',linestyle='--')
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/decoding_%s_bin_%i_thres_%.1f_pseudo_phase_%s.pdf'%(monkeys[k],dic_time['dots_on'][2],thres,phase),dpi=500,bbox_inches='tight')

   
