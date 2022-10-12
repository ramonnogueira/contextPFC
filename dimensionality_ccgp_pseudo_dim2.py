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

# pre targets: context, prev choice, prev rew
# post dots: choice, context, diff
# pre-saccade: choice, context, diff?
# post-saccade: choice, context, reward

# target onset: 'targ_on', dots onset: 'dots_on', dots offset: 'dots_off', saccade: 'response_edf'
talig_vec=np.array(['targ_on','dots_on','response_edf'])
dic_time={} # same number of steps for all time locks
dic_time['targ_on']=np.array([1000,1000,500,200]) # time pre, time post, bin size, step size
dic_time['dots_on']=np.array([1000,1000,500,200])
dic_time['response_edf']=np.array([1000,1000,500,200])
steps=int((dic_time['dots_on'][0]+dic_time['dots_on'][1])/dic_time['dots_on'][3])# Careful here!
xx_dic={}
for i in range(len(talig_vec)):
    xx_dic[talig_vec[i]]=np.linspace(-dic_time[talig_vec[i]][0]/1000,dic_time[talig_vec[i]][1]/1000,steps) 

nt=200
n_rand=10
perc_tr=0.8
thres=0
reg=1

quant_vec=[['choice','context'],['choice','difficulty'],['context','difficulty']]
col=[['blue','brown','black'],['blue','purple','black'],['brown','purple','black']]
monkeys=['Niels','Galileo']
phase='early'

for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/%s/%s/'%(phase,monkeys[k]) 
    files=os.listdir(abs_path)
    dim=nan*np.zeros((len(talig_vec),len(quant_vec),steps,n_rand,3))
    abstr=nan*np.zeros((len(talig_vec),len(quant_vec),steps,n_rand,2,2))
    for kkk in range(len(talig_vec)):
        print ('  ',talig_vec[kkk])
        for kk in range(len(quant_vec)):
            print ('    ',quant_vec[kk])
            pseudo=miscellaneous.pseudopopulation_2(abs_path,files,quant_vec[kk],talig_vec[kkk],dic_time[talig_vec[kkk]],steps,thres,nt,n_rand,perc_tr)
            pseudo_all=pseudo['pseudo_all']
            pseudo_tr=pseudo['pseudo_tr']
            pseudo_te=pseudo['pseudo_te']
            clase_all=pseudo['clase_all']
            for i in range(steps):
                for ii in range(n_rand):
                    dim[kkk,kk,i,ii]=miscellaneous.dim_pseudo_2(pseudo_tr[i,ii],pseudo_te[i,ii],clase_all,reg)
                    abstr[kkk,kk,i,ii]=miscellaneous.abstraction_2D(pseudo_all[i,ii],clase_all,reg)[:,:,1]
                    
    dim_m=np.mean(dim,axis=3)
    dim_std=np.nanstd(dim,axis=3)
    abs_m=np.mean(abstr,axis=(3,5))
    abs_std=np.std(abstr,axis=(3,5))

    #####################
    # Plots
    tl_vec=['targets on','dots on','saccade']
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

            for tt in range(3):
                ax.plot(xx_dic[talig_vec[kkk]],dim_m[kkk,kk,:,tt],color=col[kk][tt])
                ax.fill_between(xx_dic[talig_vec[kkk]],dim_m[kkk,kk,:,tt]-dim_std[kkk,kk,:,tt],dim_m[kkk,kk,:,tt]+dim_std[kkk,kk,:,tt],color=col[kk][tt],alpha=0.6)
            for tt in range(2):
               ax.plot(xx_dic[talig_vec[kkk]],abs_m[kkk,kk,:,tt],color=col[kk][tt],alpha=0.7)
               ax.fill_between(xx_dic[talig_vec[kkk]],abs_m[kkk,kk,:,tt]-abs_std[kkk,kk,:,tt],abs_m[kkk,kk,:,tt]+abs_std[kkk,kk,:,tt],color=col[kk][tt],alpha=0.3) 
            ax.axvline(0,color='black',linestyle='--')
            ax.set_ylim([0.4,1.0])
            ax.plot(xx_dic[talig_vec[kkk]],0.5*np.ones(steps),color='black',linestyle='--')
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/dim_abs_2D_%s_bin_%i_thres_%.1f_pseudo_phase_%s.pdf'%(monkeys[k],dic_time['dots_on'][2],thres,phase),dpi=500,bbox_inches='tight')

   
