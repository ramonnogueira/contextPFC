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
import miscellaneous
import datetime
from scipy.stats import ortho_group
from scipy.stats import special_ortho_group

nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def clase_resolution(group_coh,coherence):
    clase_coh=nan*np.zeros(len(coherence))
    coh_uq=np.unique(coherence)
    for i in range(len(coh_uq)):
        ind=np.where(coherence==coh_uq[i])[0]
        clase_coh[ind]=group_coh[i]
    return clase_coh
    
##############################################

monkey='Niels'

talig='dots_on'
dic_time=np.array([0,450,200,50])# time pre, time post, bin size, step size (time pre always positive) #For Galileo use timepost 800 or 1000. For Niels use 
steps=int((dic_time[0]+dic_time[1])/dic_time[3])
xx=np.linspace(-dic_time[0]/1000,dic_time[1]/1000,steps,endpoint=False)
print (xx)

nt=100 #100 for coh signed, 200 for coh unsigned, 50 for coh signed with context
n_rand=20
n_shuff=100
perc_tr=0.8
thres=0
reg=1e2
n_coh=15

tpre_sacc=50

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])
if monkey=='Niels':
    group_coh=np.array([nan,0  ,0  ,0  ,0  ,0  ,0  ,nan,1  ,1  ,1  ,1  ,1  ,1  ,nan])
if monkey=='Galileo':
    group_coh=np.array([0  ,0  ,0  ,0  ,0  ,0  ,0  ,nan,1  ,1  ,1  ,1  ,1  ,1  ,1  ])

abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/late/%s/'%(monkey)
files=miscellaneous.order_files(np.array(os.listdir(abs_path)))
print (files)

###################
# Original
perf_all=nan*np.zeros((steps,n_rand,3))
# Careful! in this function I am only using correct trials so that choice and stimulus are the same    
pseudo=miscellaneous.pseudopop_coherence_context_correct(abs_path,files,talig,dic_time,steps,thres,nt,n_rand,perc_tr,True,tpre_sacc,group_ref,shuff=False,learning=False)
    
for kk in range(steps):
    print (kk)
    pseudo_tr=pseudo['pseudo_tr'][kk]
    pseudo_te=pseudo['pseudo_te'][kk]
    context=pseudo['clase_ctx']
    clase_all=pseudo['clase_all']
    coherence=pseudo['clase_coh']

    clase_coh=clase_resolution(group_coh,coherence)
    indnan=~np.isnan(clase_coh) # Indices for the coherences we are going to use (True means to use, False to discard)
        
    feat_binary=nan*np.zeros((len(coherence),2))
    ind00=np.where((clase_coh==0)&(context==0))[0]
    ind01=np.where((clase_coh==0)&(context==1))[0]
    ind10=np.where((clase_coh==1)&(context==0))[0]
    ind11=np.where((clase_coh==1)&(context==1))[0]
    feat_binary[ind00]=np.array([0,0])
    feat_binary[ind01]=np.array([0,1])
    feat_binary[ind10]=np.array([1,0])
    feat_binary[ind11]=np.array([1,1])
        
    for ii in range(n_rand):
        #print (' ',ii)
        sum_nan=np.sum(np.isnan(pseudo_tr[ii]),axis=1)
        indnan_flat=(sum_nan==0) # True will be used, False discarded
        ind_nonan=(indnan*indnan_flat) # Index used combination of discarded from RT and discarded from group_coh
        #print ('Diff ',j,np.sum(ind_nonan))
        # Choice
        cl=LogisticRegression(C=1/reg,class_weight='balanced')
        #cl=LinearSVC(C=1/reg,class_weight='balanced')
        cl.fit(pseudo_tr[ii][ind_nonan],feat_binary[:,0][ind_nonan])
        perf_all[kk,ii,0]=cl.score(pseudo_te[ii][ind_nonan],feat_binary[:,0][ind_nonan])
        # Context
        cl=LogisticRegression(C=1/reg,class_weight='balanced')
        #cl=LinearSVC(C=1/reg,class_weight='balanced')
        cl.fit(pseudo_tr[ii][ind_nonan],feat_binary[:,1][ind_nonan])
        perf_all[kk,ii,1]=cl.score(pseudo_te[ii][ind_nonan],feat_binary[:,1][ind_nonan])
        # XOR
        cl=LogisticRegression(C=1/reg,class_weight='balanced')
        #cl=LinearSVC(C=1/reg,class_weight='balanced')
        xor=np.sum(feat_binary,axis=1)%2
        cl.fit(pseudo_tr[ii][ind_nonan],xor[ind_nonan])
        perf_all[kk,ii,2]=cl.score(pseudo_te[ii][ind_nonan],xor[ind_nonan])
        

perf_all_m=np.nanmean(perf_all,axis=1)
perf_all_std=np.std(perf_all,axis=1)
print (perf_all_m)

###########################
# Shuffle
print ('SHUFFLE...')
perf_all_sh=nan*np.zeros((n_shuff,steps,n_rand,3))

for g in range(n_shuff):
    print (g)
    pseudo=miscellaneous.pseudopop_coherence_context_correct(abs_path,files,talig,dic_time,steps,thres,nt,n_rand,perc_tr,True,tpre_sacc,group_ref,shuff=True,learning=False)
    
    for kk in range(steps):
        # Careful! in this function I am only using correct trials so that choice and stimulus are the same    
        pseudo_tr=pseudo['pseudo_tr'][kk]
        pseudo_te=pseudo['pseudo_te'][kk]
        context=pseudo['clase_ctx']
        clase_all=pseudo['clase_all']
        coherence=pseudo['clase_coh']

        clase_coh=clase_resolution(group_coh,coherence)
        indnan=~np.isnan(clase_coh) # Indices for the coherences we are going to use (True means to use, False to discard)
        
        feat_binary=nan*np.zeros((len(coherence),2))
        ind00=np.where((clase_coh==0)&(context==0))[0]
        ind01=np.where((clase_coh==0)&(context==1))[0]
        ind10=np.where((clase_coh==1)&(context==0))[0]
        ind11=np.where((clase_coh==1)&(context==1))[0]
        feat_binary[ind00]=np.array([0,0])
        feat_binary[ind01]=np.array([0,1])
        feat_binary[ind10]=np.array([1,0])
        feat_binary[ind11]=np.array([1,1])
        
        for ii in range(n_rand):
            #print (' ',ii)
            sum_nan=np.sum(np.isnan(pseudo_tr[ii]),axis=1)
            indnan_flat=(sum_nan==0) # True will be used, False discarded
            ind_nonan=(indnan*indnan_flat) # Index used combination of discarded from RT and discarded from group_coh
            #print ('Diff ',j,np.sum(ind_nonan))
            # Choice
            cl=LogisticRegression(C=1/reg,class_weight='balanced')
            #cl=LinearSVC(C=1/reg,class_weight='balanced')
            cl.fit(pseudo_tr[ii][ind_nonan],feat_binary[:,0][ind_nonan])
            perf_all_sh[g,kk,ii,0]=cl.score(pseudo_te[ii][ind_nonan],feat_binary[:,0][ind_nonan])
            # Context
            cl=LogisticRegression(C=1/reg,class_weight='balanced')
            #cl=LinearSVC(C=1/reg,class_weight='balanced')
            cl.fit(pseudo_tr[ii][ind_nonan],feat_binary[:,1][ind_nonan])
            perf_all_sh[g,kk,ii,1]=cl.score(pseudo_te[ii][ind_nonan],feat_binary[:,1][ind_nonan])
            # XOR
            cl=LogisticRegression(C=1/reg,class_weight='balanced')
            #cl=LinearSVC(C=1/reg,class_weight='balanced')
            xor=np.sum(feat_binary,axis=1)%2
            cl.fit(pseudo_tr[ii][ind_nonan],xor[ind_nonan])
            perf_all_sh[g,kk,ii,2]=cl.score(pseudo_te[ii][ind_nonan],xor[ind_nonan])
        
perf_all_sh_pre_m=np.mean(perf_all_sh,axis=2)
perf_all_sh_m=np.mean(perf_all_sh_pre_m,axis=0)
perf_all_sh_std=np.std(perf_all_sh_pre_m,axis=0)
print (perf_all_sh_m)

#######################################################################
# Plot performance Tasks and XOR vs time
fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
# Original
ax.plot(xx,perf_all_m[:,0],color='blue',label='Direction')
ax.fill_between(xx,perf_all_m[:,0]-perf_all_std[:,0],perf_all_m[:,0]+perf_all_std[:,0],color='blue',alpha=0.5)
ax.plot(xx,perf_all_m[:,1],color='brown',label='Context')
ax.fill_between(xx,perf_all_m[:,1]-perf_all_std[:,1],perf_all_m[:,1]+perf_all_std[:,1],color='brown',alpha=0.5)
ax.plot(xx,perf_all_m[:,2],color='black',label='XOR')
ax.fill_between(xx,perf_all_m[:,2]-perf_all_std[:,2],perf_all_m[:,2]+perf_all_std[:,2],color='black',alpha=0.5)
# Shuffled
ax.fill_between(xx,perf_all_sh_m[:,0]-perf_all_sh_std[:,0],perf_all_sh_m[:,0]+perf_all_sh_std[:,0],color='blue',alpha=0.3)
ax.fill_between(xx,perf_all_sh_m[:,1]-perf_all_sh_std[:,1],perf_all_sh_m[:,1]+perf_all_sh_std[:,1],color='brown',alpha=0.3)
ax.fill_between(xx,perf_all_sh_m[:,2]-perf_all_sh_std[:,2],perf_all_sh_m[:,2]+perf_all_sh_std[:,2],color='black',alpha=0.3)
#
ax.plot(xx,0.5*np.ones(len(xx)),color='black',linestyle='--')
ax.set_ylim([0.4,1])
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Decoding Performance')
plt.legend(loc='best')
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/choice_ctx_xor_time_pseudo_tl_%s_%s.pdf'%(talig,monkey),dpi=500,bbox_inches='tight')

