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
from scipy.optimize import curve_fit
#from numba import jit
import miscellaneous

nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

###################################################3

monkey='Niels'#['Galileo']#'Niels']#,]

talig='dots_on' #'targ_on','dots_on'
dic_time=np.array([100,600,200,50]) # time pre, time post, bin size, step size
steps=int((dic_time[0]+dic_time[1])/dic_time[3])
xx=np.linspace(-dic_time[0]/1000,dic_time[1]/1000,steps,endpoint=False)

nt=100 #100 for coh signed, 200 for coh unsigned, 50 for coh signed with context
n_rand=10
n_shuff=0
perc_tr=0.8
thres=0
reg=1e1
n_coh=15

# N Niels: 1018
# N Galileo: 807

tpre_sacc=50

group_coh=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7])

col2=['green','green','blue','blue']
alph2=[1,0.3,0.3,1]

abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/late/%s/'%(monkey)
files=os.listdir(abs_path)

# Careful! in this function I am only using correct trials so that choice and stimulus are the same
pseudo=miscellaneous.pseudopop_coherence_context_correct(abs_path,files,talig,dic_time,steps,thres,nt,n_rand,perc_tr,True,tpre_sacc,group_coh,shuff=False,learning=False) #This!
pseudo_all=pseudo['pseudo_all']
pseudo_tr_pre=pseudo['pseudo_tr']
pseudo_te_pre=pseudo['pseudo_te']
neu_total=pseudo_tr_pre.shape[-1]
clase_all_pre=pseudo['clase_all']
clase_coh_pre=pseudo['clase_coh']
indc0=np.where(clase_coh_pre==7)[0]
clase_ctx_pre=pseudo['clase_ctx']
stim_pre=nan*np.zeros(len(clase_coh_pre))
stim_pre[clase_coh_pre>7]=1
stim_pre[clase_coh_pre<7]=0
stim_pre[indc0]=nan

num_neu=len(pseudo_tr_pre[0,0,0])
condi=np.unique(clase_all_pre)
    
# PCA train
# mean_coh=nan*np.zeros((steps*condi,num_neu))
# for j in range(n_pca):
#     for jj in range(len(condi)):
#         mean_coh[jj*steps:(jj+1)*steps,j]=np.mean(pseudo_tr_pre[:,0,jj*nt:(jj+1)*nt][:,:,j],axis=0)

# embedding=PCA(n_components=3)
# pseudo_mds=embedding.fit(mean_coh)

# wei_trans=embedding.transform(np.array([weights[neu_rnd]]))[0]
# xx, yy = np.meshgrid(np.arange(20)-10,np.arange(20)-10)
# z = (-wei_trans[0]*xx-wei_trans[1]*yy-bias)/wei_trans[2]

# mean_coh=nan*np.zeros((steps*4,num_neu))
# for j in range(num_neu):
#     for jj in range(2):
#         mean_coh[jj*steps:(jj+1)*steps,j]=np.mean(pseudo_tr_pre[:,0,:,j][:,(stim_pre==jj)&(clase_ctx_pre==0)],axis=1)
#         mean_coh[(jj+2)*steps:(jj+3)*steps,j]=np.mean(pseudo_tr_pre[:,0,:,j][:,(stim_pre==jj)&(clase_ctx_pre==1)],axis=1)

# embedding=PCA(n_components=3)
# pseudo_mds=embedding.fit(mean_coh)

# # PCA Test
for j in range(steps):
    print (j)

    mean_coh_ctx=nan*np.zeros((4,num_neu))
    for jj in range(2):
        #mean_coh_ctx[jj]=np.nanmean(pseudo_tr_pre[j,:,(stim_pre==jj)&(clase_ctx_pre==0)],axis=(0,1))
        #mean_coh_ctx[jj+2]=np.nanmean(pseudo_tr_pre[j,:,(stim_pre==jj)&(clase_ctx_pre==1)],axis=(0,1))
        mean_coh_ctx[jj]=np.nanmean(pseudo_all[j,:,(stim_pre==jj)&(clase_ctx_pre==0)],axis=(0,1))
        mean_coh_ctx[jj+2]=np.nanmean(pseudo_all[j,:,(stim_pre==jj)&(clase_ctx_pre==1)],axis=(0,1))
        
    embedding=PCA(n_components=3)
    pseudo_mds=embedding.fit(mean_coh_ctx)
    pseudo_mds_ctx=embedding.transform(mean_coh_ctx)
    
    # 3D
    #if j==19 or j==0:
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()#figsize=(2,2)
    ax = fig.add_subplot(111, projection='3d')
    for jj in range(len(mean_coh_ctx)):
        ax.scatter(pseudo_mds_ctx[jj,0],pseudo_mds_ctx[jj,1],pseudo_mds_ctx[jj,2],color=col2[jj],alpha=alph2[jj],s=100)
        #ax.plot_surface(xx, yy, z, color='black',alpha=0.2)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_xlim([-5,5])
    ax.set_ylim([-5,5])
    ax.set_zlim([-5,5])
    plt.show()
    plt.close(fig)


