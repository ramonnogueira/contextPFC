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
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
np.set_printoptions(suppress=True)

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

monkeys=['Niels']

# target onset: 'targ_on', dots onset: 'dots_on', dots offset: 'dots_off', saccade: 'response_edf'
#talig='dots_on'#'targ_on','dots_on'
#dic_time=np.array([0,800,200,200])# time pre, time post, bin size, step size
talig='dots_on'#,'dots_on'
dic_time=np.array([-200,600,400,400])# time pre, time post, bin size, step size
steps=int((dic_time[0]+dic_time[1])/dic_time[3])
xx=np.linspace(-dic_time[0]/1000,dic_time[1]/1000,steps,endpoint=False)

nt=200 #100 for coh signed, 200 for coh unsigned, 50 for coh signed with context
n_rand=50
perc_tr=0.8
thres=0
reg=1

metric='mean'

tpre_sacc=50

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])
#group_coh=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ]) #15
#group_coh=np.array([-5 ,-4 ,-3 ,-2 ,-2 ,-1 ,-1 ,0  ,1  ,1  ,2  ,2  ,3  ,4  ,5  ]) #11
group_coh=np.array([nan,-4 ,-3 ,-2 ,-2 ,-1 ,-1 ,0  ,1  ,1  ,2  ,2  ,3  ,4  ,nan]) #9 good one
#group_coh=np.array([nan,-4 ,-4 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,4  ,4  ,nan]) #9 # not great
#group_coh=np.array([-3 ,-3 ,-3 ,-2 ,-2 ,-1 ,-1 ,0  ,1  ,1  ,2  ,2  ,3  ,3  ,3  ]) #7
#group_coh=np.array([nan,-3 ,-3 ,-2 ,-2 ,-1 ,-1 ,0  ,1  ,1  ,2  ,2  ,3  ,3  ,nan]) #7
#group_coh=np.array([-2 ,-2 ,-2 ,-2 ,-1 ,-1 ,-1 ,0  ,1  ,1  ,1  ,2  ,2  ,2  ,2  ]) #5
#group_coh=np.array([nan,-2 ,-2 ,-2 ,-1 ,-1 ,-1 ,0  ,1  ,1  ,1  ,2  ,2  ,2  ,nan]) #5

if len(np.unique(group_coh[~np.isnan(group_coh)]))==15:
    col=['darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','black','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','purple','purple','purple','purple','purple','purple','purple','black','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue']
    alph=[0.7,0.6,0.5,0.4,0.3,0.2,0.1,1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.7,0.6,0.5,0.4,0.3,0.2,0.1,1,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
if len(np.unique(group_coh[~np.isnan(group_coh)]))==11:
    col=['darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','black','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','purple','purple','purple','purple','purple','black','darkblue','darkblue','darkblue','darkblue','darkblue']
    alph=[0.7,0.6,0.5,0.4,0.2,1,0.2,0.4,0.5,0.6,0.7,0.7,0.6,0.5,0.4,0.2,1,0.2,0.4,0.5,0.6,0.7]
if len(np.unique(group_coh[~np.isnan(group_coh)]))==9:
    col=['darkgreen','darkgreen','darkgreen','darkgreen','black','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','purple','purple','purple','purple','black','darkblue','darkblue','darkblue','darkblue']
    alph=[0.6,0.5,0.4,0.2,1,0.2,0.4,0.5,0.6,0.6,0.5,0.4,0.2,1,0.2,0.4,0.5,0.6]
if len(np.unique(group_coh[~np.isnan(group_coh)]))==7:
    col=['darkgreen','darkgreen','darkgreen','black','darkgoldenrod','darkgoldenrod','darkgoldenrod','purple','purple','purple','black','darkblue','darkblue','darkblue']
    alph=[0.6,0.3,0.1,1,0.1,0.3,0.6,0.6,0.3,0.1,1,0.1,0.3,0.6]
if len(np.unique(group_coh[~np.isnan(group_coh)]))==5:
    col=['darkgreen','darkgreen','black','darkgoldenrod','darkgoldenrod','purple','purple','black','darkblue','darkblue']
    alph=[0.7,0.4,1,0.4,0.7,0.7,0.4,1,0.4,0.7]

for k in range(len(monkeys)):
    print (monkeys[k])
    if monkeys[k]=='Niels':
        indf=8 # Careful!
    if monkeys[k]=='Galileo':
        indf=26
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/unsorted/%s/'%(monkeys[k]) 
    files_pre=np.array(os.listdir(abs_path))
    order=miscellaneous.order_files(files_pre)
    files=np.array(files_pre[order])[indf:]
    print (files_pre[order])

    pseudo=miscellaneous.pseudopop_coherence_context_correct(abs_path,files,talig,dic_time,steps,thres,nt,n_rand,perc_tr,tpre_sacc,group_coh,shuff=False,learning=True) 
    pseudo_all=pseudo['pseudo_all']
    pseudo_tr=pseudo['pseudo_tr']
    pseudo_te=pseudo['pseudo_te']
    clase_all=pseudo['clase_all']
    clase_coh=pseudo['clase_coh']
    num_neu=len(pseudo_all[0,0,0])
    n_coh=len(np.unique(clase_coh))
   
    for i in range(steps):
        print (xx[i])             
        mean_coh_pre=nan*np.zeros((n_rand,2*n_coh,num_neu))
        for ii in range(n_rand):
            for j in range(2*n_coh):
                ind_coh=np.where(clase_all==j)[0]
                if metric=='mean':
                    mean_coh_pre[ii,j]=np.nanmean(pseudo_all[i,ii][ind_coh],axis=0)
                if metric=='median':
                    mean_coh_pre[ii,j]=np.nanmedian(pseudo_all[i,ii][ind_coh],axis=0)
        if metric=='mean':
            mean_coh=np.nanmedian(mean_coh_pre,axis=0)
        if metric=='median':
            mean_coh=np.nanmean(mean_coh_pre,axis=0)
     
        ind_nan=np.where(np.isnan(np.sum(mean_coh,axis=1)))[0]
        ind_use=np.delete(np.arange(len(mean_coh)),ind_nan,axis=0)
        print (ind_use)
            
        embedding=PCA(n_components=3)
        fitPCA=embedding.fit(mean_coh[ind_use])
        print (np.sum(fitPCA.explained_variance_ratio_))
        pseudo_mds=embedding.fit_transform(mean_coh[ind_use])

        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        for jj in range(len(ind_use)):
            ax.scatter(pseudo_mds[jj,0],pseudo_mds[jj,1],pseudo_mds[jj,2],color=col[ind_use[jj]],alpha=alph[ind_use[jj]])
        ax.set_xlabel('PC1')
        ax.set_xlim([-10,10])
        ax.set_ylabel('PC2')
        ax.set_ylim([-10,10])
        ax.set_zlabel('PC3')
        ax.set_zlim([-10,10])
        plt.show()
