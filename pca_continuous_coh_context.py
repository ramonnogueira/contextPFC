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

# target onset: 'targ_on', dots onset: 'dots_on', dots offset: 'dots_off', saccade: 'response_edf'
talig_vec=np.array(['dots_on'])#'targ_on','dots_on'
dic_time={} # same number of steps for all time locks
dic_time['dots_on']=np.array([0,1000,200,200])# time pre, time post, bin size, step size
dic_time['response_edf']=np.array([1000,200,200,100])
steps_dic={}
xx_dic={}
for i in range(len(talig_vec)):
    steps_dic[talig_vec[i]]=int((dic_time[talig_vec[i]][0]+dic_time[talig_vec[i]][1])/dic_time[talig_vec[i]][3])
    xx_dic[talig_vec[i]]=np.linspace(-dic_time[talig_vec[i]][0]/1000,dic_time[talig_vec[i]][1]/1000,steps_dic[talig_vec[i]],endpoint=False)

nt=50 #100 for coh signed, 200 for coh unsigned, 50 for coh signed with context
n_rand=30
perc_tr=0.8
thres=0
reg=1
n_coh=15

monkeys=['Niels','Galileo']
for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/late/%s/'%(monkeys[k]) 
    files=os.listdir(abs_path)
    if monkeys[k]=='Niels':
        col=['darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','black','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','purple','purple','purple','purple','purple','purple','black','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue']
        alph=[0.6,0.5,0.4,0.3,0.2,0.1,1,0.1,0.2,0.3,0.4,0.5,0.6,0.6,0.5,0.4,0.3,0.2,0.1,1,0.1,0.2,0.3,0.4,0.5,0.6]
    if monkeys[k]=='Galileo':
        col=['darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','black','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','purple','purple','purple','purple','purple','purple','purple','black','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue']
        alph=[0.7,0.6,0.5,0.4,0.3,0.2,0.1,1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.7,0.6,0.5,0.4,0.3,0.2,0.1,1,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    for kkk in range(len(talig_vec)):
        print ('  ',talig_vec[kkk])
        steps=steps_dic[talig_vec[kkk]]
        xx=xx_dic[talig_vec[kkk]]
        #pseudo=miscellaneous.pseudopop_coherence(abs_path,files,talig_vec[kkk],dic_time[talig_vec[kkk]],steps_dic[talig_vec[kkk]],thres,nt,n_rand,perc_tr,True)
        pseudo=miscellaneous.pseudopop_coherence_context(abs_path,files,talig_vec[kkk],dic_time[talig_vec[kkk]],steps_dic[talig_vec[kkk]],thres,nt,n_rand,perc_tr,True)
        pseudo_all=pseudo['pseudo_all']
        pseudo_tr=pseudo['pseudo_tr']
        pseudo_te=pseudo['pseudo_te']
        #clase=pseudo['clase']
        clase_all=pseudo['clase_all']
        clase_coh=pseudo['clase_coh']
        num_neu=len(pseudo_all[0,0,0])
        # clase_feat=np.zeros((len(clase),1))
        # clase_feat[:,0]=clase
        # #r2_vec_pre=nan*np.zeros((steps,len(models_vec),n_rand,num_neu,2))
        # r2_vec_pre=nan*np.zeros((steps,len(models_vec),n_rand,2))
        for i in range(steps):
            print (xx[i])
            # for j in range(len(models_vec)):
        #         print (' ',j)
        #         for ii in range(n_rand):
        #             #mlpregress=MLPRegressor(models_vec[j])
        #             mlpregress=MLPRegressor(models_vec[j],learning_rate_init=lr,alpha=reg,activation=activation)
        #             mlpregress.fit(pseudo_tr[i,ii],clase_coh)
        #             r2_vec_pre[i,j,ii,0]=mlpregress.score(pseudo_tr[i,ii],clase_coh)
        #             r2_vec_pre[i,j,ii,1]=mlpregress.score(pseudo_te[i,ii],clase_coh)
        #             #mlpregress.fit(clase_feat,pseudo_tr[i,ii])
        #             #r2_vec_pre[i,j,ii,:,0]=mlpregress.score(clase_feat,pseudo_tr[i,ii])
        #             #r2_vec_pre[i,j,ii,:,1]=mlpregress.score(clase_feat,pseudo_te[i,ii])

        # #r2_vec=np.mean(r2_vec_pre,axis=(2,3))
        # r2_vec=np.mean(r2_vec_pre,axis=(2))
        # print (r2_vec)
                
            mean_coh_pre=nan*np.zeros((n_rand,2*n_coh,num_neu))
            for ii in range(n_rand):
                for j in range(2*n_coh):
                    ind_coh=np.where(clase_all==j)[0]
                    mean_coh_pre[ii,j]=np.nanmean(pseudo_all[i,ii][ind_coh],axis=0)
            mean_coh=np.nanmean(mean_coh_pre,axis=0)
            if monkeys[k]=='Niels':
                mean_coh=np.delete(mean_coh,[0,14,15,29],axis=0)

            ind_nan=np.where(np.isnan(np.sum(mean_coh,axis=1)))[0]
            ind_use=np.delete(np.arange(len(mean_coh)),ind_nan,axis=0)
            
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
