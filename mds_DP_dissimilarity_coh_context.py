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
talig_vec=np.array(['response_edf'])#'targ_on','dots_on'
dic_time={} # same number of steps for all time locks
dic_time['targ_on']=np.array([0,1000,200,100]) # time pre, time post, bin size, step size
dic_time['dots_on']=np.array([0,1000,200,200])
dic_time['response_edf']=np.array([1000,200,200,200])
steps_dic={}
xx_dic={}
for i in range(len(talig_vec)):
    steps_dic[talig_vec[i]]=int((dic_time[talig_vec[i]][0]+dic_time[talig_vec[i]][1])/dic_time[talig_vec[i]][3])
    xx_dic[talig_vec[i]]=np.linspace(-dic_time[talig_vec[i]][0]/1000,dic_time[talig_vec[i]][1]/1000,steps_dic[talig_vec[i]],endpoint=False)

nt=50 #100 for coh signed, 200 for coh unsigned, 50 for coh signed with context
n_rand=10
perc_tr=0.8
thres=0
reg=1
n_coh=15

monkeys=['Niels','Galileo']
phase='late'

for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/%s/%s/'%(phase,monkeys[k])
    if k==0:
        col=['blue','blue','blue','blue','blue','blue','black','orange','orange','orange','orange','orange','orange','green','green','green','green','green','green','grey','red','red','red','red','red','red']
        alph=[0.6,0.5,0.4,0.3,0.2,0.1,1,0.1,0.2,0.3,0.4,0.5,0.6,0.6,0.5,0.4,0.3,0.2,0.1,1,0.1,0.2,0.3,0.4,0.5,0.6]
    if k==1:
        col=['blue','blue','blue','blue','blue','blue','blue','black','orange','orange','orange','orange','orange','orange','orange','green','green','green','green','green','green','green','grey','red','red','red','red','red','red','red']
        alph=[0.7,0.6,0.5,0.4,0.3,0.2,0.1,1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.7,0.6,0.5,0.4,0.3,0.2,0.1,1,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    files=os.listdir(abs_path)
    for kkk in range(len(talig_vec)):
        print ('  ',talig_vec[kkk])
        steps=steps_dic[talig_vec[kkk]]
        # Careful! in this function I am only using correct trials so that choice and stimulus are the same
        # Careful! Not normalizing firing rates
        pseudo=miscellaneous.pseudopop_coherence_context(abs_path,files,talig_vec[kkk],dic_time[talig_vec[kkk]],steps_dic[talig_vec[kkk]],thres,nt,n_rand,perc_tr,True)
        pseudo_all=pseudo['pseudo_all']
        pseudo_tr=pseudo['pseudo_tr']
        pseudo_te=pseudo['pseudo_te']
        clase_all=pseudo['clase_all']
        perf_pre=nan*np.zeros((steps,n_rand,2*n_coh,2*n_coh))
        for i in range(steps):
            print (i)
            for ii in range(n_rand):
                print ('  ',ii)  
                for j in range(2*n_coh):
                    for jj in range(j+1,2*n_coh):
                        ind_coh=np.where((clase_all==j)|(clase_all==jj))[0]
                        cl=LogisticRegression(C=1/reg,class_weight='balanced')
                        cl.fit(pseudo_tr[i,ii,ind_coh],clase_all[ind_coh])
                        perf_pre[i,ii,j,jj]=cl.score(pseudo_te[i,ii,ind_coh],clase_all[ind_coh])
                        
        # Plots
        vmin=0.4
        vmax=1.0
        perf=np.mean(perf_pre,axis=1)
        dprime=scipy.stats.norm.ppf(perf)
        if monkeys[k]=='Niels':
            perf=np.delete(perf,[0,14,15,29],axis=1)
            perf=np.delete(perf,[0,14,15,29],axis=2)
            dprime=np.delete(dprime,[0,14,15,29],axis=1)
            dprime=np.delete(dprime,[0,14,15,29],axis=2)
            
        for i in range(steps):
            print (xx_dic[talig_vec[kkk]][i])
            plt.imshow(perf[i],vmin=vmin,vmax=vmax)
            plt.colorbar()
            plt.show()
            
            # MDS
            diss_mat_pre=np.nanmean([dprime[i],dprime[i].T],axis=0)
            diss_min=np.nanmin(diss_mat_pre)
            diss_max=np.nanmax(diss_mat_pre[~np.isinf(diss_mat_pre)])
            diss_mat=(diss_mat_pre+abs(diss_min))
            diss_mat[np.isnan(diss_mat)]=0
            diss_mat[np.isinf(diss_mat)]=(diss_max+abs(diss_min))
            
            embedding=MDS(n_components=3,dissimilarity='precomputed')
            fitMDS=embedding.fit(diss_mat)
            pseudo_mds=embedding.fit_transform(diss_mat)

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111, projection='3d')
            for jj in range(len(diss_mat[0])):
                ax.scatter(pseudo_mds[jj,0],pseudo_mds[jj,1],pseudo_mds[jj,2],color=col[jj],alpha=alph[jj])
            ax.set_xlabel('Dim1')
            ax.set_xlim([-2,2])
            ax.set_ylabel('Dim2')
            ax.set_ylim([-2,2])
            ax.set_zlabel('Dim3')
            ax.set_zlim([-2,2])
            plt.show()

            

 
   
