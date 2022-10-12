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

aligned='dots_on'
dic_time=np.array([0,1000,200,200]) # time pre, time post, bin size, step size
#aligned='response_edf'
#dic_time=np.array([700,100,200,200]) # time pre, time post, bin size, step size
steps=int((dic_time[0]+dic_time[1])/dic_time[3])
xx=np.linspace(-dic_time[0]/1000,dic_time[1]/1000,steps,endpoint=False)

#group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7])
group_coh=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7])

monkeys=['Niels','Galileo']
n_cv=50
test_size=0.25
for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/late/%s/'%(monkeys[k]) 
    files=os.listdir(abs_path)

    psycho_pre=nan*np.zeros((len(files),steps,n_cv,15,3))
    chrono_pre=nan*np.zeros((len(files),steps,n_cv,15,3))
    for kk in range(len(files)):
        print ('  ',files[kk])
        data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
        beha=miscellaneous.behavior(data,group_coh=group_coh)
        index_nonan=beha['index_nonan']
        reward=beha['reward']
        ind_corr=np.where(reward==1)[0]
        coherence=beha['coherence_signed']#[ind_corr] #Cuidado!!
        context=beha['context']#[ind_corr] #Cuidado!!
        stimulus=beha['stimulus']#[ind_corr] #Cuidado!!
        choice=beha['choice']#[ind_corr] #Cuidado!!
        coh_uq=np.unique(coherence)
        rt=beha['reaction_time']#[ind_corr] #Cuidado!!
        firing_rate_pre=miscellaneous.getRasters(data,aligned,dic_time,index_nonan,threshold=1)
        firing_rate=miscellaneous.normalize_fr(firing_rate_pre)#[ind_corr] #Cuidado!!
        num_neu=len(firing_rate[0])

        reg=1
        tpre_sacc=50 # To avoid RT-contaminated trials this should be positive
        for j in range(steps):
            if aligned=='dots_on':
                max_t=(-dic_time[0]+j*dic_time[3]+dic_time[2]+tpre_sacc) # -initial + j*step_size + bin_size + tpre_sacc
                ind_rt=np.where(rt>max_t)[0]
            if aligned=='response_edf':
                ind_rt=np.arange(len(rt))
            firing_rt=firing_rate[ind_rt]
            choice_rt=choice[ind_rt]
            #choice_rt=stimulus[ind_rt]

            cv=StratifiedShuffleSplit(n_splits=n_cv,test_size=test_size)
            g=-1
            for train_index, test_index in cv.split(firing_rt,choice_rt):
                g=(g+1)
                
                cl=LogisticRegression(C=1/reg,class_weight='balanced')
                cl.fit(firing_rt[:,:,j][train_index],choice_rt[train_index])

                choice_te=choice_rt[test_index]
                firing_te=firing_rt[test_index]
                coherence_te=coherence[ind_rt][test_index]
                context_te=context[ind_rt][test_index]
            
                for jj in range(len(coh_uq)):
                    ind=np.where((coherence_te==coh_uq[jj]))[0]
                    ind_ctx0=np.where((coherence_te==coh_uq[jj])&(context_te==0))[0]
                    ind_ctx1=np.where((coherence_te==coh_uq[jj])&(context_te==1))[0]
                    try:
                        psycho_pre[kk,j,g,jj,0]=np.mean(cl.predict(firing_te[ind][:,:,j]))
                        psycho_pre[kk,j,g,jj,1]=np.mean(cl.predict(firing_te[ind_ctx0][:,:,j]))
                        psycho_pre[kk,j,g,jj,2]=np.mean(cl.predict(firing_te[ind_ctx1][:,:,j]))
                        chrono_pre[kk,j,g,jj,0]=cl.score(firing_te[ind][:,:,j],choice_te[ind])
                        chrono_pre[kk,j,g,jj,1]=cl.score(firing_te[ind_ctx0][:,:,j],choice_te[ind_ctx0])
                        chrono_pre[kk,j,g,jj,2]=cl.score(firing_te[ind_ctx1][:,:,j],choice_te[ind_ctx1])
                    except:
                        print ('except ',j,g,jj)

    psycho=np.nanmean(psycho_pre,axis=2)
    chrono=np.nanmean(chrono_pre,axis=2)
    #
    psycho_m=np.nanmean(psycho,axis=0)
    psycho_sem=sem(psycho,axis=0,nan_policy='omit')
    if monkeys[k]=='Niels':
        psycho_m=np.delete(psycho_m,[0,14],1)
        psycho_sem=np.delete(psycho_sem,[0,14],1)
    #    
    chrono_m=np.nanmean(chrono,axis=0)
    chrono_sem=sem(chrono,axis=0,nan_policy='omit')
    if monkeys[k]=='Niels':
        chrono_m=np.delete(chrono_m,[0,14],1)
        chrono_sem=np.delete(chrono_sem,[0,14],1)

    # Psychometric
    for j in range(steps):
        print (j)
        plt.plot(np.arange(len(psycho_m[0])),psycho_m[j,:,0],color='black')
        plt.fill_between(np.arange(len(psycho_m[0])),psycho_m[j,:,0]-psycho_sem[j,:,0],psycho_m[j,:,0]+psycho_sem[j,:,0],color='black',alpha=0.5)
        plt.plot(np.arange(len(psycho_m[0])),psycho_m[j,:,1],color='green')
        plt.fill_between(np.arange(len(psycho_m[0])),psycho_m[j,:,1]-psycho_sem[j,:,1],psycho_m[j,:,1]+psycho_sem[j,:,1],color='green',alpha=0.5)
        plt.plot(np.arange(len(psycho_m[0])),psycho_m[j,:,2],color='blue')
        plt.fill_between(np.arange(len(psycho_m[0])),psycho_m[j,:,2]-psycho_sem[j,:,2],psycho_m[j,:,2]+psycho_sem[j,:,2],color='blue',alpha=0.5)
        plt.ylim([0,1])
        plt.show()

    # Chronometric
    for j in range(steps):
        print (j)
        plt.plot(np.arange(len(chrono_m[0])),chrono_m[j,:,0],color='black')
        plt.fill_between(np.arange(len(chrono_m[0])),chrono_m[j,:,0]-chrono_sem[j,:,0],chrono_m[j,:,0]+chrono_sem[j,:,0],color='black',alpha=0.5)
        plt.plot(np.arange(len(chrono_m[0])),chrono_m[j,:,1],color='green')
        plt.fill_between(np.arange(len(chrono_m[0])),chrono_m[j,:,1]-chrono_sem[j,:,1],chrono_m[j,:,1]+chrono_sem[j,:,1],color='green',alpha=0.5)
        plt.plot(np.arange(len(chrono_m[0])),chrono_m[j,:,2],color='blue')
        plt.fill_between(np.arange(len(chrono_m[0])),chrono_m[j,:,2]-chrono_sem[j,:,2],chrono_m[j,:,2]+chrono_sem[j,:,2],color='blue',alpha=0.5)
        plt.ylim([0,1])
        plt.show()
