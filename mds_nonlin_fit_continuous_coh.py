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
dic_time['targ_on']=np.array([0,1000,200,200]) # time pre, time post, bin size, step size
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

models_vec=[(),(100),(100,100),(100,100,100),(100,100,100,100)]#,(100,100,100,100,100),(100,100,100,100,100,100)]
lr=1e-3
activation='relu'
reg=1e-3

# For "clase", the default lr initialization (lr=1e-3) converges well.
# For "clase_coh, it needs to be bigger (lr=1e-2) but still works much worse

monkeys=['Niels']#,'Galileo']
n_cv=10
for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/late/%s/'%(monkeys[k]) 
    files=os.listdir(abs_path)
    
    for kkk in range(len(talig_vec)):
        print (' ',talig_vec[kkk])
        steps=steps_dic[talig_vec[kkk]]
        xx=xx_dic[talig_vec[kkk]]
        r2_vec_pre=nan*np.zeros((len(files),steps,len(models_vec),n_cv,2))
        #r2_vec_coh_pre=nan*np.zeros((len(files),steps,n_cv,n_coh))
        conf_matrix_pre=nan*np.zeros((len(files),steps,n_cv,n_coh,n_coh,3))
        
        for kk in range(len(files)):
            print ('  ',files[kk])
            data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
            beha=miscellaneous.behavior(data)
            index_nonan=beha['index_nonan']
            reward=beha['reward']
            ind_corr=np.where(reward==1)[0]
            coherence=beha['coherence_signed'][ind_corr] #Cuidado!!
            context=beha['context'][ind_corr] #Cuidado!!
            coh_log=beha['coh_log'][ind_corr]#Cuidado!!
            coh_num=10*(beha['coh_num'][ind_corr]-7)#Cuidado!!
            coh_uq=np.unique(coh_num)
            #print (coh_uq)
            #coh_bin=np.array((beha['coh_num'][ind_corr]-7)>0,dtype=np.int16) #Cuidado!!
            firing_rate_pre=miscellaneous.getRasters(data,talig_vec[kkk],dic_time[talig_vec[kkk]],index_nonan,thres)
            firing_rate=miscellaneous.normalize_fr(firing_rate_pre)[ind_corr] #Cuidado!!
            
            for i in range(steps):
                print (i)
                # Best Model 
                # for j in range(len(models_vec)):
                #     print (' ',j)
                #     cv=ShuffleSplit(n_splits=n_cv,test_size=0.2)
                #     g=-1
                #     for train_index, test_index in cv.split(firing_rate):
                #         g=(g+1)
                #         mlpregress=MLPRegressor(models_vec[j],learning_rate_init=lr,alpha=reg,activation=activation)
                #         mlpregress.fit(firing_rate[:,:,i][train_index],coh_num[train_index])
                #         r2_vec_pre[kk,i,j,g,0]=mlpregress.score(firing_rate[:,:,i][train_index],coh_num[train_index])
                #         r2_vec_pre[kk,i,j,g,1]=mlpregress.score(firing_rate[:,:,i][test_index],coh_num[test_index])

                # Regressions with Context
                ind_model=0
                cv=ShuffleSplit(n_splits=n_cv,test_size=0.2)
                g=-1
                for train_index,test_index in cv.split(firing_rate):
                    g=(g+1)
                    mlpregress=MLPRegressor(models_vec[ind_model],learning_rate_init=lr,alpha=reg,activation=activation)
                    mlpregress.fit(firing_rate[train_index][:,:,i],coh_num[train_index])
                    
                    for c in range(n_coh):
                        try:
                            ind_coh=np.where(coh_num[test_index]==coh_uq[c])[0]
                            ind_coh0=np.where((coh_num[test_index]==coh_uq[c])&(context[test_index]==0))[0]
                            ind_coh1=np.where((coh_num[test_index]==coh_uq[c])&(context[test_index]==1))[0]
                            pred=mlpregress.predict(firing_rate[test_index][ind_coh][:,:,i])
                            pred0=mlpregress.predict(firing_rate[test_index][ind_coh0][:,:,i])
                            pred1=mlpregress.predict(firing_rate[test_index][ind_coh1][:,:,i])
                            # All
                            vec_pred=np.zeros(n_coh)
                            for f in range(len(pred)):
                                dist=(coh_uq-pred[f])**2
                                ind_min=np.argmin(dist)
                                vec_pred[ind_min]+=1
                            conf_matrix_pre[kk,i,g,c,:,0]=vec_pred/len(pred)
                            # Cont0
                            vec_pred0=np.zeros(n_coh)
                            for f in range(len(pred0)):
                                dist0=(coh_uq-pred0[f])**2
                                ind_min0=np.argmin(dist0)
                                vec_pred0[ind_min0]+=1
                            conf_matrix_pre[kk,i,g,c,:,1]=vec_pred0/len(pred0)
                            # Cont1
                            vec_pred1=np.zeros(n_coh)
                            for f in range(len(pred1)):
                                dist1=(coh_uq-pred1[f])**2
                                ind_min1=np.argmin(dist1)
                                vec_pred1[ind_min1]+=1
                            conf_matrix_pre[kk,i,g,c,:,2]=vec_pred1/len(pred1)
                        except:
                            #print ('Error coh ',coh_uq[c])
                            None

        print (np.mean(r2_vec_pre,axis=(0,3)))
        conf_matrix=np.nanmean(conf_matrix_pre,axis=(2))
        conf_matrix_m=np.nanmean(conf_matrix,axis=0)
        conf_matrix_sem=sem(conf_matrix,axis=0)
        for i in range(steps):
            print (xx[i])
            # plt.imshow(conf_matrix_m[i,1:-1,1:-1,0])
            # plt.xticks(range(13),np.unique(coherence)[1:-1])
            # plt.yticks(range(13),np.unique(coherence)[1:-1])
            # plt.ylabel('Tested on')
            # plt.xlabel('Model Prediction')
            # plt.title('All trials')
            # plt.colorbar()
            # plt.show()
            for ii in range(1,n_coh-1):
                plt.errorbar(np.arange(13),conf_matrix_m[i,ii,1:-1,0],yerr=conf_matrix_sem[i,ii,1:-1,0],color='black',label='All')
                plt.errorbar(np.arange(13),conf_matrix_m[i,ii,1:-1,1],yerr=conf_matrix_sem[i,ii,1:-1,1],color='green',label='Context Left')
                plt.errorbar(np.arange(13),conf_matrix_m[i,ii,1:-1,2],yerr=conf_matrix_sem[i,ii,1:-1,2],color='blue',label='Context Right')
                plt.title('Tested on Coherence %.1f'%(100*np.unique(coherence)[ii]))
                plt.ylabel('Percentage Prediction')
                plt.xticks(range(13),100*np.unique(coherence)[1:-1])
                plt.xlabel('Motion Coherence (%)')
                plt.ylim([0,0.45])
                plt.legend(loc='best')
                plt.show()

