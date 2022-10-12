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
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
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

# def shuffle_repr(neural,clase_all,ind_null):
#     clase_uq=np.unique(clase_all)
#     num_neu=len(neural[0])
#     pop_sh=nan*np.zeros(np.shape(neural))
#     for i in range(len(clase_uq)):
#         ind=np.where(clase_all==clase_uq[i])[0]
#         pop_sh[ind]=neural[ind][:,ind_null[]])
#     return pop_sh
    

# pre targets: context, prev choice, prev rew
# post dots: choice, context, diff
# pre-saccade: choice, context, diff?
# post-saccade: choice, context, reward

# target onset: 'targ_on', dots onset: 'dots_on', dots offset: 'dots_off', saccade: 'response_edf'
talig_vec=np.array(['dots_on'])#'targ_on','dots_on'
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
n_rand=100
perc_tr=0.8
thres=0
reg=1
n_coh=15
n_sh=100

models_vec=[(),(100),(100,100),(100,100,100),(100,100,100,100)]
lr=1e-3
activation='relu'
reg=1e-3

monkeys=['Niels'] #'Galileo',
phase='late'

for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/%s/%s/'%(phase,monkeys[k])
    files=os.listdir(abs_path)
    for kkk in range(len(talig_vec)):
        print ('  ',talig_vec[kkk])
        steps=steps_dic[talig_vec[kkk]]
        xx=xx_dic[talig_vec[kkk]]
        # Careful! in this function I am only using correct trials so that choice and stimulus are the same
        pseudo=miscellaneous.pseudopop_coherence_context(abs_path,files,talig_vec[kkk],dic_time[talig_vec[kkk]],steps_dic[talig_vec[kkk]],thres,nt,n_rand,perc_tr,True)
        pseudo_all=pseudo['pseudo_all']
        num_neu=len(pseudo_all[0,0,0])
        pseudo_tr=pseudo['pseudo_tr']
        pseudo_te=pseudo['pseudo_te']
        clase_ctx=pseudo['clase_ctx']
        clase_all=pseudo['clase_all']
        clase_all_uq=np.unique(clase_all)
        clase_coh=10*(pseudo['clase_coh']-7)
        r2_pre=nan*np.zeros((steps,n_rand,len(models_vec),2))
        para_score=nan*np.zeros((steps,n_rand))
        para_score_sh=nan*np.zeros((steps,n_sh,n_rand))
        ind_c0=np.where(clase_ctx==0)[0]
        ind_c1=np.where(clase_ctx==1)[0]  
        for i in range(steps):
            print (i)
            
            # Original
            for ii in range(n_rand):
                linreg0=LinearRegression()
                linreg0.fit(pseudo_all[i,ii,ind_c0],clase_coh[ind_c0])
                wei0=linreg0.coef_
                linreg1=LinearRegression()
                linreg1.fit(pseudo_all[i,ii,ind_c1],clase_coh[ind_c1])
                wei1=linreg1.coef_
                para_score[i,ii]=np.dot(wei0,wei1)/(np.linalg.norm(wei0)*np.linalg.norm(wei1))
                
            #Shuffled
            for iii in range(n_sh):
                #print ('  ',iii)
                ind_null=np.zeros((len(clase_all_uq),num_neu))
                for l in range(len(clase_all_uq)):
                    ind_null[l]=np.random.permutation(np.arange(num_neu))
                    
                for ii in range(n_rand):
                    pseudo_sh=nan*np.zeros(np.shape(pseudo_all[i,ii]))
                    for l in range(len(clase_all_uq)):
                        ind_c=np.where(clase_all==clase_all_uq[l])[0]
                        pseudo_sh[ind_c]=pseudo_all[i,ii,ind_c][:,np.array(ind_null[l],dtype=np.int16)]
                    linreg0=LinearRegression()
                    linreg0.fit(pseudo_sh[ind_c0],clase_coh[ind_c0])
                    wei0=linreg0.coef_
                    linreg1=LinearRegression()
                    linreg1.fit(pseudo_sh[ind_c1],clase_coh[ind_c1])
                    wei1=linreg1.coef_
                    para_score_sh[i,iii,ii]=np.dot(wei0,wei1)/(np.linalg.norm(wei0)*np.linalg.norm(wei1))

        para_score_m=np.mean(para_score,axis=1)
        para_score_sh_m=np.sort(np.mean(para_score_sh,axis=2),axis=1)
        plt.plot(xx,para_score_m,color='green')
        plt.fill_between(xx,para_score_sh_m[:,0],para_score_sh_m[:,-1],color='green',alpha=0.5)
        plt.plot(xx,np.zeros(len(xx)),color='black',linestyle='--')
        plt.xlabel('Time (sec)')
        plt.ylabel('Continuous Paralelism Score (cPS)')
        #plt.ylim([-0.05,0.2])
        plt.show()
                        
      

        #         print ('  ',ii)
        #         #Model Comparison
        #         for j in range(len(models_vec)):
        #             #print (' ',j)
        #             mlpregress=MLPRegressor(models_vec[j],learning_rate_init=lr,alpha=reg,activation=activation)
        #             mlpregress.fit(pseudo_tr[i,ii],clase_coh)
        #             r2_pre[i,ii,j,0]=mlpregress.score(pseudo_tr[i,ii],clase_coh)
        #             r2_pre[i,ii,j,1]=mlpregress.score(pseudo_te[i,ii],clase_coh)

        # r2_vec=np.mean(r2_pre,axis=1)
        # print (r2_vec)

        # col=['orange','green','green','green','green']
        # lab=['Linear','NonLin1','NonLin2','NonLin3','NonLin4']
        # alph=[1,0.3,0.5,0.7,0.9]
        # for l in range(len(models_vec)):
        #     plt.plot(xx,r2_vec[:,l,1],color=col[l],alpha=alph[l],label=lab[l])
        # plt.plot(xx,np.zeros(len(xx)),color='black',linestyle='--')
        # plt.legend(loc='best')
        # plt.xlabel('Time (sec)')
        # plt.ylabel('CV $R^{2}$')
        # plt.ylim([-0.38,1])
        # plt.show()
        
