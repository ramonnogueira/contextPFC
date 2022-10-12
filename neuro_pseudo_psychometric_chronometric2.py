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

###################################################3

monkeys=['Niels']#,'Galileo']#

talig_vec=np.array(['dots_on'])#,'response_edf'])#'targ_on','dots_on'
dic_time={} # same number of steps for all time locks
dic_time['dots_on']=np.array([0,600,200,200]) # time pre, time post, bin size, step size
dic_time['response_edf']=np.array([600,0,100,100]) # time pre, time post, bin size, step size
steps_dic={}
xx_dic={}
for i in range(len(talig_vec)):
    steps_dic[talig_vec[i]]=int((dic_time[talig_vec[i]][0]+dic_time[talig_vec[i]][1])/dic_time[talig_vec[i]][3])
    xx_dic[talig_vec[i]]=np.linspace(-dic_time[talig_vec[i]][0]/1000,dic_time[talig_vec[i]][1]/1000,steps_dic[talig_vec[i]],endpoint=False)

nt=50 #100 for coh signed, 200 for coh unsigned, 50 for coh signed with context
n_rand=10
perc_tr=0.8
thres=0
reg=1e-3
n_coh=15

coh_plot=[['-75','-51.2','-25.6','-12.8','-6.4','-3.2','-1.6','0','1.6','3.2','6.4','12.8','25.6','51.2','75'],
          ['-51.2','-25.6','-12.8','-6.4','-4.5','-3.2','-1.6','0','1.6','3.2','4.5','6.4','12.8','25.6','51.2']]

for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/late/%s/'%(monkeys[k])
    files=os.listdir(abs_path)
    for kkk in range(len(talig_vec)):
        print ('  ',talig_vec[kkk])
        steps=steps_dic[talig_vec[kkk]]
        xx=xx_dic[talig_vec[kkk]]
        
        pseudo=miscellaneous.pseudopop_coherence_context_correct(abs_path,files,talig_vec[kkk],dic_time[talig_vec[kkk]],steps_dic[talig_vec[kkk]],thres,nt,n_rand,perc_tr,True)
        pseudo_tr=pseudo['pseudo_tr']
        pseudo_te=pseudo['pseudo_te']
        clase_all=pseudo['clase_all']
        clase_coh=pseudo['clase_coh']
        indc0=np.where(clase_coh==7)[0]
        clase_ctx=pseudo['clase_ctx']
        stim=nan*np.zeros(len(clase_coh))
        stim[clase_coh>7]=1
        stim[clase_coh<7]=0
        stim[indc0]=np.array(np.random.normal(0,1,len(indc0))>0,dtype=np.int16)
        perf_pre=nan*np.zeros((steps,n_rand,n_coh,n_coh,3))
        ccgp_pre=nan*np.zeros((steps,n_rand,n_coh,n_coh,2))
        chrono_neuro_all_pre=nan*np.zeros((steps,n_rand,2,2))
        psycho_neuro_all_pre=nan*np.zeros((steps,n_rand,2,2))
        chrono_neuro_pre=nan*np.zeros((steps,n_rand,n_coh,3))
        psycho_neuro_pre=nan*np.zeros((steps,n_rand,n_coh,3))
        for i in range(steps):
            print (i)
            for ii in range(n_rand):
                #print ('  ',ii)
                if monkeys[k]=='Niels':
                    pseudo_tr_def=np.delete(pseudo_tr[i,ii],np.where((clase_coh==0)|(clase_coh==14))[0],0)
                    pseudo_te_def=np.delete(pseudo_te[i,ii],np.where((clase_coh==0)|(clase_coh==14))[0],0)
                    range_coh=np.delete(np.arange(15),[0,14])
                    stim_def=np.delete(stim,np.where((clase_coh==0)|(clase_coh==14))[0])
                if monkeys[k]=='Galileo':
                    pseudo_tr_def=pseudo_tr[i,ii]
                    pseudo_te_def=pseudo_te[i,ii]
                    range_coh=np.arange(15)
                    stim_def=stim.copy()
                cl=LogisticRegression(C=1/reg,class_weight='balanced')
                cl.fit(pseudo_tr_def,stim_def)
                    
                for j in range_coh:
                    try:
                        ind_coh=np.where((clase_coh==j))[0]
                        ind_coh0=np.where((clase_coh==j)&(clase_ctx==0))[0] # Left more rewarded
                        ind_coh1=np.where((clase_coh==j)&(clase_ctx==1))[0] # Right more rewarded
                        chrono_neuro_pre[i,ii,j,0]=cl.score(pseudo_te[i,ii][ind_coh],stim[ind_coh])
                        chrono_neuro_pre[i,ii,j,1]=cl.score(pseudo_te[i,ii][ind_coh0],stim[ind_coh0])
                        chrono_neuro_pre[i,ii,j,2]=cl.score(pseudo_te[i,ii][ind_coh1],stim[ind_coh1])
                        psycho_neuro_pre[i,ii,j,0]=np.mean(cl.predict(pseudo_te[i,ii][ind_coh]))
                        psycho_neuro_pre[i,ii,j,1]=np.mean(cl.predict(pseudo_te[i,ii][ind_coh0]))
                        psycho_neuro_pre[i,ii,j,2]=np.mean(cl.predict(pseudo_te[i,ii][ind_coh1]))
                    except:
                        print ('Error %i %i %i'%(i,ii,j))

        #psycho_neuro_all=np.mean(psycho_neuro_all_pre,axis=(1))
        psycho_neuro_m=np.nanmean(psycho_neuro_pre,axis=(1))
        #psycho_neuro_std=np.std(psycho_neuro_pre,axis=(1))
        #chrono_neuro_all=np.mean(chrono_neuro_all_pre,axis=(1))
        chrono_neuro_m=np.nanmean(chrono_neuro_pre,axis=(1))
        #chrono_neuro_std=np.std(chrono_neuro_pre,axis=(1))
        #print (psycho_neuro_all)
        print (psycho_neuro_m)
        #print (chrono_neuro_all)
        print (chrono_neuro_m)

        for i in range(steps):
            print (xx[i])
            plt.plot(np.arange(15),psycho_neuro_m[i,:,0],color='black')
            plt.plot(np.arange(15),psycho_neuro_m[i,:,1],color='green')
            plt.plot(np.arange(15),psycho_neuro_m[i,:,2],color='blue')
            plt.ylim([0,1])
            plt.ylabel('Prob. Right Response')
            #plt.title('Neural psychometric')
            plt.xlabel('Motion Coherence (%)')
            plt.axvline(7,color='black',linestyle='--')
            plt.xticks(range(15),coh_plot[k])
            plt.plot(0.5*np.ones(15),color='black',linestyle='--')
            plt.show()

        for i in range(steps):
            print (xx[i])
            plt.plot(np.arange(15),chrono_neuro_m[i,:,0],color='black')
            plt.plot(np.arange(15),chrono_neuro_m[i,:,1],color='green')
            plt.plot(np.arange(15),chrono_neuro_m[i,:,2],color='blue')
            plt.ylim([0,1])
            plt.ylabel('Prob. Correct')
            #plt.title('Neural Chronometric')
            plt.xlabel('Motion Coherence (%)')
            plt.axvline(7,color='black',linestyle='--')
            plt.xticks(range(15),coh_plot[k])
            plt.plot(0.5*np.ones(15),color='black',linestyle='--')
            plt.show()
            
