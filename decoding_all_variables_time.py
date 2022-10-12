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

# Number of neurons
# Niels
# all neurons: 337 (~220 trials per coh), 334 (~220 trials), 347 (~220 trials)
# thres 0.1:   280                        287                296
# thres 0.5:   238                        247                242
# thres 1.0:   217                        218                214
# Galileo
# all neurons: 269 (~150 trials), 268 (~180 trials), 270 (~140 trials)
# thres 0.1:   258                257                239
# thres 0.5:   210                213                188
# thres 1.0:   186                184                164

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

    
thres=1
n_cv=5
reg=1

quant=['stimulus_0','choice_0','context_0','difficulty_0','reward_0','stimulus_m1','choice_m1','context_m1','difficulty_m1','reward_m1']
col=['green','blue','brown','purple','red','lime','royalblue','orange','pink','salmon']
monkeys=['Niels','Galileo']

for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/%s/'%monkeys[k] 
    files=os.listdir(abs_path)
    perf=nan*np.zeros((3,len(talig_vec),steps,8,len(quant),2)) # files, time locks, steps, coherences, (Stimulus, Choice, Context, Difficulty), train/test
    corr_vars=nan*np.zeros((len(files),len(quant),len(quant)))
    for kk in range(len(files)):
        #Load data
        print ('  ',files[kk])
        data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
        behavior=miscellaneous.behavior(data)
        index_nonan=behavior['index_nonan']
        response=behavior['response']
        choice=behavior['choice']
        stimulus=behavior['stimulus']
        coh=behavior['coherence']
        coh_unique=behavior['coherence_uq']
        context=behavior['context']
        difficulty=behavior['difficulty']
        reward=behavior['reward']

        for h in range(len(quant)):
            for hh in range(len(quant)):
                corr_vars[kk,h,hh]=pearsonr(behavior[quant[h]],behavior[quant[hh]])[0]

    corr_m=np.mean(corr_vars,axis=0)
    plt.imshow(corr_m)
    plt.colorbar()
    #plt.xticks([np.arange(len(quant))],quant,rotation=90)
    #plt.yticks([np.arange(len(quant))],quant)
    plt.show()
        
        # for kkk in range(len(talig_vec)):
    #         print ('    ',talig_vec[kkk])
    #         firing_rate=miscellaneous.getRasters(data,talig_vec[kkk],dic_time[talig_vec[kkk]],index_nonan,thres)
    #         #print (np.shape(firing_rate))
    #         for i in range(steps):
    #             #print ('step ',i)
    #             for ii in range(len(coh_unique)):
    #                 #print ('cohe ',ii)
    #                 ind_coh=np.where(coh==coh_unique[ii])[0]
    #                 for hh in range(len(quant)):
    #                     if quant[hh]=='stimulus':
    #                         neu_dec=firing_rate[ind_coh,:,i]
    #                         clase=stimulus[ind_coh]
    #                     if quant[hh]=='choice':
    #                         neu_dec=firing_rate[ind_coh,:,i]
    #                         clase=choice[ind_coh]
    #                     if quant[hh]=='context':
    #                         neu_dec=firing_rate[ind_coh,:,i]
    #                         clase=context[ind_coh]
    #                     if quant[hh]=='difficulty':
    #                         neu_dec=firing_rate[:,:,i]
    #                         clase=difficulty.copy()
    #                     if quant[hh]=='reward':
    #                         neu_dec=firing_rate[:,:,i]
    #                         clase=reward.copy()
    #                     if quant[hh]=='stimulus_m1':
    #                         neu_dec=firing_rate[ind_coh,:,i][1:]
    #                         clase=stimulus[ind_coh][0:-1]
    #                     if quant[hh]=='choice_m1':
    #                         neu_dec=firing_rate[ind_coh,:,i][1:]
    #                         clase=choice[ind_coh][0:-1]
    #                     if quant[hh]=='context_m1':
    #                         neu_dec=firing_rate[ind_coh,:,i][1:]
    #                         clase=context[ind_coh][0:-1]
    #                     if quant[hh]=='difficulty_m1':
    #                         neu_dec=firing_rate[:,:,i][1:]
    #                         clase=difficulty.copy()[0:-1]
    #                     if quant[hh]=='reward_m1':
    #                         neu_dec=firing_rate[:,:,i][1:]
    #                         clase=reward.copy()[0:-1]
    #                     perf[kk,kkk,i,ii,hh]=miscellaneous.classifier(neu_dec,clase,n_cv,reg)

    # perf_m=np.nanmean(perf,axis=0)
    # perf_sem=sem(perf,axis=0,nan_policy='omit')

    # #####################
    # if monkeys[k]=='Niels':
    #     n_coh=7
    # if monkeys[k]=='Galileo':
    #     n_coh=8
    # # Plots
    # tl_vec=['targets on','dots on','saccade']
    # alph=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # fig=plt.figure(figsize=(len(tl_vec)*6,len(quant)*4))
    # for kk in range(len(quant)):
    #     for kkk in range(len(talig_vec)):
    #         ax=fig.add_subplot(len(quant),len(tl_vec),kk*3+kkk+1)
    #         miscellaneous.adjust_spines(ax,['left','bottom'])
    #         if kk==0:
    #             ax.set_title('Time lock %s'%tl_vec[kkk])
    #         if kkk==0:
    #             ax.set_ylabel('Decoding Performance \n %s'%quant[kk])
    #         if kk==(len(quant)-1):
    #             ax.set_xlabel('Time from %s (sec)'%tl_vec[kkk])
    #         if kk==0 and kkk==0:
    #             for i in range(n_coh):
    #                 ax.plot(xx_dic[talig_vec[kkk]],perf_m[kkk,:,i,kk,1],color=col[kk],alpha=alph[i],label='%.1f'%(coh_unique[i]*100))
    #                 ax.fill_between(xx_dic[talig_vec[kkk]],perf_m[kkk,:,i,kk,1]-perf_sem[kkk,:,i,kk,1],perf_m[kkk,:,i,kk,1]+perf_sem[kkk,:,i,kk,1],color=col[kk],alpha=0.8*alph[i])
    #             plt.legend(loc='best')
    #         if kk!=0 or kkk!=0:
    #             for i in range(n_coh):
    #                 ax.plot(xx_dic[talig_vec[kkk]],perf_m[kkk,:,i,kk,1],color=col[kk],alpha=alph[i])
    #                 ax.fill_between(xx_dic[talig_vec[kkk]],perf_m[kkk,:,i,kk,1]-perf_sem[kkk,:,i,kk,1],perf_m[kkk,:,i,kk,1]+perf_sem[kkk,:,i,kk,1],color=col[kk],alpha=0.8*alph[i])
    #         ax.axvline(0,color='black',linestyle='--')
    #         ax.set_ylim([0.4,1.0])
    #         ax.plot(xx_dic[talig_vec[kkk]],0.5*np.ones(steps),color='black',linestyle='--')
    # fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/decoding_%s_bin_%i_thres_%.1f_all.pdf'%(monkeys[k],dic_time['dots_on'][2],thres),dpi=500,bbox_inches='tight')

