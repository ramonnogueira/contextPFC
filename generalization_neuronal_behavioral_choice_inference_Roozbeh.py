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
from scipy.stats import spearmanr
from numpy.random import permutation
import miscellaneous
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy.stats import ortho_group 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from scipy import stats
from scipy.optimize import curve_fit
plt.close('all')
nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def log_curve(x,a,c):
    num=1+np.exp(-a*(x+c))
    return 1/num

def func_eval(index,t_back,t_forw,stimulus,choice,new_ctx):
    pp=nan*np.zeros((2,t_forw+t_back))
    for j in range(t_back):
        indj=(index-t_back+j)
        try:
            if new_ctx=='right':
                pp[int(stimulus[indj]),j]=choice[indj]
            if new_ctx=='left':
                pp[int(stimulus[indj]),j]=int(1-choice[indj])
        except:
            None
    for j in range(t_forw):
        indj=(index+j)
        try:
            if new_ctx=='right':
                pp[int(stimulus[indj]),j+t_back]=choice[indj]
            if new_ctx=='left':
                pp[int(stimulus[indj]),j+t_back]=int(1-choice[indj])
        except:
            None
    return pp

# Extract indices for training classifier (remove the one for testing from the entire dataset) and fit the classifier
def ret_ind_train(coherence,ind_ch,t_back,t_forw):      
    ind_train=np.arange(len(coherence))
    for p in range(len(ind_ch)):
        ind_t=np.arange(t_back+t_forw)-t_back+ind_ch[p]
        ind_del=[]
        for pp in range(len(ind_t)):
            try:
                ind_del.append(np.where(ind_train==ind_t[pp])[0][0])
            except:
                None
                #print ('error aqui')
        ind_del=np.array(ind_del)
        ind_train=np.delete(ind_train,ind_del)
    return ind_train


#################################################

monkeys=['Niels']#,'Galileo']
stage='late'
nback=30

talig='dots_on' #'response_edf' #dots_on
thres=0
reg=1e-3
maxfev=100000
method='dogbox'

#xx=np.arange(t_back+t_forw)-t_back

t_after=10 # 1 is for the inference trial

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])

# Careful with length of files groups
delta_beha_all=nan*np.zeros((2,2,5,len(monkeys)))
delta_neuro_all=nan*np.zeros((2,2,5,len(monkeys)))

for k in range(len(monkeys)):
    if monkeys[k]=='Niels':
        dic_time=np.array([0,200,200,200])# time pre, time post, bin size, step size (time pre always positive)
        if stage=='early':
            files_groups=[[0,1],[1,2],[2,3],[3,4]]
        if stage=='mid':
            files_groups=[[4,5],[5,6],[6,7],[7,8]]
        if stage=='late':
            files_groups=[[8,9],[9,10],[10,11],[11,12]]
    if monkeys[k]=='Galileo':
        dic_time=np.array([0,300,300,300])# time pre, time post, bin size, step size (time pre always positive)
        if stage=='early':
            files_groups=[[0,2],[2,4],[4,6],[6,8],[8,10]]
        if stage=='mid':
            files_groups=[[10,12],[12,14],[14,16],[16,18],[18,20]]
        if stage=='late':
            files_groups=[[20,22],[22,24],[24,26],[26,28],[28,30]]
        
    abs_path='/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/data/unsorted/%s/'%(monkeys[k]) 
    files_pre=np.array(os.listdir(abs_path))
    order=miscellaneous.order_files(files_pre)
    files_all=np.array(files_pre[order])
    print (files_all)

    beha_te_unte=nan*np.zeros((2,2,len(files_groups)))
    
    for hh in range(len(files_groups)):
        beha_tested_rlow=[]
        beha_tested_rhigh=[]
        beha_untested_rlow=[]
        beha_untested_rhigh=[]
        #neuro_tested_rlow=[]
        #neuro_tested_rhigh=[]
        #neuro_untested_rlow=[]
        #neuro_untested_rhigh=[]
        files=files_all[files_groups[hh][0]:files_groups[hh][1]]
        print (files)
        
        for kk in range(len(files)):
            print (files[kk])
            #Load data
            data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
            beha=miscellaneous.behavior(data,group_ref)
            index_nonan=beha['index_nonan']
            # We discard first trial of session because we are interested in context changes
            stimulus=beha['stimulus'][1:]
            choice=beha['choice'][1:]
            coherence=beha['coherence_signed'][1:]
            coh_uq=np.unique(coherence)
            reward=beha['reward'][1:]
            rt=beha['reaction_time'][1:]
            context_prepre=beha['context']
            ctx_ch=(context_prepre[1:]-context_prepre[0:-1])
            context_pre=context_prepre[1:] 
            ind_ch_pre=np.where(abs(ctx_ch)==1)[0] # ind_ch_pre index where there is a context change
            #ind_ch=np.where(abs(ctx_ch)==1)[0] # ind_ch_pre index where there is a context change
            indch_ct01_pre=np.where(ctx_ch==1)[0]
            indch_ct10_pre=np.where(ctx_ch==-1)[0]
            ind_ch=miscellaneous.calculate_ind_ch_corr(ind_ch_pre,reward) # ind_ch first correct trial after context change (otherwise animal doesn't know there was a change)
            context=miscellaneous.create_context_subj(context_pre,ind_ch_pre,ind_ch) # Careful! this is subjective context
            ind_ch01_s0,ind_ch01_s1,ind_ch10_s0,ind_ch10_s1=miscellaneous.calculate_ind_ch_corr_stim(indch_ct01_pre,indch_ct10_pre,reward,stimulus)

            firing_rate_pre=miscellaneous.getRasters_unsorted(data,talig,dic_time,index_nonan,threshold=thres)
            firing_rate=miscellaneous.normalize_fr(firing_rate_pre)[1:,:,0]

            print (ind_ch01_s0,ind_ch01_s1,ind_ch10_s0,ind_ch10_s1)
            xx=np.array([100*coherence]).T
        
            ##################################################
            # Behavior
            # Probability of Choice = Context for all possibilities: 01 0, 01 1, 10 0, 10 1

            # Current high is Right and current low is Left. Previous low is Right and previous high is Left. First reward trial after context change is Left.
            # rlow tested is left, rhigh untested is right
            ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
            for h in range(len(ind_ch01_s0)):
                ind_pre=(np.arange(nback)-nback+ind_ch01_s0[h]+1)
                ind_used[ind_pre]=True
                #ind_used[np.isnan(rt)]=False
            for h in range(len(ind_ch01_s0)):
                cl=LogisticRegression(C=reg,class_weight='balanced')
                cl.fit(xx[ind_used],choice[ind_used])
                exp_ch=cl.predict_proba(xx[(ind_ch01_s0[h]+t_after):(ind_ch01_s0[h]+t_after+1)])[0]
                dev=(choice[ind_ch01_s0[h]+t_after]-exp_ch[1]) # if there is inference this number should be larger than 0 
            #    ch_pre=np.mean(choice[ind_used])
            #    dev=(choice[ind_ch01_s0[h]+t_after]-ch_pre) # if there is inference this number should be larger than 0
                if stimulus[ind_ch01_s0[h]+t_after]==0: 
                    beha_tested_rlow.append(dev)
                if stimulus[ind_ch01_s0[h]+t_after]==1:
                    beha_untested_rhigh.append(dev)

            # Current high is Right and current low is Left. Previous low is Right and previous high is Left. First reward trial after context change is Right.
            # rlow untested is left, rhigh tested is right
            ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
            for h in range(len(ind_ch01_s1)):
                ind_pre=(np.arange(nback)-nback+ind_ch01_s1[h]+1)
                ind_used[ind_pre]=True
                #ind_used[np.isnan(rt)]=False
            for h in range(len(ind_ch01_s1)):
                cl=LogisticRegression(C=reg,class_weight='balanced')
                cl.fit(xx[ind_used],choice[ind_used])
                exp_ch=cl.predict_proba(xx[(ind_ch01_s1[h]+t_after):(ind_ch01_s1[h]+t_after+1)])[0]
                dev=(choice[ind_ch01_s1[h]+t_after]-exp_ch[1]) # if there is inference this number should be larger than 0 
                #ch_pre=np.mean(choice[ind_used])
                #dev=(choice[ind_ch01_s1[h]+t_after]-ch_pre) # if there is inference this number should be larger than 0
                if stimulus[ind_ch01_s1[h]+t_after]==0:
                    beha_untested_rlow.append(dev)
                if stimulus[ind_ch01_s1[h]+t_after]==1:
                    beha_tested_rhigh.append(dev)

            # Current high is Left and current low is Right. Previous low is Left and previous high is Right. First reward trial after context change is Left.
            # rhigh tested is left, rlow untested is right
            ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
            for h in range(len(ind_ch10_s0)):
                ind_pre=(np.arange(nback)-nback+ind_ch10_s0[h]+1)
                ind_used[ind_pre]=True
                #ind_used[np.isnan(rt)]=False
            for h in range(len(ind_ch10_s0)):
                cl=LogisticRegression(C=reg,class_weight='balanced')
                cl.fit(xx[ind_used],choice[ind_used])
                exp_ch=cl.predict_proba(xx[(ind_ch10_s0[h]+t_after):(ind_ch10_s0[h]+t_after+1)])[0]
                dev=(exp_ch[1]-choice[ind_ch10_s0[h]+t_after]) # if there is inference this number should be larger than 0 
                #ch_pre=np.mean(choice[ind_used])
                #dev=(ch_pre-choice[ind_ch10_s0[h]+t_after]) # if there is inference this number should be larger than 0  
                if stimulus[ind_ch10_s0[h]+t_after]==0: 
                    beha_tested_rhigh.append(dev)
                if stimulus[ind_ch10_s0[h]+t_after]==1:
                    beha_untested_rlow.append(dev)
                
            # Current high is Left and current low is Right. Previous low is Left and previous high is Right. First reward trial after context change is Right.
            # rhigh untested is left, rlow tested is right
            ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
            for h in range(len(ind_ch10_s1)):
                ind_pre=(np.arange(nback)-nback+ind_ch10_s1[h]+1)
                ind_used[ind_pre]=True
                #ind_used[np.isnan(rt)]=False
            for h in range(len(ind_ch10_s1)):
                cl=LogisticRegression(C=reg,class_weight='balanced')
                cl.fit(xx[ind_used],choice[ind_used])
                exp_ch=cl.predict_proba(xx[(ind_ch10_s1[h]+t_after):(ind_ch10_s1[h]+t_after+1)])[0]
                dev=(exp_ch[1]-choice[ind_ch10_s1[h]+t_after]) # if there is inference this number should be larger than 0 
                #ch_pre=np.mean(choice[ind_used])
                #dev=(ch_pre-choice[ind_ch10_s1[h]+t_after]) # if there is inference this number should be larger than 0  
                if stimulus[ind_ch10_s1[h]+t_after]==0: 
                    beha_untested_rhigh.append(dev)
                if stimulus[ind_ch10_s1[h]+t_after]==1:
                    beha_tested_rlow.append(dev)        

#             ##################################################
#             # Neuro
#             # Probability of Choice of classifier = Context for all possibilities: 01 0, 01 1, 10 0, 10 1
#             ind_train=ret_ind_train(coherence,ind_ch,t_back,t_forw)
#             cl=LogisticRegression(C=1/reg,class_weight='balanced')
#             cl.fit(firing_rate[ind_train],context[ind_train])
#             choice_cl=cl.predict(firing_rate)

        beha_te_unte[0,0,hh]=np.nanmean(beha_tested_rlow,axis=0)
        beha_te_unte[0,1,hh]=np.nanmean(beha_tested_rhigh,axis=0)
        beha_te_unte[1,0,hh]=np.nanmean(beha_untested_rlow,axis=0)
        beha_te_unte[1,1,hh]=np.nanmean(beha_untested_rhigh,axis=0)

    print (beha_te_unte)
    print (np.nanmean(beha_te_unte,axis=2),np.nanmean(beha_te_unte))
