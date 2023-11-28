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
import datetime

nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines: 
            if loc=='left':
                spine.set_position(('outward', 10))  # outward by 10 points
            if loc=='bottom':
                spine.set_position(('outward', 0))  # outward by 10 points
         #   spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

# Fira[0] general information
# Fira[1] behavioral information
# Fira[2] neuronal information. First index is trials, next is unknown but pick index 0 (always len 4), next is unit index (electrode), next is unit index within electrode, next is time stamps for spikes

def getField(data,name,extra=False):
    ind=np.where(data['FIRA'][0]['events'][0]==name)[0][0]
    var_pre=data['FIRA'][1][:,ind]
    if extra:
        var=nan*np.zeros(np.shape(var_pre))
        for i in range(len(var)):
            if type(var_pre[i])==int:
                var[i]=var_pre[i]
            else:
                var[i]=nan
    else:
        var=var_pre.copy()    
    return var

def trans_rew(x):
    rew=nan*np.zeros((len(x),2))
    for i in range(len(x)):
        rew[i]=x[i]
    return rew

# It works with both sorted and unsorted data
def getRasters(data,aligned,dic_time,index_nonan,threshold):
    pre_time=dic_time[0]
    post_time=dic_time[1]
    bin_size=dic_time[2]
    step_size=dic_time[3]
    #
    ind_event=np.where(data['FIRA'][0]['events'][0]==aligned)[0][0]
    t_aligned=data['FIRA'][1][:,ind_event][index_nonan]
    t_start=(t_aligned-pre_time)
    t_end=(t_aligned+post_time)
    steps=int((pre_time+post_time)/step_size)

    neu_total=len(data['FIRA'][0]['spikecd'])
    unit_id=data['FIRA'][0]['spikecd'].copy()
    unit_id[:,0]=(unit_id[:,0]-1)

    data_neural=data['FIRA'][2][index_nonan,0] # Careful with this zero. What does it mean?
    
    fr_mat_pre=nan*np.zeros((len(data_neural),neu_total,steps))
    for i in range(len(fr_mat_pre)): #loop trials
        for ii in range(len(fr_mat_pre[0])): #loop neurons
            sp=data_neural[i][unit_id[ii,0]][unit_id[ii,1]]
            if (type(sp)==int) or (type(sp)==float): #loop time steps
                sp=np.array([sp])
            for iii in range(steps):
                t_low=(t_start[i]+iii*step_size)
                t_high=(t_start[i]+iii*step_size+bin_size)
                fr_mat_pre[i,ii,iii]=len(sp[(sp>t_low)&(sp<t_high)])/(bin_size/1000) # Careful!

    mean_fr=np.mean(fr_mat_pre*(bin_size/1000),axis=(0,2))
    good_fr=(mean_fr>=threshold)
    fr_mat=fr_mat_pre[:,good_fr]
    return fr_mat

# If used with sorted data it'll produce an error
def getRasters_unsorted(data,aligned,dic_time,index_nonan,threshold):
    pre_time=dic_time[0]
    post_time=dic_time[1]
    bin_size=dic_time[2]
    step_size=dic_time[3]
    #
    ind_event=np.where(data['FIRA'][0]['events'][0]==aligned)[0][0]
    t_aligned=data['FIRA'][1][:,ind_event][index_nonan]
    t_start=(t_aligned-pre_time)
    t_end=(t_aligned+post_time)
    steps=int((pre_time+post_time)/step_size)

    neu_total=len(data['FIRA'][0]['spikecd'])
    data_neural=data['FIRA'][2][index_nonan,0] # Careful with this zero. What does it mean?
    
    fr_mat_pre=nan*np.zeros((len(data_neural),neu_total,steps))
    for i in range(len(fr_mat_pre)): #loop trials
        for ii in range(len(fr_mat_pre[0])): #loop units
            sp=data_neural[i][ii]
            if (type(sp)==int) or (type(sp)==float): 
                sp=np.array([sp])
            for iii in range(steps): #loop time steps
                t_low=(t_start[i]+iii*step_size)
                t_high=(t_start[i]+iii*step_size+bin_size)
                fr_mat_pre[i,ii,iii]=len(sp[(sp>t_low)&(sp<t_high)])/(bin_size/1000) # Careful!

    mean_fr=np.mean(fr_mat_pre*(bin_size/1000),axis=(0,2))
    good_fr=(mean_fr>=threshold)
    fr_mat=fr_mat_pre[:,good_fr]
    return fr_mat


def normalize_fr(fr):
    t_steps=len(fr[0,0])
    n_neu=len(fr[0])
    fr_norm=-31415*np.ones(np.shape(fr))
    for i in range(n_neu):
        for ii in range(t_steps):        
            rate_m=np.mean(fr[:,i,ii])
            rate_std=np.std(fr[:,i,ii])
            fr_norm[:,i,ii]=(fr[:,i,ii]-rate_m)/rate_std
    fr_norm[np.isnan(fr_norm)]=0
    return fr_norm

# Choice: Left 0, Right 1.
# Coherence: negative Left, positive  Right.
# Context: 0 Left more rewarded, Right more rewarded 1. 
def behavior(data,group_coh=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7])):
    right=1
    thres_diff=0.05
    #
    choice_pre=getField(data,'targ_cho',extra=True)
    index_nonan=~np.isnan(choice_pre)
    choice=choice_pre[index_nonan]
    choice[choice==2]=0 # 1 Right, 0 Left
    stimulus=np.array(getField(data,'targ_cor')[index_nonan],dtype=np.int16)
    stimulus[stimulus==2]=0 # 1 Right, 0 Left
    coherence=np.array(getField(data,'dot_coh')[index_nonan],dtype=np.float64)/1000
    difficulty=np.zeros(len(stimulus))
    difficulty[coherence>thres_diff]=1
    coh_signed=coherence.copy()
    coh_signed[stimulus!=right]=-coh_signed[stimulus!=right] # Negative
    coh_signed_uq=np.unique(coh_signed)
    #
    coh_resol=nan*np.zeros(len(coherence)) #reduce the resolution of coherences (group them)
    for i in range(len(coh_signed_uq)):
        ind=np.where(coh_signed==coh_signed_uq[i])[0]
        coh_resol[ind]=group_coh[i]
    #
    coh_log=nan*np.zeros(len(coh_signed))
    coh_log[coh_signed>0]=-np.log10(coh_signed[coh_signed>0])
    coh_log[coh_signed<0]=np.log10(-coh_signed[coh_signed<0])
    coh_log[coh_signed==0]=0
    coh_num=nan*np.zeros(len(coherence))
    for i in range(len(coh_signed_uq)):
        ind=np.where(coh_signed==coh_signed_uq[i])[0]
        coh_num[ind]=i    
    reward=np.array(choice==stimulus,dtype=np.int16)
    rew_pulse=trans_rew(getField(data,'rew_pulse_dur'))[index_nonan]
    #Context 1 higher than 2: 1. Context left more reward 1, context right more reward, 0. 
    context=np.array((rew_pulse[:,0]-rew_pulse[:,1])>0,dtype=np.int16) 
    response_edf=np.array(getField(data,'response_edf')[index_nonan],dtype=np.float64)
    dots_on=np.array(getField(data,'dots_on')[index_nonan],dtype=np.float64)
    rt=(response_edf-dots_on)
    change_ctx=(context[1:]-context[0:-1])
    
    dic={}
    dic['index_nonan']=index_nonan
    dic['choice']=choice
    dic['stimulus']=stimulus
    dic['coherence']=coherence
    dic['coh_log']=coh_log
    dic['coh_num']=coh_num
    dic['difficulty']=difficulty
    dic['coherence_uq']=np.unique(coherence)
    dic['coherence_signed']=coh_signed
    dic['coherence_signed_uq']=np.unique(coh_signed)
    dic['coh_resol']=coh_resol
    dic['context']=context
    dic['reward']=reward
    dic['response_edf']=response_edf
    dic['reaction_time']=rt
    dic['change_ctx']=change_ctx
    #
    dic['stimulus_0']=stimulus[1:]
    dic['choice_0']=choice[1:]
    dic['context_0']=context[1:]
    dic['reward_0']=reward[1:]
    dic['difficulty_0']=difficulty[1:]
    dic['stimulus_m1']=stimulus[0:-1]
    dic['choice_m1']=choice[0:-1]
    dic['context_m1']=context[0:-1]
    dic['reward_m1']=reward[0:-1]
    dic['difficulty_m1']=difficulty[0:-1]
    return dic

def classifier(neural,clase,n_cv,reg):
    perf=nan*np.zeros((n_cv,2))
    cv=StratifiedKFold(n_splits=n_cv)
    g=-1
    for train_index, test_index in cv.split(neural,clase):
        g=(g+1)
        cl=LogisticRegression(C=1/reg,class_weight='balanced')
        #cl=LinearSVC(C=1/reg,class_weight='balanced')
        cl.fit(neural[train_index],clase[train_index])
        perf[g,0]=cl.score(neural[train_index],clase[train_index])
        perf[g,1]=cl.score(neural[test_index],clase[test_index])
    return np.mean(perf,axis=0)

def pseudopopulation_1(abs_path,files,quant,talig,dic_time,steps,thres,nt,n_rand,perc_tr,trials_back):
    pseudo_tr_pre=nan*np.zeros((steps,n_rand,2*nt,1200)) 
    pseudo_te_pre=nan*np.zeros((steps,n_rand,2*nt,1200))
    neu_t=0
    for kk in range(len(files)):
        print (files[kk])
        data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
        beha=behavior(data)
        index_nonan=beha['index_nonan']
        len_trials=len(index_nonan)
        
        firing_rate_pre=getRasters(data,talig,dic_time,index_nonan,thres)
        n_neu=len(firing_rate_pre[0])
        if trials_back==0:
            variable=beha[quant]
            firing_rate=firing_rate_pre.copy()
        else:
            variable=beha[quant][0:(-trials_back)]
            firing_rate=firing_rate_pre[trials_back:]
        ind1=np.where(variable==1)[0]
        ind0=np.where(variable==0)[0]
        nt1=int(len(ind1)*perc_tr)  
        nt0=int(len(ind0)*perc_tr)
        
        for i in range(steps):
            for ii in range(n_rand):
                ind1_p=np.random.permutation(ind1)
                ind0_p=np.random.permutation(ind0)
                pseudo_tr_pre[i,ii,0:nt,neu_t:(neu_t+n_neu)]=firing_rate[np.random.choice(ind1_p[0:nt1],nt,replace=True)][:,:,i]
                pseudo_tr_pre[i,ii,nt:,neu_t:(neu_t+n_neu)]=firing_rate[np.random.choice(ind0_p[0:nt0],nt,replace=True)][:,:,i]
                pseudo_te_pre[i,ii,0:nt,neu_t:(neu_t+n_neu)]=firing_rate[np.random.choice(ind1_p[nt1:],nt,replace=True)][:,:,i]
                pseudo_te_pre[i,ii,nt:,neu_t:(neu_t+n_neu)]=firing_rate[np.random.choice(ind0_p[nt0:],nt,replace=True)][:,:,i]
        neu_t=(neu_t+n_neu)
        
    pseudo_tr=pseudo_tr_pre[:,:,:,0:neu_t]
    pseudo_te=pseudo_te_pre[:,:,:,0:neu_t]
    clase=np.zeros(2*nt)
    clase[0:nt]=1    
    return pseudo_tr,pseudo_te,clase

def pseudopop_coherence(abs_path,files,talig,dic_time,steps,thres,nt,n_rand,perc_tr,signed):
    if signed==True:
        n_coh=15
        quant='coherence_signed'
    if signed==False:
        n_coh=8
        quant='coherence'
    pseudo_tr_pre=nan*np.zeros((steps,n_rand,n_coh*nt,1200)) 
    pseudo_te_pre=nan*np.zeros((steps,n_rand,n_coh*nt,1200))
    pseudo_all_pre=nan*np.zeros((steps,n_rand,n_coh*nt,1200))
    clase=nan*np.zeros(n_coh*nt)
    clase_coh=nan*np.zeros(n_coh*nt)
    neu_t=0
    for kk in range(len(files)):
        print (files[kk])
        data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
        beha=behavior(data)
        index_nonan=beha['index_nonan']
        reward=beha['reward'] #Cuidado!!!
        ind_correct=np.where(reward==1)[0]
        
        firing_rate=getRasters(data,talig,dic_time,index_nonan,thres)[ind_correct] #Cuidado!
        n_neu=len(firing_rate[0])
        
        coherence=beha[quant][ind_correct]#Cuidado!
        coh_uq=np.unique(coherence)
        
        for j in range(n_coh):
            ind_coh=np.where(coherence==coh_uq[j])[0]
            nt_coh=int(len(ind_coh)*perc_tr)
            clase[j*nt:(j+1)*nt]=j
            clase_coh[j*nt:(j+1)*nt]=coh_uq[j]
            for i in range(steps):
                for ii in range(n_rand):
                    ind_coh_p=np.random.permutation(ind_coh)
                    pseudo_all_pre[i,ii,j*nt:(j+1)*nt,neu_t:(neu_t+n_neu)]=firing_rate[np.random.choice(ind_coh_p,nt,replace=True)][:,:,i]
                    pseudo_tr_pre[i,ii,j*nt:(j+1)*nt,neu_t:(neu_t+n_neu)]=firing_rate[np.random.choice(ind_coh_p[0:nt_coh],nt,replace=True)][:,:,i]
                    pseudo_te_pre[i,ii,j*nt:(j+1)*nt,neu_t:(neu_t+n_neu)]=firing_rate[np.random.choice(ind_coh_p[nt_coh:],nt,replace=True)][:,:,i]
        neu_t=(neu_t+n_neu)

    pseudo_all=pseudo_all_pre[:,:,:,0:neu_t]
    pseudo_tr=pseudo_tr_pre[:,:,:,0:neu_t]
    pseudo_te=pseudo_te_pre[:,:,:,0:neu_t]
    dic={}
    dic['pseudo_all']=pseudo_all
    dic['pseudo_tr']=pseudo_tr
    dic['pseudo_te']=pseudo_te
    dic['clase']=clase
    dic['clase_coh']=clase_coh
    return dic

def pseudopop_coherence_context(abs_path,files,talig,dic_time,steps,thres,nt,n_rand,perc_tr,signed,tpre_sacc,group_coh):
    if signed==True:
        n_coh=15
        quant='coherence_signed'
    if signed==False:
        n_coh=8
        quant='coherence'
    pseudo_all_pre=nan*np.zeros((steps,n_rand,2*n_coh*nt,1200))
    pseudo_tr_pre=nan*np.zeros((steps,n_rand,2*n_coh*nt,1200)) 
    pseudo_te_pre=nan*np.zeros((steps,n_rand,2*n_coh*nt,1200))
    neu_t=0
    time_w=dic_time[-2]
    step_t=dic_time[-1]
    for kk in range(len(files)):
        print (files[kk])
        data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
        beha=behavior(data,group_coh)
        index_nonan=beha['index_nonan']
        len_trials=len(index_nonan)
        rt=beha['reaction_time']
        
        firing_rate_pre=getRasters(data,talig,dic_time,index_nonan,thres)
        firing_rate=normalize_fr(firing_rate_pre)
        n_neu=len(firing_rate[0])

        context=beha['context']
        coherence=beha[quant]
        coh_uq=np.unique(coherence)
        choice=beha['choice']

        for i in range(steps):
            if talig=='dots_on':
                max_t=(-dic_time[0]+i*step_t+time_w+100) # -initial + j*step_size + bin_size + 100
            if talig=='response_edf':
                max_t=0

            for k in range(2): # loop contexts
                for j in range(n_coh): #loop coherences
                    ind_coh_ctx=np.where((coherence==coh_uq[j])&(context==k)&(rt>max_t))[0]
                    nt_coh_ctx=int(len(ind_coh_ctx)*perc_tr)
                    for ii in range(n_rand):
                        ind_coh_ctx_p=np.random.permutation(ind_coh_ctx)
                        dow=(k*n_coh+j)*nt
                        up=(k*n_coh+j+1)*nt
                        try:
                            pseudo_all_pre[i,ii,dow:up,neu_t:(neu_t+n_neu)]=firing_rate[np.random.choice(ind_coh_ctx_p,nt,replace=True)][:,:,i]
                            pseudo_tr_pre[i,ii,dow:up,neu_t:(neu_t+n_neu)]=firing_rate[np.random.choice(ind_coh_ctx_p[0:nt_coh_ctx],nt,replace=True)][:,:,i]
                            pseudo_te_pre[i,ii,dow:up,neu_t:(neu_t+n_neu)]=firing_rate[np.random.choice(ind_coh_ctx_p[nt_coh_ctx:],nt,replace=True)][:,:,i]
                        except:
                            #print ('Error t %i coh %i '%(i,j))
                            None
        neu_t=(neu_t+n_neu)

    clase_all=nan*np.zeros(2*n_coh*nt)
    clase_coh=nan*np.zeros(2*n_coh*nt)
    clase_ctx=nan*np.zeros(2*n_coh*nt)
    for k in range(2):
        for j in range(n_coh):
            clase_all[(k*n_coh+j)*nt:(k*n_coh+j+1)*nt]=(k*n_coh+j)
            clase_coh[(k*n_coh+j)*nt:(k*n_coh+j+1)*nt]=j
            clase_ctx[(k*n_coh+j)*nt:(k*n_coh+j+1)*nt]=k

    pseudo_all=pseudo_all_pre[:,:,:,0:neu_t]
    pseudo_tr=pseudo_tr_pre[:,:,:,0:neu_t]
    pseudo_te=pseudo_te_pre[:,:,:,0:neu_t]
    dic={}
    dic['pseudo_all']=pseudo_all
    dic['pseudo_tr']=pseudo_tr
    dic['pseudo_te']=pseudo_te
    dic['clase_all']=clase_all
    dic['clase_coh']=clase_coh
    dic['clase_ctx']=clase_ctx
    return dic

def pseudopop_coherence_context_correct(abs_path,files,talig,dic_time,steps,thres,nt,n_rand,perc_tr,signed,tpre_sacc,group_coh,shuff,learning):
    if signed==True:
        n_coh=15
        quant='coherence_signed'
    if signed==False:
        n_coh=8
        quant='coherence'
    pseudo_all_pre=nan*np.zeros((steps,n_rand,2*n_coh*nt,400*len(files)))
    pseudo_tr_pre=nan*np.zeros((steps,n_rand,2*n_coh*nt,400*len(files))) 
    pseudo_te_pre=nan*np.zeros((steps,n_rand,2*n_coh*nt,400*len(files)))

    #context_new=nan*np.zeros((n_rand,2*n_coh*nt))
    #stimulus_new=nan*np.zeros((n_rand,2*n_coh*nt))
    #choice_new=nan*np.zeros((n_rand,2*n_coh*nt))
    
    neu_t=0
    time_w=dic_time[-2]
    step_t=dic_time[-1]
    for kk in range(len(files)):
        print (files[kk])
        data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
        beha=behavior(data,group_coh)
        index_nonan=beha['index_nonan']
        len_trials=len(index_nonan)
        reward=beha['reward'] 
        ind_correct=np.where(reward==1)[0]
        rt=beha['reaction_time'][ind_correct] 

        if learning==True:
            firing_rate_pre=getRasters_unsorted(data,talig,dic_time,index_nonan,thres) 
        if learning==False:
            firing_rate_pre=getRasters(data,talig,dic_time,index_nonan,thres)
        
        firing_rate=normalize_fr(firing_rate_pre)[ind_correct] 
        n_neu=len(firing_rate[0])

        context=beha['context'][ind_correct]
        coherence=beha[quant][ind_correct]
        coh_uq=np.unique(coherence)
        choice=beha['choice'][ind_correct]
        if shuff:
            context=permutation(context)
            coherence=permutation(coherence)
            choice=permutation(choice)

        for i in range(steps):
            if talig=='dots_on':
                max_t=(-dic_time[0]+i*step_t+time_w+tpre_sacc) # -initial + j*step_size + bin_size + 100
            if talig=='response_edf':
                max_t=0

            for k in range(2):
                for j in range(n_coh):
                    ind_coh_ctx=np.where((coherence==coh_uq[j])&(context==k)&(rt>max_t))[0]
                    #print (i,k,j,len(ind_coh_ctx))
                    nt_coh_ctx=int(len(ind_coh_ctx)*perc_tr)
                    for ii in range(n_rand):
                        ind_coh_ctx_p=np.random.permutation(ind_coh_ctx)
                        dow=(k*n_coh+j)*nt
                        up=(k*n_coh+j+1)*nt
                        try:
                            ind_coh_ctx_p_ch_all=np.random.choice(ind_coh_ctx_p,nt,replace=True)
                            ind_coh_ctx_p_ch_tr=np.random.choice(ind_coh_ctx_p[0:nt_coh_ctx],nt,replace=True)
                            ind_coh_ctx_p_ch_te=np.random.choice(ind_coh_ctx_p[nt_coh_ctx:],nt,replace=True)
                            pseudo_all_pre[i,ii,dow:up,neu_t:(neu_t+n_neu)]=firing_rate[ind_coh_ctx_p_ch_all][:,:,i]
                            pseudo_tr_pre[i,ii,dow:up,neu_t:(neu_t+n_neu)]=firing_rate[ind_coh_ctx_p_ch_tr][:,:,i]
                            pseudo_te_pre[i,ii,dow:up,neu_t:(neu_t+n_neu)]=firing_rate[ind_coh_ctx_p_ch_te][:,:,i]
                        except:
                            None
                            #print ('Error %i %i %i %i '%(i,k,j,ii))
        neu_t=(neu_t+n_neu)

    clase_all=nan*np.zeros(2*n_coh*nt)
    clase_coh=nan*np.zeros(2*n_coh*nt)
    clase_ctx=nan*np.zeros(2*n_coh*nt)
    for k in range(2):
        for j in range(n_coh):
            clase_all[(k*n_coh+j)*nt:(k*n_coh+j+1)*nt]=(k*n_coh+j)
            clase_coh[(k*n_coh+j)*nt:(k*n_coh+j+1)*nt]=j
            clase_ctx[(k*n_coh+j)*nt:(k*n_coh+j+1)*nt]=k

    pseudo_all=pseudo_all_pre[:,:,:,0:neu_t]
    pseudo_tr=pseudo_tr_pre[:,:,:,0:neu_t]
    pseudo_te=pseudo_te_pre[:,:,:,0:neu_t]
    dic={}
    dic['pseudo_all']=pseudo_all
    dic['pseudo_tr']=pseudo_tr
    dic['pseudo_te']=pseudo_te
    dic['clase_all']=clase_all
    dic['clase_coh']=clase_coh
    dic['clase_ctx']=clase_ctx
    #dic['context']=context_new
    #dic['stimulus']=stimulus_new
    #dic['choice']=choice_new
    return dic

def pseudopopulation_nvar(abs_path,files,quant,talig,dic_time,steps,thres,nt,n_rand,perc_tr,tback):
    nvar=len(quant)
    cond=int(2**nvar)
    pseudo_all_pre=nan*np.zeros((steps,n_rand,cond*nt,1200)) #Careful here
    pseudo_tr_pre=nan*np.zeros((steps,n_rand,cond*nt,1200)) #Careful here
    pseudo_te_pre=nan*np.zeros((steps,n_rand,cond*nt,1200))
    neu_t=0
    for kk in range(len(files)):
        print (files[kk])
        data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
        beha=behavior(data)
        index_nonan=beha['index_nonan']
        rt=beha['reaction_time'][tback:]
        
        firing_rate=getRasters(data,talig,dic_time,index_nonan,thres)[tback:]
        n_neu=len(firing_rate[0])

        for i in range(steps):
            if talig=='dots_on':
                max_t=(-dic_time[0]+i*dic_time[3]+dic_time[2]+100) # -initial + j*step_size + bin_size + 100
            if talig=='response_edf':
                max_t=0
            
            ind_all=[]
            len_ind_all=[]
            if nvar==2:
                c_uq=np.array([[1,1],[1,0],[0,1],[0,0]])
                for j in range(len(c_uq)):
                    ind_all.append(np.where((beha[quant[0]]==c_uq[j,0])&(beha[quant[1]]==c_uq[j,1])&(rt>max_t))[0])
                    len_ind_all.append(len(ind_all[-1]))
                
            if nvar==3:
                c_uq=np.array([[1,1,1],[1,1,0],[1,0,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]])
                for j in range(len(c_uq)):
                    ind_all.append(np.where((beha[quant[0]]==c_uq[j,0])&(beha[quant[1]]==c_uq[j,1])&(beha[quant[2]]==c_uq[j,2])&(rt>max_t))[0])
                    len_ind_all.append(len(ind_all[-1]))
                
            if nvar==4:
                c_uq=np.array([[1,1,1,1],[1,1,1,0],[1,1,0,1],[1,1,0,0],[1,0,1,1],[1,0,1,0],[1,0,0,1],[1,0,0,0],[0,1,1,1],[0,1,1,0],[0,1,0,1],[0,1,0,0],[0,0,1,1],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
                for j in range(len(c_uq)):
                    ind_all.append(np.where((beha[quant[0]]==c_uq[j,0])&(beha[quant[1]]==c_uq[j,1])&(beha[quant[2]]==c_uq[j,2])&(beha[quant[3]]==c_uq[j,3])&(rt>max_t))[0])
                    len_ind_all.append(len(ind_all[-1]))

            if nvar==5:
                c_uq=np.array([[1,1,1,1,1],[1,1,1,1,0],[1,1,1,0,1],[1,1,1,0,0],[1,1,0,1,1],[1,1,0,1,0],[1,1,0,0,1],[1,1,0,0,0],[1,0,1,1,1],[1,0,1,1,0],[1,0,1,0,1],[1,0,1,0,0],[1,0,0,1,1],[1,0,0,1,0],[1,0,0,0,1],[1,0,0,0,0],[0,1,1,1,1],[0,1,1,1,0],[0,1,1,0,1],[0,1,1,0,0],[0,1,0,1,1],[0,1,0,1,0],[0,1,0,0,1],[0,1,0,0,0],[0,0,1,1,1],[0,0,1,1,0],[0,0,1,0,1],[0,0,1,0,0],[0,0,0,1,1],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,0]])
                for j in range(len(c_uq)):
                    ind_all.append(np.where((beha[quant[0]]==c_uq[j,0])&(beha[quant[1]]==c_uq[j,1])&(beha[quant[2]]==c_uq[j,2])&(beha[quant[3]]==c_uq[j,3])&(beha[quant[4]]==c_uq[j,4]))[0])
                    len_ind_all.append(len(ind_all[-1]))

            print (i,len_ind_all)
        
            for ii in range(n_rand):
                for iii in range(cond):
                    try:
                        pseudo_all_pre[i,ii,iii*nt:(iii+1)*nt,neu_t:(neu_t+n_neu)]=firing_rate[np.random.choice(ind_all[iii],nt,replace=True)][:,:,i]
                        ind_p=np.random.permutation(ind_all[iii])
                        ntu=int(len(ind_p)*perc_tr)
                        pseudo_tr_pre[i,ii,iii*nt:(iii+1)*nt,neu_t:(neu_t+n_neu)]=firing_rate[np.random.choice(ind_p[0:ntu],nt,replace=True)][:,:,i]
                        pseudo_te_pre[i,ii,iii*nt:(iii+1)*nt,neu_t:(neu_t+n_neu)]=firing_rate[np.random.choice(ind_p[ntu:],nt,replace=True)][:,:,i]
                    except:
                        print (i,ii,iii)
        neu_t=(neu_t+n_neu)
    pseudo_all=pseudo_all_pre[:,:,:,0:neu_t]
    pseudo_tr=pseudo_tr_pre[:,:,:,0:neu_t]
    pseudo_te=pseudo_te_pre[:,:,:,0:neu_t]

    clase_all=nan*np.zeros(cond*nt)
    clase_var=nan*np.zeros((cond*nt,nvar))
    for i in range(cond):
        clase_all[i*nt:(i+1)*nt]=i
        clase_var[i*nt:(i+1)*nt]=c_uq[i]
    
    dic={}
    dic['pseudo_all']=pseudo_all
    dic['pseudo_tr']=pseudo_tr
    dic['pseudo_te']=pseudo_te
    dic['clase_all']=clase_all
    dic['clase_var']=clase_var
    return dic


def dim_pseudo_3(feat_tr,feat_te,clase,n_dim,reg):
    dim=nan*np.zeros(n_dim)
    #word=np.zeros(8)
    #word[0:4]=1
    i=0
    while i<n_dim:
        #word_sh=np.random.permutation(word)
        try:
            word_sh=np.array(np.random.normal(0,1,8)>=0,dtype=np.int16)
            clase_d=nan*np.zeros(len(clase))
            for ii in range(8):
                clase_d[clase==ii]=word_sh[ii]
            cl=LogisticRegression(C=1/reg,class_weight='balanced')
            cl.fit(feat_tr,clase_d)
            dim[i]=cl.score(feat_te,clase_d)
            i=(i+1)
        except:
            None
    return np.mean(dim)

def dim_pseudo_2(feat_tr,feat_te,clase,reg):
    tasks=[[1,1,0,0],[1,0,1,0],[0,1,1,0]]
    perf_tasks=nan*np.zeros(len(tasks))
    for i in range(len(tasks)):
        clase_d=nan*np.zeros(len(clase))
        for ii in range(4):
            clase_d[clase==ii]=tasks[i][ii]
        cl=LogisticRegression(C=1/reg,class_weight='balanced')
        cl.fit(feat_tr,clase_d)
        perf_tasks[i]=cl.score(feat_te,clase_d)
    return perf_tasks

def abstraction_2D(feat_decod,clase_all,reg):
    dichotomies=np.array([[0,0,1,1],[0,1,0,1]])
    train_dich=np.array([[[0,2],[1,3]],[[0,1],[2,3]]])
    test_dich=np.array([[[1,3],[0,2]],[[2,3],[0,1]]])
    
    perf=nan*np.zeros((len(dichotomies),len(train_dich[0]),2))
    for k in range(len(dichotomies)): #Loop on "dichotomies"
      for kk in range(len(train_dich[0])): #Loop on ways to train this particular "dichotomy"
         ind_train=np.where((clase_all==train_dich[k][kk][0])|(clase_all==train_dich[k][kk][1]))[0]
         ind_test=np.where((clase_all==test_dich[k][kk][0])|(clase_all==test_dich[k][kk][1]))[0]

         task=nan*np.zeros(len(clase_all))
         for i in range(4):
             ind_task=(clase_all==i)
             task[ind_task]=dichotomies[k][i]

         supp=LogisticRegression(C=1/reg,class_weight='balanced')
         mod=supp.fit(feat_decod[ind_train],task[ind_train])
         perf[k,kk,0]=supp.score(feat_decod[ind_train],task[ind_train])
         perf[k,kk,1]=supp.score(feat_decod[ind_test],task[ind_test])
    return perf


def abstraction_3D(feat_decod,clase_all):
    dichotomies=np.array([[0,0,0,0,1,1,1,1],
                          [0,0,1,1,0,0,1,1],
                          [0,1,0,1,0,1,0,1]])
    
    train_dich=np.array([[[0,2,3,4,6,7],[1,2,3,5,6,7],[0,1,3,4,5,7],[0,1,2,4,5,6]],
                         [[0,1,2,3,4,6],[0,2,4,5,6,7],[1,3,4,5,6,7],[0,1,2,3,5,7]],
                         [[2,3,4,5,6,7],[0,1,4,5,6,7],[0,1,2,3,6,7],[0,1,2,3,4,5]]])
                          
    test_dich=np.array([[[1,5],[0,4],[2,6],[3,7]],
                        [[5,7],[1,3],[0,2],[4,6]],
                        [[0,1],[2,3],[4,5],[6,7]]])
    
    perf=nan*np.zeros((len(dichotomies),len(train_dich[0]),2))
    for k in range(len(dichotomies)): #Loop on "dichotomies"
      for kk in range(len(train_dich[0])): #Loop on ways to train this particular "dichotomy"
         ind_train=np.where((clase_all==train_dich[k][kk][0])|(clase_all==train_dich[k][kk][1])|(clase_all==train_dich[k][kk][2])|(clase_all==train_dich[k][kk][3])|(clase_all==train_dich[k][kk][4])|(clase_all==train_dich[k][kk][5]))[0]
         ind_test=np.where((clase_all==test_dich[k][kk][0])|(clase_all==test_dich[k][kk][1]))[0]

         task=nan*np.zeros(len(clase_all))
         for i in range(8):
             ind_task=(clase_all==i)
             task[ind_task]=dichotomies[k][i]

         supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
         mod=supp.fit(feat_decod[ind_train],task[ind_train])
         perf[k,kk,0]=supp.score(feat_decod[ind_train],task[ind_train])
         perf[k,kk,1]=supp.score(feat_decod[ind_test],task[ind_test])
    return perf

def order_files(files_pre):
    tag_vec=[i[1:5]+'-'+i[5:7]+'-'+i[7:9] for i in files_pre]
    dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in tag_vec]
    files=files_pre[np.argsort(dates)]
    return files
