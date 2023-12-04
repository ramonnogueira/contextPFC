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

# In this script we evaluate generalization throughout learning.
# Behavior: proabability that choice = context after context switch
# Neural: decoding of context right after context switch. Classifier is trained on all the rest of trials (no context switch)

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


def order_files(x):
    ord_pre=[]
    for i in range(len(x)):
        ord_pre.append(x[i][1:9])
    ord_pre=np.array(ord_pre)
    order=np.argsort(ord_pre)
    return order

# Best for behavior
def func_p(x,a,b,c):
    return np.exp(-a*x)*b+c

def func_n(x,a,b,c):
    return -np.exp(-a*x)*b+c

def norm_quant_coh(quant,coherence):
    coh_uq=np.unique(coherence)
    quant_rel_coh=nan*np.zeros(len(quant))
    for i in range(len(coh_uq)):
        #print (coh_uq[i])
        ind=(coherence==coh_uq[i])
        rt_coh=np.nanmean(quant[ind])
        rt_std=np.nanstd(quant[ind])
        quant_rel_coh[ind]=(quant[ind]-rt_coh)/rt_std
        #plt.hist(rt[ind])
        #plt.axvline(rt_coh)
        #plt.show()
    return quant_rel_coh

def calculate_ind_ch_corr(ind_ch,reward):
    n_forw=7
    n_ch=len(ind_ch)
    ind_ch_corr=np.zeros(n_ch)
    for i in range(n_ch):
        aa=(np.arange(n_forw)+ind_ch[i])
        bb_pre=aa*reward[aa]
        bb=bb_pre[bb_pre>0]
        ind_ch_corr[i]=bb[0]
    return np.array(ind_ch_corr,dtype=np.int16)

# This function returns the indices for trials where:
# - context change from 0 to 1 correct stimulus 0
# - context change from 0 to 1 correct stimulus 1
# - context change from 1 to 0 correct stimulus 0
# - context change from 1 to 0 correct stimulus 1 
def calculate_ind_ch_corr2(ind_ch01,ind_ch10,reward,stimulus):
    n_forw=7

    # Context change from 0 to 1
    ind_ch01_s0=[]
    ind_ch01_s1=[]
    n_ch01=len(ind_ch01)
    for i in range(n_ch01):
        aa=(np.arange(n_forw)+ind_ch01[i])
        bb_pre=aa*reward[aa]
        bb=bb_pre[bb_pre>0]
        if stimulus[bb[0]]==0:
            ind_ch01_s0.append(bb[0])
        if stimulus[bb[0]]==1:
            ind_ch01_s1.append(bb[0])

    # Context change from 1 to 0
    ind_ch10_s0=[]
    ind_ch10_s1=[]
    n_ch10=len(ind_ch10)
    for i in range(n_ch10):
        aa=(np.arange(n_forw)+ind_ch10[i])
        bb_pre=aa*reward[aa]
        bb=bb_pre[bb_pre>0]
        if stimulus[bb[0]]==0:
            ind_ch10_s0.append(bb[0])
        if stimulus[bb[0]]==1:
            ind_ch10_s1.append(bb[0])
    return np.array(ind_ch01_s0,dtype=np.int16),np.array(ind_ch01_s1,dtype=np.int16),np.array(ind_ch10_s0,dtype=np.int16),np.array(ind_ch10_s1,dtype=np.int16)

def func_eval(index,t_back,t_forw,stimulus,rt):#,new_ctx):
    pp=nan*np.zeros((2,t_forw+t_back))
    for j in range(t_back):
        indj=(index-t_back+j)
        try:
            pp[int(stimulus[indj]),j]=rt[indj]
        except:
            None
    for j in range(t_forw):
        indj=(index+j)
        try:
            pp[int(stimulus[indj]),j+t_back]=rt[indj]
        except:
            None
    return pp

def proj_dist(wei,wei0,fr):
    return abs(np.dot(fr,wei.T)+wei0)

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

def fr_rt_nan(reaction_time,firing_rate,tt,tw,time_extra):
    fr_nan=firing_rate.copy()
    rt_max=(reaction_time-time_extra)
    for i in range(len(tt)):
        ind_nan=(rt_max<(1000*tt[i]+tw))
        fr_nan[ind_nan,:,i]=nan
    return fr_nan

def fit_plot(xx,yy,t_back,t_forw,maxfev,method,bounds,p0,sign):
    if sign==1:
        popt,pcov=curve_fit(func_p,xx[(t_back+1):],yy[(t_back+1):],nan_policy='omit',maxfev=maxfev,bounds=bounds,p0=p0,method=method)
        fit_func=func_p(xx[(t_back+1):],popt[0],popt[1],popt[2])
    if sign==-1:
        popt,pcov=curve_fit(func_n,xx[(t_back+1):],yy[(t_back+1):],nan_policy='omit',maxfev=maxfev,bounds=bounds,p0=p0,method=method)
        fit_func=func_n(xx[(t_back+1):],popt[0],popt[1],popt[2])
    print ('Fit ',popt)
    print (pcov)
    # plt.scatter(xx,yy,color='blue',s=1)
    # plt.plot(xx[(t_back+1):],fit_func,color='black')
    # plt.axvline(0,color='black',linestyle='--')
    # plt.plot(xx,0*np.ones(len(xx)),color='black',linestyle='--')
    # plt.ylim([-3,3])
    # plt.show()
    return fit_func
  
#################################################

# Function 2 for both. Bounds and p0 are important. 
# Niels: t_back 20, t_forw 80, time window 200ms. No kernel. Groups of 1 session
# Galileo: t_back 20, t_forw 80, time window 300ms. No kernel. Groups of 3 sessions

monkey='Galileo'
t_back=30
t_forw=90
delta_type='fit'

time_extra=50

talig='dots_on' #'response_edf' #dots_on
dic_time=np.array([-200,400,200,200])# time pre, time post, bin size, step size (time pre always positive) #For Galileo use timepost 800 or 1000. For Niels use
steps=int((dic_time[0]+dic_time[1])/dic_time[3])
tt=np.linspace(-dic_time[0]/1000,dic_time[1]/1000,steps,endpoint=False)

thres=0
reg=1e0
maxfev=100000
method='dogbox'
bounds=([0,0,-1],[1000,10,1])
p0=(0.05,0.5,0.1)

xx=np.arange(t_back+t_forw)-t_back

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])
if monkey=='Niels':
    files_groups=[[8,9],[9,10],[10,11],[11,12]]
   
if monkey=='Galileo':
    #files_groups=[[0,10],[10,20],[20,30]]
    #files_groups=[[0,5],[5,10],[10,15],[15,20],[20,25],[25,30]]
    files_groups=[[0,3],[3,6],[6,9],[9,12],[12,15],[15,18],[18,21],[21,24],[24,27],[27,30]]
    #files_groups=[[0,2],[2,4],[4,6],[6,8],[8,10],[10,12],[12,14],[14,16],[16,18],[18,20],[20,22],[22,24],[24,26],[26,28],[28,30]]
    #files_groups=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],[16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[22,23],[23,24],[24,25],[25,26],[26,27],[27,28],[28,29],[29,30]]

abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/unsorted/%s/'%(monkey) 
files_pre=np.array(os.listdir(abs_path))
order=order_files(files_pre)
files_all=np.array(files_pre[order])
print (files_all)

fit_beha=nan*np.zeros((2,2,len(files_groups),t_back+t_forw))
y0_beha=nan*np.zeros((2,2,len(files_groups)))
beha_te_unte=nan*np.zeros((2,2,len(files_groups),t_back+t_forw))
fit_neuro=nan*np.zeros((2,2,len(files_groups),t_back+t_forw))
y0_neuro=nan*np.zeros((2,2,len(files_groups)))
neuro_te_unte=nan*np.zeros((2,2,len(files_groups),t_back+t_forw))
for hh in range(len(files_groups)):
    beha_tested_rlow=[]
    beha_tested_rhigh=[]
    beha_untested_rlow=[]
    beha_untested_rhigh=[]
    neuro_tested_rlow=[]
    neuro_tested_rhigh=[]
    neuro_untested_rlow=[]
    neuro_untested_rhigh=[]
    files=files_all[files_groups[hh][0]:files_groups[hh][1]]
    print (files)
    for kk in range(len(files)):
        print (files[kk])
        #Load data
        data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
        beha=miscellaneous.behavior(data)
        index_nonan=beha['index_nonan']
        # We discard first trial of session because we are interested in context changes
        stimulus=beha['stimulus'][1:]
        choice=beha['choice'][1:]
        coherence=beha['coherence_signed'][1:]
        coh_uq=np.unique(coherence)
        reward=beha['reward'][1:]
        rt_pre=beha['reaction_time'][1:]
        rt=norm_quant_coh(rt_pre,coherence)
        context_pre=beha['context']
        ctx_ch=(context_pre[1:]-context_pre[0:-1])
        context=context_pre[1:] #FIX THIS. WE NEED A VARIABLE "SUBJECTIVE CONTEXT"
        ind_ch_pre=np.where(abs(ctx_ch)==1)[0] # ind_ch_pre index where there is a context change
        ind_ch=calculate_ind_ch_corr(ind_ch_pre,reward) # ind_ch first correct trial after context change (otherwise animal doesn't know there was a change)
        indch_ct01_pre=np.where(ctx_ch==1)[0]
        indch_ct10_pre=np.where(ctx_ch==-1)[0]
        ind_ch01_s0,ind_ch01_s1,ind_ch10_s0,ind_ch10_s1=calculate_ind_ch_corr2(indch_ct01_pre,indch_ct10_pre,reward,stimulus)
        
        # Careful contamination saccades!
        firing_rate_pre1=miscellaneous.getRasters_unsorted(data,talig,dic_time,index_nonan,threshold=thres)
        print (np.shape(firing_rate_pre1))
        num_neu=len(firing_rate_pre1[0])
        steps=len(firing_rate_pre1[0,0])
        firing_rate_pre2=miscellaneous.normalize_fr(firing_rate_pre1)[1:]
        fr_nan=np.reshape(fr_rt_nan(rt_pre,firing_rate_pre2,tt,dic_time[2],time_extra),(-1,num_neu*steps))
        ind_nnan_bool=~np.isnan(np.sum(fr_nan,axis=1))
        ind_nnan=np.where(ind_nnan_bool)[0]
        
        ##################################################
        # Behavior
        # Numero 1 y 2 top
        for h in range(len(ind_ch01_s0)):
            cc_01_0=func_eval(ind_ch01_s0[h],t_back,t_forw,stimulus,rt)
            beha_tested_rlow.append(cc_01_0[0]) #1
            beha_untested_rhigh.append(cc_01_0[1]) #2
        # Numero 3 y 4 top
        for h in range(len(ind_ch01_s1)):
            cc_01_1=func_eval(ind_ch01_s1[h],t_back,t_forw,stimulus,rt)
            beha_untested_rlow.append(cc_01_1[0]) #3
            beha_tested_rhigh.append(cc_01_1[1]) #4
        # Numero 3 y 4 bottom           
        for h in range(len(ind_ch10_s0)):
            cc_10_0=func_eval(ind_ch10_s0[h],t_back,t_forw,stimulus,rt)
            beha_untested_rlow.append(cc_10_0[1]) #3 
            beha_tested_rhigh.append(cc_10_0[0]) #4 
        # Numero 1 y 2 bottom           
        for h in range(len(ind_ch10_s1)):
            cc_10_1=func_eval(ind_ch10_s1[h],t_back,t_forw,stimulus,rt)
            beha_tested_rlow.append(cc_10_1[1]) #1
            beha_untested_rhigh.append(cc_10_1[0]) #2

        ##################################################
        # Neuro
        # ind_train=ret_ind_train(coherence,ind_ch,t_back,t_forw)
        # cl=LogisticRegression(C=1/reg,class_weight='balanced')
        # cl.fit(fr_nan[np.intersect1d(ind_train,ind_nnan)],choice[np.intersect1d(ind_train,ind_nnan)])
        # distances=norm_quant_coh(proj_dist(cl.coef_[0],cl.intercept_[0],fr_nan),coherence)
        # print ('Perc. not discarded ',np.sum(ind_nnan_bool)/len(ind_nnan_bool),np.sum(~np.isnan(distances))/len(distances))

        # # Numero 1 y 2 top
        # for h in range(len(ind_ch01_s0)):
        #     cc_01_0=func_eval(ind_ch01_s0[h],t_back,t_forw,stimulus,distances)#choice_cl,new_ctx='right')
        #     neuro_tested_rlow.append(cc_01_0[0]) #1
        #     neuro_untested_rhigh.append(cc_01_0[1]) #2
        # # Numero 3 y 4 top
        # for h in range(len(ind_ch01_s1)):
        #     cc_01_1=func_eval(ind_ch01_s1[h],t_back,t_forw,stimulus,distances)#choice_cl,new_ctx='right')
        #     neuro_untested_rlow.append(cc_01_1[0]) #3
        #     neuro_tested_rhigh.append(cc_01_1[1]) #4
        # # Numero 3 y 4 bottom           
        # for h in range(len(ind_ch10_s0)):
        #     cc_10_0=func_eval(ind_ch10_s0[h],t_back,t_forw,stimulus,distances)#choice_cl,new_ctx='left')
        #     neuro_untested_rlow.append(cc_10_0[1]) #3
        #     neuro_tested_rhigh.append(cc_10_0[0]) #4
        # # Numero 1 y 2 bottom           
        # for h in range(len(ind_ch10_s1)):
        #     cc_10_1=func_eval(ind_ch10_s1[h],t_back,t_forw,stimulus,distances)#choice_cl,new_ctx='left')
        #     neuro_tested_rlow.append(cc_10_1[1]) #1
        #     neuro_untested_rhigh.append(cc_10_1[0]) #2
        ############################################

    # Behavior
    beha_te_unte[0,0,hh]=np.nanmean(beha_tested_rlow,axis=0)
    beha_te_unte[0,1,hh]=np.nanmean(beha_tested_rhigh,axis=0)
    beha_te_unte[1,0,hh]=np.nanmean(beha_untested_rlow,axis=0)
    beha_te_unte[1,1,hh]=np.nanmean(beha_untested_rhigh,axis=0)
    aa00=fit_plot(xx,beha_te_unte[0,0,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds,sign=-1)
    aa01=fit_plot(xx,beha_te_unte[0,1,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds,sign=1)
    aa10=fit_plot(xx,beha_te_unte[1,0,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds,sign=-1)
    aa11=fit_plot(xx,beha_te_unte[1,1,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds,sign=1)
    fit_beha[0,0,hh,(t_back+1):]=aa00
    fit_beha[0,0,hh,0:t_back]=np.nanmean(beha_te_unte[0,0,hh,0:t_back])
    fit_beha[0,1,hh,(t_back+1):]=aa01
    fit_beha[0,1,hh,0:t_back]=np.nanmean(beha_te_unte[0,1,hh,0:t_back])
    fit_beha[1,0,hh,(t_back+1):]=aa10
    fit_beha[1,0,hh,0:t_back]=np.nanmean(beha_te_unte[1,0,hh,0:t_back])
    fit_beha[1,1,hh,(t_back+1):]=aa11
    fit_beha[1,1,hh,0:t_back]=np.nanmean(beha_te_unte[1,1,hh,0:t_back])
    y0_beha[0,0,hh]=aa00[0]
    y0_beha[0,1,hh]=aa01[0]
    y0_beha[1,0,hh]=aa10[0]
    y0_beha[1,1,hh]=aa11[0]

    # Neuro
    # neuro_te_unte[0,0,hh]=np.nanmean(neuro_tested_rlow,axis=0)
    # neuro_te_unte[0,1,hh]=np.nanmean(neuro_tested_rhigh,axis=0)
    # neuro_te_unte[1,0,hh]=np.nanmean(neuro_untested_rlow,axis=0)
    # neuro_te_unte[1,1,hh]=np.nanmean(neuro_untested_rhigh,axis=0)
    # aa00=fit_plot(xx,neuro_te_unte[0,0,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds,sign=-1)
    # aa01=fit_plot(xx,neuro_te_unte[0,1,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds,sign=1)
    # aa10=fit_plot(xx,neuro_te_unte[1,0,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds,sign=-1)
    # aa11=fit_plot(xx,neuro_te_unte[1,1,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds,sign=1)
    # fit_neuro[0,0,hh,(t_back+1):]=aa00
    # fit_neuro[0,0,hh,0:t_back]=np.nanmean(neuro_te_unte[0,0,hh,0:t_back])
    # fit_neuro[0,1,hh,(t_back+1):]=aa01
    # fit_neuro[0,1,hh,0:t_back]=np.nanmean(neuro_te_unte[0,1,hh,0:t_back])
    # fit_neuro[1,0,hh,(t_back+1):]=aa10
    # fit_neuro[1,0,hh,0:t_back]=np.nanmean(neuro_te_unte[1,0,hh,0:t_back])
    # fit_neuro[1,1,hh,(t_back+1):]=aa11
    # fit_neuro[1,1,hh,0:t_back]=np.nanmean(neuro_te_unte[1,1,hh,0:t_back])
    # y0_neuro[0,0,hh]=aa00[0]
    # y0_neuro[0,1,hh]=aa01[0]
    # y0_neuro[1,0,hh]=aa10[0]
    # y0_neuro[1,1,hh]=aa11[0]

# ####################################################
# # Behavior
beha_m=np.nanmean(beha_te_unte,axis=2)
beha_sem=sem(beha_te_unte,axis=2,nan_policy='omit')
beha_fit_m=np.nanmean(fit_beha,axis=2)
beha_fit_sem=sem(fit_beha,axis=2,nan_policy='omit')
if delta_type=='raw':
    delta_beha=(beha_te_unte[:,:,:,t_back+1]-np.nanmean(beha_te_unte[:,:,:,0:t_back],axis=3))
if delta_type=='fit':
    delta_beha=(y0_beha-np.nanmean(beha_te_unte[:,:,:,0:t_back],axis=3))
delta_beha_m=np.nanmean(delta_beha,axis=2)
delta_beha_sem=sem(delta_beha,axis=2,nan_policy='omit')

fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.scatter(xx,beha_m[0,0],color='green',s=3)
ax.scatter(xx,beha_m[1,0],color='blue',s=3)
ax.axvline(0,color='black',linestyle='--')
ax.plot(xx,0.0*np.ones(len(xx)),color='black',linestyle='--')
ax.plot(xx,beha_fit_m[0,0],color='green',label='Tested')
ax.fill_between(xx,beha_fit_m[0,0]-beha_fit_sem[0,0],beha_fit_m[0,0]+beha_fit_sem[0,0],color='green',alpha=0.5)
ax.plot(xx,beha_fit_m[1,0],color='blue',label='Untested')
ax.fill_between(xx,beha_fit_m[1,0]-beha_fit_sem[1,0],beha_fit_m[1,0]+beha_fit_sem[1,0],color='blue',alpha=0.5)
ax.set_ylim([-1.2,1.2])
ax.set_xlabel('Trials after context change')
ax.set_ylabel('Normalized RT')
plt.legend(loc='best')
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/beha_rt_inference2_%s_%s_lowR_stim.pdf'%(monkey,delta_type),dpi=500,bbox_inches='tight')

fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.scatter(xx,beha_m[0,1],color='green',s=3)
ax.scatter(xx,beha_m[1,1],color='blue',s=3)
ax.axvline(0,color='black',linestyle='--')
ax.plot(xx,0.0*np.ones(len(xx)),color='black',linestyle='--')
ax.plot(xx,beha_fit_m[0,1],color='green',label='Tested')
ax.fill_between(xx,beha_fit_m[0,1]-beha_fit_sem[0,1],beha_fit_m[0,1]+beha_fit_sem[0,1],color='green',alpha=0.5)
ax.plot(xx,beha_fit_m[1,1],color='blue',label='Untested')
ax.fill_between(xx,beha_fit_m[1,1]-beha_fit_sem[1,1],beha_fit_m[1,1]+beha_fit_sem[1,1],color='blue',alpha=0.5)
ax.set_ylim([-1.2,1.2])
ax.set_xlabel('Trials after context change')
ax.set_ylabel('Normalized RT')
plt.legend(loc='best')
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/beha_rt_inference2_%s_%s_highR_stim.pdf'%(monkey,delta_type),dpi=500,bbox_inches='tight')

width=0.3
fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.bar(-width/2.0,delta_beha_m[0,0],yerr=delta_beha_sem[0,0],color='green',width=width,label='Tested')
ax.bar(+width/2.0,delta_beha_m[1,0],yerr=delta_beha_sem[1,0],color='blue',width=width,label='Untested')
ax.bar(1-width/2.0,delta_beha_m[0,1],yerr=delta_beha_sem[0,1],color='green',width=width)
ax.bar(1+width/2.0,delta_beha_m[1,1],yerr=delta_beha_sem[1,1],color='blue',width=width)
#ax.set_ylim([0,1])
ax.set_ylabel('$\Delta$Normalized RT')
ax.set_xlabel('Stimulus')
plt.xticks([0,1],['Previos Ctx','New Ctx'])
plt.legend(loc='best')
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/rt_inference2_%s_%s.pdf'%(monkey,delta_type),dpi=500,bbox_inches='tight')


######################################################
# Neuro

# neu_m=np.nanmean(neuro_te_unte,axis=2)
# neu_sem=sem(neuro_te_unte,axis=2,nan_policy='omit')
# neu_fit_m=np.nanmean(fit_neuro,axis=2)
# neu_fit_sem=sem(fit_neuro,axis=2,nan_policy='omit')
# if delta_type=='raw':
#     delta_neuro=(neuro_te_unte[:,:,:,t_back+1]-np.nanmean(neuro_te_unte[:,:,:,0:t_back],axis=3))
# if delta_type=='fit':
#     delta_neuro=(y0_neuro-np.nanmean(neuro_te_unte[:,:,:,0:t_back],axis=3))
# delta_neuro_m=np.nanmean(delta_neuro,axis=2)
# delta_neuro_sem=sem(delta_neuro,axis=2,nan_policy='omit')

# fig=plt.figure(figsize=(2.3,2))
# ax=fig.add_subplot(111)
# miscellaneous.adjust_spines(ax,['left','bottom'])
# ax.scatter(xx,neu_m[0,0],color='green',s=3)
# ax.scatter(xx,neu_m[1,0],color='blue',s=3)
# ax.axvline(0,color='black',linestyle='--')
# ax.plot(xx,0*np.ones(len(xx)),color='black',linestyle='--')
# ax.plot(xx,neu_fit_m[0,0],color='green')
# ax.fill_between(xx,neu_fit_m[0,0]-neu_fit_sem[0,0],neu_fit_m[0,0]+neu_fit_sem[0,0],color='green',alpha=0.5)
# ax.plot(xx,neu_fit_m[1,0],color='blue')
# ax.fill_between(xx,neu_fit_m[1,0]-neu_fit_sem[1,0],neu_fit_m[1,0]+neu_fit_sem[1,0],color='blue',alpha=0.5)
# #ax.set_ylim([-0.05,1.05])
# ax.set_xlabel('Trials after context change')
# #ax.set_ylabel('Prob. (Choice = New Ctx)')
# fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/neuro_rt_inference2_%s_%s_lowR.pdf'%(monkey,delta_type),dpi=500,bbox_inches='tight')

# fig=plt.figure(figsize=(2.3,2))
# ax=fig.add_subplot(111)
# miscellaneous.adjust_spines(ax,['left','bottom'])
# ax.scatter(xx,neu_m[0,1],color='green',s=3)
# ax.scatter(xx,neu_m[1,1],color='blue',s=3)
# ax.axvline(0,color='black',linestyle='--')
# ax.plot(xx,0*np.ones(len(xx)),color='black',linestyle='--')
# ax.plot(xx,neu_fit_m[0,1],color='green')
# ax.fill_between(xx,neu_fit_m[0,1]-neu_fit_sem[0,1],neu_fit_m[0,1]+neu_fit_sem[0,1],color='green',alpha=0.5)
# ax.plot(xx,neu_fit_m[1,1],color='blue')
# ax.fill_between(xx,neu_fit_m[1,1]-neu_fit_sem[1,1],neu_fit_m[1,1]+neu_fit_sem[1,1],color='blue',alpha=0.5)
# #ax.set_ylim([-0.05,1.05])
# ax.set_xlabel('Trials after context change')
# #ax.set_ylabel('Prob. (Choice = New Ctx)')
# fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/neuro_rt_inference2_%s_%s_highR.pdf'%(monkey,delta_type),dpi=500,bbox_inches='tight')

# width=0.3
# fig=plt.figure(figsize=(2.3,2))
# ax=fig.add_subplot(111)
# miscellaneous.adjust_spines(ax,['left','bottom'])
# ax.bar(-width/2.0,delta_neuro_m[0,0],yerr=delta_neuro_sem[0,0],color='green',width=width,label='Tested')
# ax.bar(+width/2.0,delta_neuro_m[1,0],yerr=delta_neuro_sem[1,0],color='blue',width=width,label='Untested')
# ax.bar(1-width/2.0,delta_neuro_m[0,1],yerr=delta_neuro_sem[0,1],color='green',width=width)
# ax.bar(1+width/2.0,delta_neuro_m[1,1],yerr=delta_neuro_sem[1,1],color='blue',width=width)
# #ax.set_ylim([0,1])
# ax.set_ylabel('$\Delta$Prob. (Choice = New Ctx)')
# ax.set_xlabel('Stimulus')
# plt.xticks([0,1],['Previos Ctx','New Ctx'])
# plt.legend(loc='best')
# fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/neuro_rt_inference2_%s_%s.pdf'%(monkey,delta_type),dpi=500,bbox_inches='tight')

# # Main Figure both neuro and Behavior
# delta_behaf=np.reshape(delta_beha,(2,-1))
# delta_behaf_m=np.nanmean(delta_behaf,axis=1)
# delta_behaf_sem=sem(delta_behaf,axis=1,nan_policy='omit')
# print ('Delta Behavior')
# print ('Tested ',scipy.stats.wilcoxon(delta_behaf[0]))
# print ('Untested ',scipy.stats.wilcoxon(delta_behaf[1]))
# print ('Tested - Untested ',scipy.stats.wilcoxon(delta_behaf[1]-delta_behaf[0]))
# delta_neurof=np.reshape(delta_neuro,(2,-1))
# delta_neurof_m=np.nanmean(delta_neurof,axis=1)
# delta_neurof_sem=sem(delta_neurof,axis=1,nan_policy='omit')
# print ('Delta Neuro')
# print ('Tested ',scipy.stats.wilcoxon(delta_neurof[0]))
# print ('Untested ',scipy.stats.wilcoxon(delta_neurof[1]))
# print ('Tested - Untested ',scipy.stats.wilcoxon(delta_neurof[1]-delta_neurof[0]))

# width=0.3
# fig=plt.figure(figsize=(2.3,2))
# ax=fig.add_subplot(111)
# miscellaneous.adjust_spines(ax,['left','bottom'])
# ax.bar(-width/2.0,delta_behaf_m[0],yerr=delta_behaf_sem[0],color='green',width=width,label='Tested')
# ax.bar(+width/2.0,delta_behaf_m[1],yerr=delta_behaf_sem[1],color='blue',width=width,label='Untested')
# ax.bar(1-width/2.0,delta_neurof_m[0],yerr=delta_neurof_sem[0],color='green',width=width)
# ax.bar(1+width/2.0,delta_neurof_m[1],yerr=delta_neurof_sem[1],color='blue',width=width)
# ax.set_ylabel('$\Delta$ Prob. (Choice = New Ctx)')
# plt.xticks([0,1],['Behavior','Neuronal'])
# plt.legend(loc='best')
# fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/inference2_%s_%s_final.pdf'%(monkey,delta_type),dpi=500,bbox_inches='tight')
