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

# In this script we evaluate generalization through learning.
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

def func1(x,a,b,c):
    #return np.exp(-a*x+b)+c
    return np.exp(-a*x)*b+c

def func2(x,a,b,c):
    #return -np.exp(-a*x+b)+c
    return -np.exp(-a*x)*b+c

def fit_plot(xx,yy,sign,t_back,t_forw,sig_kernel,maxfev,method,bounds,p0):
    if sign==1:
        popt,pcov=curve_fit(func1,xx[t_back:],yy[t_back:],nan_policy='omit',maxfev=maxfev,bounds=bounds,p0=p0,method=method)
        fit_func=func1(xx[t_back:],popt[0],popt[1],popt[2])
    if sign==-1:
        popt,pcov=curve_fit(func2,xx[t_back:],yy[t_back:],nan_policy='omit',maxfev=maxfev,bounds=bounds,p0=p0,method=method)
        fit_func=func2(xx[t_back:],popt[0],popt[1],popt[2])
    print ('Fit ',popt)
    print (pcov)
    # plt.scatter(xx,yy,color='blue',s=1)
    # plt.plot(xx[t_back:],fit_func,color='black')
    # plt.axvline(0,color='black',linestyle='--')
    # plt.show()
    return fit_func,popt[0]

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

def proj_dist(wei,wei0,fr):
    return abs(np.dot(fr,wei.T)+wei0)

def fr_rt_nan(reaction_time,firing_rate,tt,tw,time_extra):
    fr_nan=firing_rate.copy()
    rt_max=(reaction_time-time_extra)
    for i in range(len(tt)):
        ind_nan=(rt_max<(1000*tt[i]+tw))
        fr_nan[ind_nan,:,i]=nan
    return fr_nan

def create_context_subj(context_pre,ctx_ch_pre,ctx_ch):
    context_subj=context_pre.copy()
    for i in range(len(ctx_ch)):
        diff=(ctx_ch[i]-ctx_ch_pre[i])
        context_subj[ctx_ch_pre[i]:(ctx_ch_pre[i]+diff+1)]=context_pre[ctx_ch_pre[i]-1]
    return context_subj
  
#################################################

# Niels: t_back 20, t_forw 80, dic_time (-200,400,200,200)ms. No kernel. Groups of 1 session
# Galileo: t_back 20, t_forw 80, dic_time (-200,600,200,200)ms. No kernel. Groups of 3 sessions

monkeys=['Niels','Galileo']
t_back=20
t_forw=80
sig_kernel=1 # not smaller than 1

#thres=0
#reg=1e0
maxfev=100000
method='dogbox'

bounds=([0,0,-1],[1000,10,1])
#bounds=([-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf])
p0=(0.05,0.5,0.1)

xx=np.arange(t_back+t_forw)-t_back

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])

beha_ctx_ch_all=nan*np.zeros((2,9,t_back+t_forw))
fit_beha_all=nan*np.zeros((2,9,t_back+t_forw))
inter_beha_all=nan*np.zeros((2,9))
uu=-1
for k in range(len(monkeys)):

    if monkeys[k]=='Niels':
        files_groups=[[8,9],[9,10],[10,11],[11,12]]

    if monkeys[k]=='Galileo':
        files_groups=[[20,22],[22,24],[24,26],[26,28],[28,30]]
        #files_groups=[[20,21],[21,22],[22,23],[23,24],[24,25],[25,26],[26,27],[27,28],[28,29],[29,30]]
    
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/unsorted/%s/'%(monkeys[k]) 
    files_pre=np.array(os.listdir(abs_path))
    order=order_files(files_pre)
    files_all=np.array(files_pre[order])
    print (files_all)

    beha_ctx_ch=nan*np.zeros((2,len(files_groups),t_back+t_forw))
    fit_beha=nan*np.zeros((2,len(files_groups),t_back+t_forw))
    inter_beha=nan*np.zeros((2,len(files_groups)))
    
    for hh in range(len(files_groups)):
        uu+=1
        xx_forw_pre=nan*np.zeros((100,(t_back+t_forw)))
        beha_pre=nan*np.zeros((2,100,(t_back+t_forw)))
        neu_pre=nan*np.zeros((2,100,(t_back+t_forw)))
        gg=-1
        oo=-1
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
            context_prepre=beha['context']
            ctx_ch=(context_prepre[1:]-context_prepre[0:-1])
            context_pre=context_prepre[1:]
            #ind_ch=np.where(abs(ctx_ch)==1)[0]#Careful!
            ind_ch_pre=np.where(abs(ctx_ch)==1)[0]
            ind_ch=calculate_ind_ch_corr(ind_ch_pre,reward)
            context=create_context_subj(context_pre,ind_ch_pre,ind_ch) # Careful! this is subjective context
            #indch_ct10=np.where(ctx_ch==-1)[0]
            #indch_ct01=np.where(ctx_ch==1)[0]
            #print (ind_ch,len(choice))

            ##################################################
            # Behavior
            # Probability of Choice

            for h in range(len(ind_ch)):
                gg+=1
                for j in range(t_back):
                    ind=(ind_ch[h]-t_back+j)
                    try:
                        if context[ind]==stimulus[ind]:
                            beha_pre[0,gg,j]=rt[ind]
                        if context[ind]!=stimulus[ind]:
                            beha_pre[1,gg,j]=rt[ind]
                    except:
                        None
                
                for j in range(t_forw):
                    ind=(ind_ch[h]+j)
                    try:
                        if context[ind]==stimulus[ind]:
                            beha_pre[0,gg,t_back+j]=rt[ind]
                        if context[ind]!=stimulus[ind]:
                            beha_pre[1,gg,t_back+j]=rt[ind]
                    except:
                        None

        beha_ctx_ch[:,hh]=np.nanmean(beha_pre,axis=1)
        beha_ctx_ch_all[:,uu]=np.nanmean(beha_pre,axis=1)

        # Behavior
        aa0=fit_plot(xx,beha_ctx_ch[0,hh],1,t_back,t_forw,sig_kernel,maxfev,method=method,p0=p0,bounds=bounds)
        fit_beha[0,hh,t_back:]=aa0[0]
        fit_beha[0,hh,0:t_back]=np.mean(beha_ctx_ch[0,hh,0:t_back])
        inter_beha[0,hh]=aa0[1]
        fit_beha_all[0,uu,t_back:]=aa0[0]
        fit_beha_all[0,uu,0:t_back]=np.mean(beha_ctx_ch[0,hh,0:t_back])
        inter_beha_all[0,uu]=aa0[1]

        aa1=fit_plot(xx,beha_ctx_ch[1,hh],-1,t_back,t_forw,sig_kernel,maxfev,method=method,p0=p0,bounds=bounds)
        fit_beha[1,hh,t_back:]=aa1[0]
        fit_beha[1,hh,0:t_back]=np.mean(beha_ctx_ch[1,hh,0:t_back])
        inter_beha[1,hh]=aa1[1]
        fit_beha_all[1,uu,t_back:]=aa1[0]
        fit_beha_all[1,uu,0:t_back]=np.mean(beha_ctx_ch[1,hh,0:t_back])
        inter_beha_all[1,uu]=aa1[1]
        
    ##################################
    beha_ch_m=np.nanmean(beha_ctx_ch,axis=1)
    beha_ch_sem=sem(beha_ctx_ch,axis=1,nan_policy='omit')
   
    fit_beha_m=np.nanmean(fit_beha,axis=1)
    fit_beha_sem=sem(fit_beha,axis=1,nan_policy='omit')
   
    # Plot Behavior
    fig=plt.figure(figsize=(2.3,2))
    ax=fig.add_subplot(111)
    miscellaneous.adjust_spines(ax,['left','bottom'])
    ax.axvline(0,color='black',linestyle='--')
    ax.plot(xx,0*np.ones(len(xx)),color='black',linestyle='--')
    ax.plot(xx,fit_beha_m[0],color='green',label='Stim.& Ctx congruent')
    ax.fill_between(xx,fit_beha_m[0]-fit_beha_sem[0],fit_beha_m[0]+fit_beha_sem[0],color='green',alpha=0.5)
    ax.plot(xx,fit_beha_m[1],color='blue',label='Stim. & Ctx incongruent')
    ax.fill_between(xx,fit_beha_m[1]-fit_beha_sem[1],fit_beha_m[1]+fit_beha_sem[1],color='blue',alpha=0.5)
    ax.scatter(xx,beha_ch_m[0],color='green',s=1)
    ax.scatter(xx,beha_ch_m[1],color='blue',s=1)
    #ax.set_ylim([0,1])
    ax.set_xlabel('Trials after context change')
    ax.set_ylabel('Normalized Reaction Time')
    #plt.legend(loc='best')
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/reaction_time_context_beha_%s.pdf'%(monkeys[k]),dpi=500,bbox_inches='tight')

##################################
beha_ch_all_m=np.nanmean(beha_ctx_ch_all,axis=1)
beha_ch_all_sem=sem(beha_ctx_ch_all,axis=1,nan_policy='omit')
fit_beha_all_m=np.nanmean(fit_beha_all,axis=1)
fit_beha_all_sem=sem(fit_beha_all,axis=1,nan_policy='omit')

# Plot Behavior
fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.axvline(0,color='black',linestyle='--')
ax.plot(xx,0*np.ones(len(xx)),color='black',linestyle='--')
ax.plot(xx,fit_beha_all_m[0],color='green',label='Stim.& Ctx congruent')
ax.fill_between(xx,fit_beha_all_m[0]-fit_beha_all_sem[0],fit_beha_all_m[0]+fit_beha_all_sem[0],color='green',alpha=0.5)
ax.plot(xx,fit_beha_all_m[1],color='blue',label='Stim. & Ctx incongruent')
ax.fill_between(xx,fit_beha_all_m[1]-fit_beha_all_sem[1],fit_beha_all_m[1]+fit_beha_all_sem[1],color='blue',alpha=0.5)
ax.scatter(xx,beha_ch_all_m[1],color='blue',s=1)
ax.scatter(xx,beha_ch_all_m[0],color='green',s=1)
#ax.set_ylim([0,1])
ax.set_xlabel('Trials after context change')
ax.set_ylabel('Normalized Reaction Time')
#plt.legend(loc='best')
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/reaction_time_context_beha_both.pdf',dpi=500,bbox_inches='tight')
    
   
