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

def func1(x,a,b,c):
    #return np.exp(-a*x+b)+c
    return np.exp(-a*x)*b+c

def func2(x,a,b,c):
    #return -np.exp(-a*x+b)+c
    return -np.exp(-a*x)*b+c

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

# def func_eval_behav(index,stimulus,rt):
#     #pp=nan*np.zeros((2,t_forw+t_back))
#     pp=nan*np.zeros(2)
    
#     stimp=stimulus[index]
    
#     t_back=70
#     rt_low_pre=np.mean(rt[])
#     if stimulus[index+1]==0:
#         pp[0]=rt[index+1]
#     if stimulus[index+1]==1:
#         pp[1]=rt[index+1]
#     return pp

# def func_eval_neuro(index,t_back,t_forw,stimulus,context,fr,reg):
#     #
#     ind_del_pre=(index+(np.arange(t_back+t_forw)-t_back))
#     ind_train_pre=np.arange(len(context))
#     ind_del=ind_del_pre[ind_del_pre<=np.max(ind_train_pre)]
#     ind_train=np.delete(ind_train_pre,ind_del)
#     #
#     cl=LogisticRegression(C=1/reg,class_weight='balanced')
#     cl.fit(fr[ind_train],context[ind_train])
#     choice=cl.predict(fr)
       
#     pp=nan*np.zeros((2,t_forw+t_back))
#     for j in range(t_back):
#         indj=(index-t_back+j)
#         try:
#             if stimulus[indj]==0:
#                 pp[0,j]=(choice[indj]==context[indj])
#             if stimulus[indj]==1:
#                 pp[1,j]=(choice[indj]==context[indj])
#         except:
#             None
#     for j in range(t_forw):
#         indj=(index+j)
#         try:
#             if stimulus[indj]==0:
#                 pp[0,j+t_back]=(choice[indj]==context[indj])
#             if stimulus[indj]==1:
#                 pp[1,j+t_back]=(choice[indj]==context[indj])            
#         except:
#             None
#     return pp

def fit_plot(xx,yy,sign,t_back,t_forw,sig_kernel,maxfev,method,bounds,p0):
    if sign==1:
        popt,pcov=curve_fit(func1,xx[t_back:],yy[t_back:],nan_policy='omit',maxfev=maxfev,bounds=bounds,p0=p0,method=method)
        fit_func=func1(xx[t_back:],popt[0],popt[1],popt[2])
    if sign==-1:
        popt,pcov=curve_fit(func2,xx[t_back:],yy[t_back:],nan_policy='omit',maxfev=maxfev,bounds=bounds,p0=p0,method=method)
        fit_func=func2(xx[t_back:],popt[0],popt[1],popt[2])
    #print ('Fit ',popt)
    #print (pcov)
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

def proj_dist(wei,wei0,fr):
    return abs(np.dot(fr,wei.T)+wei0)

def fr_rt_nan(reaction_time,firing_rate,tt,tw,time_extra):
    fr_nan=firing_rate.copy()
    rt_max=(reaction_time-time_extra)
    for i in range(len(tt)):
        ind_nan=(rt_max<(1000*tt[i]+tw))
        fr_nan[ind_nan,:,i]=nan
    return fr_nan

def chrono_curve(x,Bl,Br,K,c_shift,t0l,t0r): # x[:,0] is coherence and x[:,1] is choice
    decl=(1-x[:,1])*(Bl/(K*(x[:,0]-c_shift)))*np.tanh(K*Bl*(x[:,0]-c_shift))
    decr=x[:,1]*(Br/(K*(x[:,0]-c_shift)))*np.tanh(K*Br*(x[:,0]-c_shift))
    tnd=(1-x[:,1])*t0l+x[:,1]*t0r
    return decl+decr+tnd

def func_fit_chrono(ind_fit,xx,rt,coh_signed,coh_uq,maxfev,p0,method):
    popt,pcov,infodict,mesg,ier=curve_fit(chrono_curve,xx[ind_fit],rt[ind_fit],maxfev=maxfev,p0=p0,method=method,full_output=True)
    yy=chrono_curve(xx[ind_fit],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
    #print (popt)
    fit_chrono=nan*np.zeros(len(coh_uq))
    for ii in range(len(coh_uq)):
        fit_chrono[ii]=np.mean(yy[np.where(coh_signed[ind_fit]==coh_uq[ii])[0]])
    return fit_chrono,popt

def create_context_subj(context_pre,ctx_ch_pre,ctx_ch):
    context_subj=context_pre.copy()
    for i in range(len(ctx_ch)):
        diff=(ctx_ch[i]-ctx_ch_pre[i])
        context_subj[ctx_ch_pre[i]:(ctx_ch_pre[i]+diff)]=context_pre[ctx_ch_pre[i]-1]
    return context_subj

#################################################

# Function 2 for both. Bounds and p0 are important.
# Niels: t_back 20, t_forw 80, dic_time (-200,400,200,200)ms. No kernel. Groups of 1 session
# Galileo: t_back 20, t_forw 80, dic_time (-200,600,200,200)ms. No kernel. Groups of 3 sessions

monkeys=['Niels','Galileo']
stage='late'

nback=30
rt_fit=True

maxfev=100000
p0=(-20,20,-0.005,0.1,500,500)
p0l=(-20,20,-0.005,-3,500,700)
p0r=(-20,20,-0.005,3,700,500)
method='trf'

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])

beha_te_unte_all=nan*np.zeros((2,2,9))
uu=-1
for k in range(len(monkeys)):

    if monkeys[k]=='Niels':
        if stage=='early':
            files_groups=[[0,1],[1,2],[2,3],[3,4]]
        if stage=='mid':
            files_groups=[[4,5],[5,6],[6,7],[7,8]]
        if stage=='late':
            files_groups=[[8,9],[9,10],[10,11],[11,12]]
    if monkeys[k]=='Galileo':
        if stage=='early':
            files_groups=[[0,2],[2,4],[4,6],[6,8],[8,10]]
        if stage=='mid':
            files_groups=[[10,12],[12,14],[14,16],[16,18],[18,20]]
        if stage=='late':
            files_groups=[[20,22],[22,24],[24,26],[26,28],[28,30]]
   
    abs_path='/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/data/unsorted/%s/'%(monkeys[k]) 
    files_pre=np.array(os.listdir(abs_path))
    order=order_files(files_pre)
    files_all=np.array(files_pre[order])
    print (files_all)

    beha_te_unte=nan*np.zeros((2,2,len(files_groups)))
    for hh in range(len(files_groups)):
        uu+=1
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
            beha=miscellaneous.behavior(data,group_ref)
            index_nonan=beha['index_nonan']
            # We discard first trial of session because we are interested in context changes
            stimulus=beha['stimulus'][1:]
            choice=beha['choice'][1:]
            coh_signed=beha['coherence_signed'][1:]
            coh_set_signed=np.unique(coh_signed)
            reward=beha['reward'][1:]
            if rt_fit==True:
                rt=beha['reaction_time'][1:]
            if rt_fit==False:
                rt_pre=beha['reaction_time'][1:]
                rt=norm_quant_coh(rt_pre,coh_signed)
            context_prepre=beha['context'] 
            ctx_ch=(context_prepre[1:]-context_prepre[0:-1])
            context_pre=context_prepre[1:]
            # Indices for first trial rewarded after change
            ind_ch_pre=np.where(abs(ctx_ch)==1)[0] # ind_ch_pre index where there is a context change
            ind_ch=calculate_ind_ch_corr(ind_ch_pre,reward) # ind_ch first correct trial after context change (otherwise animal doesn't know there was a change)
            context=create_context_subj(context_pre,ind_ch_pre,ind_ch) # CAREFUL! this is subjective context
            indch_ct01_pre=np.where(ctx_ch==1)[0]
            indch_ct10_pre=np.where(ctx_ch==-1)[0]
            ind_ch01_s0,ind_ch01_s1,ind_ch10_s0,ind_ch10_s1=calculate_ind_ch_corr2(indch_ct01_pre,indch_ct10_pre,reward,stimulus)
            print (ind_ch01_s0,ind_ch01_s1,ind_ch10_s0,ind_ch10_s1)
    
            ##################################################
            # Behavior
            # Probability of Choice = Context for all possibilities: 01 0, 01 1, 10 0, 10 1
            
            # In order to get the results from Roozbeh, we need to sustract the rt of the previous low and high.
            # For instance, to get the results for the "High Reward" column, we need to evaluate the rt for the stimuli that correspond to current high reward and sustract the rt for the same stimuli before the context switch (low reward before context switch).
            # The same could be potentially done for choice
            # We can also do the same for neural data

            xx=np.array([100*coh_signed,choice]).T
        
            
            # Current high is Right and current low is Left. Previous low is Right and previous high is Left. First reward trial after context change is Left.
            # rlow tested is left, rhigh untested is right
            ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
            for h in range(len(ind_ch01_s0)):
                ind_pre=(np.arange(nback)-nback+ind_ch01_s0[h]+1)
                ind_used[ind_pre]=True
                ind_used[np.isnan(rt)]=False
            for h in range(len(ind_ch01_s0)):
                if rt_fit==True:
                    popt=func_fit_chrono(ind_used,xx,rt,coh_signed,coh_set_signed,maxfev,p0l,method)[1]
                    #print (popt)
                    rt_mean=chrono_curve(xx[(ind_ch01_s0[h]+1):(ind_ch01_s0[h]+2)],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])[0]
                    dev=(rt[ind_ch01_s0[h]+1]-rt_mean)
                    if stimulus[ind_ch01_s0[h]+1]==0: 
                        beha_tested_rlow.append(dev)
                    if stimulus[ind_ch01_s0[h]+1]==1:
                        beha_untested_rhigh.append(dev)
                if rt_fit==False:
                    rt_low_pre=np.mean(rt[(ind_used)&(stimulus==1)]) # Previous Right (low in previous context)
                    rt_high_pre=np.mean(rt[(ind_used)&(stimulus==0)]) # Previous Left (high in previous context)
                    if stimulus[ind_ch01_s0[h]+1]==0: 
                        beha_tested_rlow.append(rt[ind_ch01_s0[h]+1]-rt_high_pre)
                    if stimulus[ind_ch01_s0[h]+1]==1:
                        beha_untested_rhigh.append(rt[ind_ch01_s0[h]+1]-rt_low_pre)

            # Current high is Right and current low is Left. Previous low is Right and previous high is Left. First reward trial after context change is Right.
            # rlow untested is left, rhigh tested is right
            ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
            for h in range(len(ind_ch01_s1)):
                ind_pre=(np.arange(nback)-nback+ind_ch01_s1[h]+1)
                ind_used[ind_pre]=True
                ind_used[np.isnan(rt)]=False
            for h in range(len(ind_ch01_s1)):
                if rt_fit==True:
                    popt=func_fit_chrono(ind_used,xx,rt,coh_signed,coh_set_signed,maxfev,p0l,method)[1]
                    rt_mean=chrono_curve(xx[(ind_ch01_s1[h]+1):(ind_ch01_s1[h]+2)],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])[0]
                    dev=(rt[ind_ch01_s1[h]+1]-rt_mean)
                    if stimulus[ind_ch01_s1[h]+1]==0:
                        beha_untested_rlow.append(dev)
                    if stimulus[ind_ch01_s1[h]+1]==1:
                        beha_tested_rhigh.append(dev)
                if rt_fit==False:
                    rt_low_pre=np.mean(rt[(ind_used)&(stimulus==1)]) # Previous Right
                    rt_high_pre=np.mean(rt[(ind_used)&(stimulus==0)]) # Previous Left
                    if stimulus[ind_ch01_s1[h]+1]==0:
                        beha_untested_rlow.append(rt[ind_ch01_s1[h]+1]-rt_high_pre)
                    if stimulus[ind_ch01_s1[h]+1]==1:
                        beha_tested_rhigh.append(rt[ind_ch01_s1[h]+1]-rt_low_pre)

            # Current high is Left and current low is Right. Previous low is Left and previous high is Right. First reward trial after context change is Left.
            # rhigh tested is left, rlow untested is right
            ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
            for h in range(len(ind_ch10_s0)):
                ind_pre=(np.arange(nback)-nback+ind_ch10_s0[h]+1)
                ind_used[ind_pre]=True
                ind_used[np.isnan(rt)]=False
            for h in range(len(ind_ch10_s0)):
                if rt_fit==True:
                    popt=func_fit_chrono(ind_used,xx,rt,coh_signed,coh_set_signed,maxfev,p0r,method)[1]
                    rt_mean=chrono_curve(xx[(ind_ch10_s0[h]+1):(ind_ch10_s0[h]+2)],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])[0]
                    dev=(rt[ind_ch10_s0[h]+1]-rt_mean)
                    if stimulus[ind_ch10_s0[h]+1]==0: 
                        beha_tested_rhigh.append(dev)
                    if stimulus[ind_ch10_s0[h]+1]==1:
                        beha_untested_rlow.append(dev)
                if rt_fit==False:
                    rt_low_pre=np.mean(rt[(ind_used)&(stimulus==0)]) # Previous Left
                    rt_high_pre=np.mean(rt[(ind_used)&(stimulus==1)]) # Previous Right
                    if stimulus[ind_ch10_s0[h]+1]==0: 
                        beha_tested_rhigh.append(rt[ind_ch10_s0[h]+1]-rt_low_pre)
                    if stimulus[ind_ch10_s0[h]+1]==1:
                        beha_untested_rlow.append(rt[ind_ch10_s0[h]+1]-rt_high_pre)

            # Current high is Left and current low is Right. Previous low is Left and previous high is Right. First reward trial after context change is Right.
            # rhigh untested is left, rlow tested is right
            ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
            for h in range(len(ind_ch10_s1)):
                ind_pre=(np.arange(nback)-nback+ind_ch10_s1[h]+1)
                ind_used[ind_pre]=True
                ind_used[np.isnan(rt)]=False
            for h in range(len(ind_ch10_s1)):
                if rt_fit==True:
                    popt=func_fit_chrono(ind_used,xx,rt,coh_signed,coh_set_signed,maxfev,p0r,method)[1]
                    rt_mean=chrono_curve(xx[(ind_ch10_s1[h]+1):(ind_ch10_s1[h]+2)],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])[0]
                    dev=(rt[ind_ch10_s1[h]+1]-rt_mean)
                    if stimulus[ind_ch10_s1[h]+1]==0: 
                        beha_untested_rhigh.append(dev)
                    if stimulus[ind_ch10_s1[h]+1]==1:
                        beha_tested_rlow.append(dev)
                if rt_fit==False:
                    rt_low_pre=np.mean(rt[(ind_used)&(stimulus==0)]) # Previous Left
                    rt_high_pre=np.mean(rt[(ind_used)&(stimulus==1)]) # Previous Right
                    if stimulus[ind_ch10_s1[h]+1]==0: 
                        beha_untested_rhigh.append(rt[ind_ch10_s1[h]+1]-rt_low_pre)
                    if stimulus[ind_ch10_s1[h]+1]==1:
                        beha_tested_rlow.append(rt[ind_ch10_s1[h]+1]-rt_high_pre)

        # Behavior
        beha_te_unte[0,0,hh]=np.nanmean(beha_tested_rlow,axis=0)
        beha_te_unte[0,1,hh]=np.nanmean(beha_tested_rhigh,axis=0)
        beha_te_unte[1,0,hh]=np.nanmean(beha_untested_rlow,axis=0)
        beha_te_unte[1,1,hh]=np.nanmean(beha_untested_rhigh,axis=0)
        print (beha_tested_rlow)
        print (beha_tested_rhigh)
        print (beha_untested_rlow)
        print (beha_untested_rhigh)
        beha_te_unte_all[0,0,uu]=np.nanmean(beha_tested_rlow,axis=0)
        beha_te_unte_all[0,1,uu]=np.nanmean(beha_tested_rhigh,axis=0)
        beha_te_unte_all[1,0,uu]=np.nanmean(beha_untested_rlow,axis=0)
        beha_te_unte_all[1,1,uu]=np.nanmean(beha_untested_rhigh,axis=0)

    ####################################################
    beha_m=np.nanmean(beha_te_unte,axis=2)
    beha_sem=sem(beha_te_unte,axis=2,nan_policy='omit')
   
    width=0.3
    fig=plt.figure(figsize=(1.75,2))
    ax=fig.add_subplot(111)
    miscellaneous.adjust_spines(ax,['left','bottom'])
    ax.bar(-width/2.0,beha_m[0,1],yerr=beha_sem[0,1],color='green',width=width)
    ax.bar(width/2.0,beha_m[1,1],yerr=beha_sem[1,1],color='blue',width=width)
    ax.bar(1-width/2.0,beha_m[0,0],yerr=beha_sem[0,0],color='green',width=width,label='Tested')
    ax.bar(1+width/2.0,beha_m[1,0],yerr=beha_sem[1,0],color='blue',width=width,label='Untested')
    ax.set_ylabel('$\Delta$Reaction Time')
    if monkeys[k]=='Niels':
        ax.set_ylim([-250,400])
    if monkeys[k]=='Galileo':
        ax.set_ylim([-200,650])
    plt.xticks([0,1],['High Rew.','Low Rew.'])
    plt.legend(loc='best')
    if rt_fit==True:
        fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/rt_inference_Roozbeh_nback_%i_%s_rt_fit_%s.pdf'%(nback,monkeys[k],stage),dpi=500,bbox_inches='tight')
    if rt_fit==False:
        fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/rt_inference_Roozbeh_nback_%i_%s_raw_%s.pdf'%(nback,monkeys[k],stage),dpi=500,bbox_inches='tight')


####################################################
beha_m=np.nanmean(beha_te_unte_all,axis=2)
beha_sem=sem(beha_te_unte_all,axis=2,nan_policy='omit')

width=0.3
fig=plt.figure(figsize=(1.75,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.bar(-width/2.0,beha_m[0,1],yerr=beha_sem[0,1],color='green',width=width)
ax.bar(width/2.0,beha_m[1,1],yerr=beha_sem[1,1],color='blue',width=width)
ax.bar(1-width/2.0,beha_m[0,0],yerr=beha_sem[0,0],color='green',width=width,label='Tested')
ax.bar(1+width/2.0,beha_m[1,0],yerr=beha_sem[1,0],color='blue',width=width,label='Untested')
ax.set_ylabel('$\Delta$Reaction Time')
ax.set_ylim([-200,400])
plt.xticks([0,1],['High Rew.','Low Rew.'])
plt.legend(loc='best')
if rt_fit==True:
    fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/rt_inference_Roozbeh_nback_%i_both_rt_fit_%s.pdf'%(nback,stage),dpi=500,bbox_inches='tight')
if rt_fit==False:
    fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/rt_inference_Roozbeh_nback_%i_both_raw_%s.pdf'%(nback,stage),dpi=500,bbox_inches='tight')

pp=[['Tested Low','Tested High'],
    ['Untested Low','Untested High']]
for i in range(2):
    for ii in range(2):
        print (pp[i][ii])
        print (scipy.stats.wilcoxon(beha_te_unte_all[i,ii],nan_policy='omit'))

beha_red=np.reshape(beha_te_unte_all,(2,-1))
beha_red[:,9:]=-beha_red[:,9:]
ppp=['Tested','Untested']
for i in range(2):
    print (ppp[i])
    print (scipy.stats.wilcoxon(beha_red[i],nan_policy='omit'))

