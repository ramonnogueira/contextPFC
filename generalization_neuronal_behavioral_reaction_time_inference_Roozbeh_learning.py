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
    #print ('Fit ',popt)
    #print (pcov)
    # plt.scatter(xx,yy,color='blue',s=1)
    # plt.plot(xx[t_back:],fit_func,color='black')
    # plt.axvline(0,color='black',linestyle='--')
    # plt.show()
    return fit_func,popt[0]

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
    return fit_chrono,popt,pcov

#################################################

monkeys=['Niels','Galileo']
stage_vec=['early','mid','late']

nback=30

maxfev=100000000
p0l=(-20,20,-0.005,-3,500,700)
p0r=(-20,20,-0.005,3,700,500)
method='dogbox'

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])

beha_te_unte_all=nan*np.zeros((len(stage_vec),2,2,5,len(monkeys)))

for ss in range(len(stage_vec)):
    stage=stage_vec[ss]
    print ('#############################')
    print (stage)
        
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
        order=miscellaneous.order_files(files_pre)
        files_all=np.array(files_pre[order])
        #print (files_all)

        beha_te_unte=nan*np.zeros((2,2,len(files_groups)))
        
        for hh in range(len(files_groups)):
            files=files_all[files_groups[hh][0]:files_groups[hh][1]]
            #print (files)
            beha_tested_rlow=[]
            beha_tested_rhigh=[]
            beha_untested_rlow=[]
            beha_untested_rhigh=[]
            
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
                rt=beha['reaction_time'][1:]
                context_prepre=beha['context'] 
                ctx_ch=(context_prepre[1:]-context_prepre[0:-1])
                context_pre=context_prepre[1:]
                # Indices for first trial rewarded after change
                ind_ch_pre=np.where(abs(ctx_ch)==1)[0] # ind_ch_pre index where there is a context change
                ind_ch=miscellaneous.calculate_ind_ch_corr(ind_ch_pre,reward) # subjective 
                context=miscellaneous.create_context_subj(context_pre,ind_ch_pre,ind_ch) # CAREFUL! this is subjective context
                indch_ct01_pre=np.where(ctx_ch==1)[0]
                indch_ct10_pre=np.where(ctx_ch==-1)[0]
                ind_ch01_s0,ind_ch01_s1,ind_ch10_s0,ind_ch10_s1=miscellaneous.calculate_ind_ch_corr_stim(indch_ct01_pre,indch_ct10_pre,reward,stimulus)

                xx=np.array([100*coh_signed,choice]).T
    
                ##################################################
                # Behavior
                # Probability of Choice = Context for all possibilities: 01 0, 01 1, 10 0, 10 1
        
                #ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
                #for h in range(len(ind_ch01_s0)):
                for h in range(len(ind_ch01_s0)):
                    ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
                    ind_pre=(np.arange(nback)-nback+ind_ch01_s0[h]+1)
                    ind_used[ind_pre]=True
                    ind_used[np.isnan(rt)]=False
                    popt,pcov=func_fit_chrono(ind_used,xx,rt,coh_signed,coh_set_signed,maxfev,p0l,method)[1:]
                    rt_mean=chrono_curve(xx[(ind_ch01_s0[h]+1):(ind_ch01_s0[h]+2)],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])[0]
                    dev=(rt[ind_ch01_s0[h]+1]-rt_mean)
                    if stimulus[ind_ch01_s0[h]+1]==0: 
                        beha_tested_rlow.append(dev)
                    if stimulus[ind_ch01_s0[h]+1]==1:
                        beha_untested_rhigh.append(dev)
                   
                #ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
                #for h in range(len(ind_ch01_s1)):
                for h in range(len(ind_ch01_s1)):
                    ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
                    ind_pre=(np.arange(nback)-nback+ind_ch01_s1[h]+1)
                    ind_used[ind_pre]=True
                    ind_used[np.isnan(rt)]=False
                    popt,pcov=func_fit_chrono(ind_used,xx,rt,coh_signed,coh_set_signed,maxfev,p0l,method)[1:]
                    rt_mean=chrono_curve(xx[(ind_ch01_s1[h]+1):(ind_ch01_s1[h]+2)],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])[0]
                    dev=(rt[ind_ch01_s1[h]+1]-rt_mean)
                    if stimulus[ind_ch01_s1[h]+1]==0:
                        beha_untested_rlow.append(dev)
                    if stimulus[ind_ch01_s1[h]+1]==1:
                        beha_tested_rhigh.append(dev)
                 
                #ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
                #for h in range(len(ind_ch10_s0)):
                for h in range(len(ind_ch10_s0)):
                    ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
                    ind_pre=(np.arange(nback)-nback+ind_ch10_s0[h]+1)
                    ind_used[ind_pre]=True
                    ind_used[np.isnan(rt)]=False
                    popt,pcov=func_fit_chrono(ind_used,xx,rt,coh_signed,coh_set_signed,maxfev,p0r,method)[1:]
                    rt_mean=chrono_curve(xx[(ind_ch10_s0[h]+1):(ind_ch10_s0[h]+2)],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])[0]
                    dev=(rt[ind_ch10_s0[h]+1]-rt_mean)
                    if stimulus[ind_ch10_s0[h]+1]==0: 
                        beha_tested_rhigh.append(dev)
                    if stimulus[ind_ch10_s0[h]+1]==1:
                        beha_untested_rlow.append(dev)

                #ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
                #for h in range(len(ind_ch10_s1)):
                for h in range(len(ind_ch10_s1)):
                    ind_used=np.array(np.zeros(len(stimulus)),dtype=bool)
                    ind_pre=(np.arange(nback)-nback+ind_ch10_s1[h]+1)
                    ind_used[ind_pre]=True
                    ind_used[np.isnan(rt)]=False
                    popt,pcov=func_fit_chrono(ind_used,xx,rt,coh_signed,coh_set_signed,maxfev,p0r,method)[1:]
                    rt_mean=chrono_curve(xx[(ind_ch10_s1[h]+1):(ind_ch10_s1[h]+2)],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])[0]
                    dev=(rt[ind_ch10_s1[h]+1]-rt_mean)
                    if stimulus[ind_ch10_s1[h]+1]==0: 
                        beha_untested_rhigh.append(dev)
                    if stimulus[ind_ch10_s1[h]+1]==1:
                        beha_tested_rlow.append(dev)

            thresp=1e5 
            beha_tested_rlow=np.array(beha_tested_rlow)
            beha_tested_rlow=beha_tested_rlow[abs(beha_tested_rlow)<thresp]
            beha_tested_rhigh=np.array(beha_tested_rhigh)
            beha_tested_rhigh=beha_tested_rhigh[abs(beha_tested_rhigh)<thresp]
            beha_untested_rlow=np.array(beha_untested_rlow)
            beha_untested_rlow=beha_untested_rlow[abs(beha_untested_rlow)<thresp]
            beha_untested_rhigh=np.array(beha_untested_rhigh)
            beha_untested_rhigh=beha_untested_rhigh[abs(beha_untested_rhigh)<thresp]
            beha_te_unte[0,0,hh]=np.nanmean(beha_tested_rlow,axis=0)
            beha_te_unte[0,1,hh]=np.nanmean(beha_tested_rhigh,axis=0)
            beha_te_unte[1,0,hh]=np.nanmean(beha_untested_rlow,axis=0)
            beha_te_unte[1,1,hh]=np.nanmean(beha_untested_rhigh,axis=0)
            print (beha_tested_rlow)
            print (beha_tested_rhigh)
            print (beha_untested_rlow)
            print (beha_untested_rhigh)
            beha_te_unte_all[ss,0,0,hh,k]=np.nanmean(beha_tested_rlow,axis=0)
            beha_te_unte_all[ss,0,1,hh,k]=np.nanmean(beha_tested_rhigh,axis=0)
            beha_te_unte_all[ss,1,0,hh,k]=np.nanmean(beha_untested_rlow,axis=0)
            beha_te_unte_all[ss,1,1,hh,k]=np.nanmean(beha_untested_rhigh,axis=0)

####################################################
beha_te_unte_all[:,0,1]=-beha_te_unte_all[:,0,1]
beha_te_unte_all[:,1,1]=-beha_te_unte_all[:,1,1]

beha_def=np.reshape(beha_te_unte_all,(len(stage_vec),-1))
beha_def_m=np.nanmean(beha_def,axis=1)
beha_def_sem=sem(beha_def,axis=1,nan_policy='omit')

for i in range(len(stage_vec)):
    print ('Stage ',stage_vec[i])
    print (beha_def_m[i],beha_def_sem[i])
    print (scipy.stats.wilcoxon(beha_def[i],nan_policy='omit'))

fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.plot(np.arange(len(stage_vec)),beha_def_m,color='green')
ax.fill_between(np.arange(len(stage_vec)),beha_def_m-beha_def_sem,beha_def_m+beha_def_sem,color='green',alpha=0.5)
ax.plot(np.arange(len(stage_vec)),np.zeros(len(stage_vec)),color='black',linestyle='--')
ax.set_ylabel('$\Delta$Reaction Time')
ax.set_ylim([-150,250])
fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/inference_rt_both_learning.pdf',dpi=500,bbox_inches='tight')

