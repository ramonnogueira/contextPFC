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
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from scipy.optimize import curve_fit
import datetime
import miscellaneous
from brokenaxes import brokenaxes

nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
plt.close('all')

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

def log_curve(x,a,c):
    num=1+np.exp(-a*(x+c))
    return 1/num

# def chrono_curve(x,b0,b1,b2,t0l,t0r): # x[:,0] is coherence and x[:,1] is choice
#     return (b0/(x[:,0]+b2))*np.tanh(b1*(x[:,0]+b2))+(1-x[:,1])*t0l+x[:,1]*t0r#(1-x[:,1])*t0r+x[:,1]*t0l#

def chrono_curve(x,Bl,Br,K,c_shift,t0l,t0r): # x[:,0] is coherence and x[:,1] is choice
    decl=(1-x[:,1])*(Bl/(K*(x[:,0]+c_shift)))*np.tanh(K*Bl*(x[:,0]+c_shift))
    decr=x[:,1]*(Br/(K*(x[:,0]+c_shift)))*np.tanh(K*Br*(x[:,0]+c_shift))
    tnd=(1-x[:,1])*t0l+x[:,1]*t0r
    return decl+decr+tnd

def func_fit_chrono(xx,rt,params_psy,n_gen,maxfev,p0,method):
    popt,pcov,infodict,mesg,ier=curve_fit(chrono_curve,xx,rt,maxfev=maxfev,p0=p0,method=method,full_output=True)
    
    coh_uq=np.unique(xx[:,0])
    xx_gen_pre=np.linspace(coh_uq[0],coh_uq[-1],n_gen)
    
    n_inst=10000
    yy=nan*np.zeros((n_inst,len(xx_gen_pre)))
    for i in range(n_inst): #loop on instances
        p_inst=log_curve(xx_gen_pre,params_psy[0],params_psy[1])
        choice_inst=np.random.binomial(n=1,size=len(p_inst),p=p_inst)
        xx_gen=np.array([xx_gen_pre,choice_inst]).T
        yy[i]=chrono_curve(xx_gen,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
    
    return np.mean(yy,axis=0),popt

#################################################

monkeys=['Niels','Galileo']

maxfev=100000
p00_psy=(0.2,0)
p01_psy=(0.2,-3)
p02_psy=(0.2,3)
p00_chr=(-20,20,-0.005,-0.1,500,500)
p01_chr=(-20,20,-0.005,3,500,700)
p02_chr=(-20,20,-0.005,-3,700,500)
method='lm'

n_gen=1000

# psycho_all=nan*np.zeros((14,15,3))
# fit_psycho_all=nan*np.zeros((14,n_gen,3))
# chrono_all=nan*np.zeros((14,15,3))
# fit_chrono_all=nan*np.zeros((14,n_gen,3))
# params_psy_all=nan*np.zeros((14,2,3))
# params_rt_all=nan*np.zeros((14,6,3))
psycho_all=nan*np.zeros((42,15,3))
fit_psycho_all=nan*np.zeros((42,n_gen,3))
chrono_all=nan*np.zeros((42,15,3))
fit_chrono_all=nan*np.zeros((42,n_gen,3))
params_psy_all=nan*np.zeros((42,2,3))
params_rt_all=nan*np.zeros((42,6,3))

xx_coh_all=np.array([-51.2,-25.6,-12.8,-6.4,-4.5,-3.2,-1.6,0,1.6,3.2,4.5,6.4,12.8,25.6,51.2])

group_coh=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7])

uu=-1
for k in range(len(monkeys)):
    if monkeys[k]=='Niels':
        xx_coh=np.array([-51.2,-25.6,-12.8,-6.4,-3.2,-1.6,0,1.6,3.2,6.4,12.8,25.6,51.2])
        xx_plot=np.array(['-51.2','-25.6','-12.8','-6.4','-3.2','-1.6','0','1.6','3.2','6.4','12.8','25.6','51.2'])
        fused=-12 # Careful!
        ind_uu=np.array([0,1,2,3,5,6,7,8,9,11,12,13,14])
    if monkeys[k]=='Galileo':
        xx_coh=np.array([-51.2,-25.6,-12.8,-6.4,-4.5,-3.2,-1.6,0,1.6,3.2,4.5,6.4,12.8,25.6,51.2])
        xx_plot=np.array(['-51.2','-25.6','-12.8','-6.4','-4.5','-3.2','-1.6','0','1.6','3.2','4.5','6.4','12.8','25.6','51.2'])
        fused=-30 # Careful!
        ind_uu=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    xx_fit=np.linspace(xx_coh[0],xx_coh[-1],n_gen)

    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/unsorted/%s/'%(monkeys[k]) 
    files_pre=np.array(os.listdir(abs_path))
    order=order_files(files_pre)
    files=np.array(files_pre[order])[fused:]
    print (files_pre[order])
    print (files)

    psycho=nan*np.zeros((len(files),len(xx_coh),3))
    fit_psycho=nan*np.zeros((len(files),n_gen,3))
    chrono=nan*np.zeros((len(files),len(xx_coh),3))
    fit_chrono=nan*np.zeros((len(files),n_gen,3))
    params_psy=nan*np.zeros((len(files),2,3))
    params_rt=nan*np.zeros((len(files),6,3))
    for kk in range(len(files)):
        uu+=1
        print (files[kk],uu)
        #Load data
        data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
        beha=miscellaneous.behavior(data,group_coh)
        change_ctx=beha['change_ctx']
        ind_ch=np.where(change_ctx!=0)[0]
        index_nonan=beha['index_nonan']
        reward=beha['reward']
        coh_signed=beha['coherence_signed']
        coh_set_signed=np.unique(coh_signed)
        if monkeys[k]=='Niels':
            coh_set_signed=coh_set_signed[1:-1]
        context=beha['context']
        stimulus=beha['stimulus']
        choice=beha['choice']
        rt=beha['reaction_time']
    
        # Psychometric curve
        for i in range(len(coh_set_signed)):
            ind_coh=np.where(coh_signed==coh_set_signed[i])[0]
            ind_ct0=np.where(context[ind_coh]==0)[0]
            ind_ct1=np.where(context[ind_coh]==1)[0]
            psycho[kk,i,0]=np.mean(choice[ind_coh])
            psycho[kk,i,1]=np.mean(choice[ind_coh][ind_ct0])
            psycho[kk,i,2]=np.mean(choice[ind_coh][ind_ct1])

        # Fit Psychometric
        popt0,pcov0=curve_fit(log_curve,100*coh_set_signed,psycho[kk,:,0],maxfev=maxfev,p0=p00_psy,method=method)
        fit_psycho[kk,:,0]=log_curve(xx_fit,popt0[0],popt0[1])
        params_psy[kk,:,0]=popt0
        popt1,pcov1=curve_fit(log_curve,100*coh_set_signed,psycho[kk,:,1],maxfev=maxfev,p0=p01_psy,method=method)
        fit_psycho[kk,:,1]=log_curve(xx_fit,popt1[0],popt1[1])
        params_psy[kk,:,1]=popt1
        popt2,pcov2=curve_fit(log_curve,100*coh_set_signed,psycho[kk,:,2],maxfev=maxfev,p0=p02_psy,method=method)
        fit_psycho[kk,:,2]=log_curve(xx_fit,popt2[0],popt2[1])
        params_psy[kk,:,2]=popt2

        print (params_psy[kk])

        #########################################
        # Compute chronometric curves
        for i in range(len(coh_set_signed)):
            ind_coh=np.where(coh_signed==coh_set_signed[i])[0]
            ind_ct0=np.where(context[ind_coh]==0)[0]
            ind_ct1=np.where(context[ind_coh]==1)[0]
            chrono[kk,i,0]=np.nanmean(rt[ind_coh])
            chrono[kk,i,1]=np.nanmean(rt[ind_coh][ind_ct0])
            chrono[kk,i,2]=np.nanmean(rt[ind_coh][ind_ct1])
        
        # # Fit Chronometric
        xx=np.array([100*coh_signed,choice]).T
        ind_fit0=(~np.isnan(rt))*(abs(coh_signed)<0.75)
        ff0=func_fit_chrono(xx[ind_fit0],rt[ind_fit0],params_psy[kk,:,0],n_gen,maxfev,p00_chr,method)
        fit_chrono[kk,:,0]=ff0[0]
        params_rt[kk,:,0]=ff0[1]
        ind_fit1=(context==0)*(~np.isnan(rt))*(abs(coh_signed)<0.75)
        ff1=func_fit_chrono(xx[ind_fit1],rt[ind_fit1],params_psy[kk,:,1],n_gen,maxfev,p01_chr,method)
        fit_chrono[kk,:,1]=ff1[0]
        params_rt[kk,:,1]=ff1[1]
        ind_fit2=(context==1)*(~np.isnan(rt))*(abs(coh_signed)<0.75)
        ff2=func_fit_chrono(xx[ind_fit2],rt[ind_fit2],params_psy[kk,:,2],n_gen,maxfev,p02_chr,method)
        fit_chrono[kk,:,2]=ff2[0]
        params_rt[kk,:,2]=ff2[1]

        #######################################
        # All
        if monkeys[k]=='Niels':
            ff=14*5/40
        if monkeys[k]=='Galileo':
            ff=14*2/40
        ff=1
        psycho_all[uu,ind_uu]=ff*psycho[kk]
        fit_psycho_all[uu]=ff*fit_psycho[kk]
        params_psy_all[uu]=ff*params_psy[kk]
        chrono_all[uu,ind_uu]=ff*chrono[kk]
        fit_chrono_all[uu]=ff*fit_chrono[kk]
        params_rt_all[uu]=ff*params_rt[kk]

        
    #################################################################
    # Plot Psychometric
    psycho_m=np.mean(psycho,axis=0)
    psycho_sem=sem(psycho,axis=0)
    fit_psycho_m=np.mean(fit_psycho,axis=0)
    fit_psycho_sem=sem(fit_psycho,axis=0)
    
    fig=plt.figure(figsize=(2.3,2))
    bax=brokenaxes(xlims=((-55,-50),(-27,27),(50,55)),hspace=0.1)
    bax.plot(xx_fit,fit_psycho_m[:,0],color='black',label='All',alpha=0.5)
    bax.fill_between(xx_fit,fit_psycho_m[:,0]-fit_psycho_sem[:,0],fit_psycho_m[:,0]+fit_psycho_sem[:,0],color='black',alpha=0.5)
    bax.plot(xx_fit,fit_psycho_m[:,1],color='green',label='Context Left',linewidth=1,alpha=0.5)
    bax.fill_between(xx_fit,fit_psycho_m[:,1]-fit_psycho_sem[:,1],fit_psycho_m[:,1]+fit_psycho_sem[:,1],color='green',alpha=0.5)
    bax.plot(xx_fit,fit_psycho_m[:,2],color='blue',label='Context Right',linewidth=1,alpha=0.5)
    bax.fill_between(xx_fit,fit_psycho_m[:,2]-fit_psycho_sem[:,2],fit_psycho_m[:,2]+fit_psycho_sem[:,2],color='blue',alpha=0.5)
    bax.scatter(xx_coh,psycho_m[:,0],color='black',s=3)
    bax.scatter(xx_coh,psycho_m[:,1],color='green',s=3)#,alpha=0.5)
    bax.scatter(xx_coh,psycho_m[:,2],color='blue',s=3)#,alpha=0.5)
    bax.plot(xx_coh,0.5*np.ones(len(xx_coh)),color='black',linestyle='--')
    bax.set_ylabel('Probability Right Response')
    bax.axvline(0,color='black',linestyle='--')
    bax.set_xlabel('Evidence Right (% Coherence)')
    bax.set_ylim([-0.05,1.05])
    bax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    #bax.set_xticks([-0.5,-0.2,0,0.2,0.5])
    #plt.xticks([-2.54,0,2.54],['-12.8','0','12.8'])
    #fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/psychometric_monkey_%s_def.pdf'%monkeys[k],dpi=500,bbox_inches='tight')
    
    #########################################
    # Plot Chronometric
    chrono_m=np.mean(chrono,axis=0)
    chrono_sem=sem(chrono,axis=0)
    fit_chrono_m=np.mean(fit_chrono,axis=0)
    fit_chrono_sem=sem(fit_chrono,axis=0)
    
    fig=plt.figure(figsize=(2.3,2))
    ax=fig.add_subplot(1,1,1)
    adjust_spines(ax,['left','bottom'])
    ax.plot(xx_fit,fit_chrono_m[:,0],color='black',alpha=0.5)
    ax.fill_between(xx_fit,fit_chrono_m[:,0]-fit_chrono_sem[:,0],fit_chrono_m[:,0]+fit_chrono_sem[:,0],color='black',alpha=0.5)
    ax.plot(xx_fit,fit_chrono_m[:,1],color='green',linewidth=1,alpha=0.5)
    ax.fill_between(xx_fit,fit_chrono_m[:,1]-fit_chrono_sem[:,1],fit_chrono_m[:,1]+fit_chrono_sem[:,1],color='green',alpha=0.5)
    ax.plot(xx_fit,fit_chrono_m[:,2],color='blue',linewidth=1,alpha=0.5)
    ax.fill_between(xx_fit,fit_chrono_m[:,2]-fit_chrono_sem[:,2],fit_chrono_m[:,2]+fit_chrono_sem[:,2],color='blue',alpha=0.5)
    ax.scatter(xx_coh,chrono_m[:,0],color='black',s=3)
    ax.scatter(xx_coh,chrono_m[:,1],color='green',s=3)#,alpha=0.5)
    ax.scatter(xx_coh,chrono_m[:,2],color='blue',s=3)#,alpha=0.5)
    ax.set_ylabel('Reaction Time (ms)')
    ax.set_xlabel('Evidence Right (% Coherence)')
    #plt.legend(loc='best')
    #fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/chronometric_monkey_%s_def.pdf'%monkeys[k],dpi=500,bbox_inches='tight')

############################################
# Psycho all
# psycho_all_m=np.nanmean(psycho_all,axis=0)
# psycho_all_sem=sem(psycho_all,axis=0,nan_policy='omit')
# fit_psycho_all_m=np.nanmean(fit_psycho_all,axis=0)
# fit_psycho_all_sem=sem(fit_psycho_all,axis=0,nan_policy='omit')

psycho_all_m=np.nanmean(np.array([np.mean(psycho_all[0:4],axis=0),np.mean(psycho_all[4:],axis=0)]),axis=0)
psycho_all_sem=sem(np.array([np.mean(psycho_all[0:4],axis=0),np.mean(psycho_all[4:],axis=0)]),axis=0,nan_policy='omit')
fit_psycho_all_m=np.nanmean(np.array([np.mean(fit_psycho_all[0:4],axis=0),np.mean(fit_psycho_all[4:],axis=0)]),axis=0)
fit_psycho_all_sem=sem(np.array([np.mean(fit_psycho_all[0:4],axis=0),np.mean(fit_psycho_all[4:],axis=0)]),axis=0,nan_policy='omit')

fig=plt.figure(figsize=(2.3,2))
bax=brokenaxes(xlims=((-55,-50),(-27,27),(50,55)),hspace=0.1)
#adjust_spines(bax,['left','bottom'])
bax.plot(xx_fit,fit_psycho_all_m[:,0],color='black',label='All',alpha=0.5)
bax.fill_between(xx_fit,fit_psycho_all_m[:,0]-fit_psycho_all_sem[:,0],fit_psycho_all_m[:,0]+fit_psycho_all_sem[:,0],color='black',alpha=0.5)
bax.plot(xx_fit,fit_psycho_all_m[:,1],color='green',label='Context Left',linewidth=1,alpha=0.5)
bax.fill_between(xx_fit,fit_psycho_all_m[:,1]-fit_psycho_all_sem[:,1],fit_psycho_all_m[:,1]+fit_psycho_all_sem[:,1],color='green',alpha=0.5)
bax.plot(xx_fit,fit_psycho_all_m[:,2],color='blue',label='Context Right',linewidth=1,alpha=0.5)
bax.fill_between(xx_fit,fit_psycho_all_m[:,2]-fit_psycho_all_sem[:,2],fit_psycho_all_m[:,2]+fit_psycho_all_sem[:,2],color='blue',alpha=0.5)
bax.scatter(xx_coh_all,psycho_all_m[:,0],color='black',s=3)
bax.scatter(xx_coh_all,psycho_all_m[:,1],color='green',s=3)#,alpha=0.5)
bax.scatter(xx_coh_all,psycho_all_m[:,2],color='blue',s=3)#,alpha=0.5)
bax.plot(xx_coh_all,0.5*np.ones(len(xx_coh_all)),color='black',linestyle='--')
bax.set_ylabel('Probability Right Response')
bax.axvline(0,color='black',linestyle='--')
bax.set_xlabel('Evidence Right (% Coherence)')
bax.set_ylim([-0.05,1.05])
bax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
#bax.set_xticks([-0.5,-0.2,0,0.2,0.5])
#plt.xticks([-2.54,0,2.54],['-12.8','0','12.8'])
#fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/psychometric_both_def.pdf',dpi=500,bbox_inches='tight')

# Chronometric All
chrono_all_m=np.nanmean(chrono_all,axis=0)
chrono_all_sem=sem(chrono_all,axis=0,nan_policy='omit')
fit_chrono_all_m=np.nanmean(fit_chrono_all,axis=0)
fit_chrono_all_sem=sem(fit_chrono_all,axis=0,nan_policy='omit')

# chrono_all_m=np.nanmean(np.array([np.mean(chrono_all[0:4],axis=0),np.mean(chrono_all[4:],axis=0)]),axis=0)
# chrono_all_sem=sem(np.array([np.mean(chrono_all[0:4],axis=0),np.mean(chrono_all[4:],axis=0)]),axis=0,nan_policy='omit')
# fit_chrono_all_m=np.nanmean(np.array([np.mean(fit_chrono_all[0:4],axis=0),np.mean(fit_chrono_all[4:],axis=0)]),axis=0)
# fit_chrono_all_sem=sem(np.array([np.mean(fit_chrono_all[0:4],axis=0),np.mean(fit_chrono_all[4:],axis=0)]),axis=0,nan_policy='omit')

fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(1,1,1)
adjust_spines(ax,['left','bottom'])
ax.plot(xx_fit,fit_chrono_all_m[:,0],color='black',alpha=0.5)
ax.fill_between(xx_fit,fit_chrono_all_m[:,0]-fit_chrono_all_sem[:,0],fit_chrono_all_m[:,0]+fit_chrono_all_sem[:,0],color='black',alpha=0.5)
ax.plot(xx_fit,fit_chrono_all_m[:,1],color='green',linewidth=1,alpha=0.5)
ax.fill_between(xx_fit,fit_chrono_all_m[:,1]-fit_chrono_all_sem[:,1],fit_chrono_all_m[:,1]+fit_chrono_all_sem[:,1],color='green',alpha=0.5)
ax.plot(xx_fit,fit_chrono_all_m[:,2],color='blue',linewidth=1,alpha=0.5)
ax.fill_between(xx_fit,fit_chrono_all_m[:,2]-fit_chrono_all_sem[:,2],fit_chrono_all_m[:,2]+fit_chrono_all_sem[:,2],color='blue',alpha=0.5)
ax.scatter(xx_coh_all,chrono_all_m[:,0],color='black',s=3)
ax.scatter(xx_coh_all,chrono_all_m[:,1],color='green',s=3)#,alpha=0.5)
ax.scatter(xx_coh_all,chrono_all_m[:,2],color='blue',s=3)#,alpha=0.5)
ax.set_ylabel('Reaction Time (ms)')
ax.set_xlabel('Evidence Right (% Coherence)')
#plt.legend(loc='best')
#fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/chronometric_both_def2.pdf',dpi=500,bbox_inches='tight')
