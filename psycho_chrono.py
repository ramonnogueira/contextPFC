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


def trans_rew(x):
    rew=nan*np.zeros((len(x),2))
    for i in range(len(x)):
        rew[i]=x[i]
    return rew

def order_files(x):
    ord_pre=[]
    for i in range(len(x)):
        ord_pre.append(x[i][1:9])
    ord_pre=np.array(ord_pre)
    order=np.argsort(ord_pre)
    return order

def log_curve(x,a,c):
    num=1+np.exp(-a*x+c)
    return 1/num

def chrono_curve(x,b0,b1,b2,t0l,t0r): # x[:,0] is coherence and x[:,1] is choice
    return (b0/(x[:,0]+b2))*np.tanh(b1*(x[:,0]+b2))+(1-x[:,1])*t0l+x[:,1]*t0r#(1-x[:,1])*t0r+x[:,1]*t0l#

def chrono_curve2(x,Bl,Br,K,c_shift,t0l,t0r): # x[:,0] is coherence and x[:,1] is choice
    decl=(1-x[:,1])*(Bl/(K*(x[:,0]-c_shift)))*np.tanh(K*Bl*(x[:,0]-c_shift))
    decr=x[:,1]*(Br/(K*(x[:,0]-c_shift)))*np.tanh(K*Br*(x[:,0]-c_shift))
    tnd=(1-x[:,1])*t0l+x[:,1]*t0r
    return decl+decr+tnd


# def func_fit_chrono_log(ind_fit,xx,rt,coh_signed,coh_uq,maxfev,p0,method):
#     popt,pcov=curve_fit(chrono_curve,xx[ind_fit],np.log(rt[ind_fit]),maxfev=maxfev,p0=p0,method=method)
#     #print (popt)
#     yy=np.exp(chrono_curve(xx[ind_fit],popt[0],popt[1],popt[2],popt[3],popt[4]))
#     fit_chrono=nan*np.zeros(len(coh_uq))
#     for ii in range(len(coh_uq)):
#         fit_chrono[ii]=np.mean(yy[np.where(coh_signed[ind_fit]==coh_uq[ii])[0]])
#     return fit_chrono

def func_fit_chrono(ind_fit,xx,rt,coh_signed,coh_uq,maxfev,p0,method):
    popt,pcov,infodict,mesg,ier=curve_fit(chrono_curve,xx[ind_fit],rt[ind_fit],maxfev=maxfev,p0=p0,method=method,full_output=True)
    yy=chrono_curve(xx[ind_fit],popt[0],popt[1],popt[2],popt[3],popt[4])
    #print (popt)
    print ('Chrono1 LLH ',1-np.mean(infodict['fvec']**2)/np.var(rt[ind_fit]))
    fit_chrono=nan*np.zeros(len(coh_uq))
    for ii in range(len(coh_uq)):
        fit_chrono[ii]=np.mean(yy[np.where(coh_signed[ind_fit]==coh_uq[ii])[0]])
    return fit_chrono

def func_fit_chrono2(ind_fit,xx,rt,coh_signed,coh_uq,maxfev,p0,method):
    popt,pcov,infodict,mesg,ier=curve_fit(chrono_curve2,xx[ind_fit],rt[ind_fit],maxfev=maxfev,p0=p0,method=method,full_output=True)
    yy=chrono_curve2(xx[ind_fit],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
    #print (popt)
    print ('Chrono 2 LLH ',1-np.mean(infodict['fvec']**2)/np.var(rt[ind_fit]))
    fit_chrono=nan*np.zeros(len(coh_uq))
    for ii in range(len(coh_uq)):
        fit_chrono[ii]=np.mean(yy[np.where(coh_signed[ind_fit]==coh_uq[ii])[0]])
    return fit_chrono,popt

#################################################

monkey='Niels'
phase='late'

maxfev=100000
p100=(4000,0.1,0.1,500,500)
p101=(4000,0.1,-3,500,700)
p102=(4000,0.1,3,700,500)
p00=(-20,20,-0.005,0.1,500,500)
p01=(-20,20,-0.005,-3,500,700)
p02=(-20,20,-0.005,3,700,500)
#p0_log=(7,0.07,1,6,6)
method='lm'

#abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/%s/%s/'%(phase,monkey) 
#files=miscellaneous.order_files(np.array(os.listdir(abs_path)))
abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/unsorted/%s/'%(monkey) 
files_pre=np.array(os.listdir(abs_path))
order=order_files(files_pre)
files=np.array(files_pre[order])
print (files)

if monkey=='Niels':
    xx_coh_pre=np.array([-51.2,-25.6,-12.8,-6.4,-3.2,-1.6,0,1.6,3.2,6.4,12.8,25.6,51.2])
    xx_plot=np.array(['-51.2','-25.6','-12.8','-6.4','-3.2','-1.6','0','1.6','3.2','6.4','12.8','25.6','51.2'])
    #xx_coh_pre=np.array([-75,-51.2,-25.6,-12.8,-6.4,-3.2,-1.6,0,1.6,3.2,6.4,12.8,25.6,51.2,75])
    #xx_plot=np.array(['-75','-51.2','-25.6','-12.8','-6.4','-3.2','-1.6','0','1.6','3.2','6.4','12.8','25.6','51.2','75'])
if monkey=='Galileo':
    xx_coh_pre=np.array([-51.2,-25.6,-12.8,-6.4,-4.5,-3.2,-1.6,0,1.6,3.2,4.5,6.4,12.8,25.6,51.2])
    xx_plot=np.array(['-51.2','-25.6','-12.8','-6.4','-4.5','-3.2','-1.6','0','1.6','3.2','4.5','6.4','12.8','25.6','51.2'])
xx_coh=np.log(abs(xx_coh_pre))
xx_coh[xx_coh<-100]=0
xx_coh[xx_coh_pre<0]=-1*xx_coh[xx_coh_pre<0]

xx_fit=np.linspace(xx_coh_pre[0],xx_coh_pre[-1],1000)

psycho=nan*np.zeros((len(files),len(xx_coh),3))
fit_psycho=nan*np.zeros((len(files),len(xx_coh),3))
chrono=nan*np.zeros((len(files),len(xx_coh_pre),3))
fit_chrono=nan*np.zeros((len(files),len(xx_coh_pre),3))

params=nan*np.zeros((len(files),6,3))
    
for kk in range(len(files)):
    print (kk)
    #Load data
    data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
    beha=miscellaneous.behavior(data)
    change_ctx=beha['change_ctx']
    ind_ch=np.where(change_ctx!=0)[0]
    index_nonan=beha['index_nonan']
    reward=beha['reward']
    coh_signed=beha['coherence_signed']
    # coh_log=np.log(abs(100*coh_signed))
    # coh_log[coh_log<-100]=0
    # coh_log[coh_signed<0]=-1*coh_log[coh_signed<0]
    coh_set_signed=np.unique(coh_signed)
    if monkey=='Niels':
        coh_set_signed=coh_set_signed[1:-1]
    
    context=beha['context']
    stimulus=beha['stimulus']
    choice=beha['choice']
    rt=beha['reaction_time']

    # Plot context per session
    # rew_rat=np.array(context,dtype=np.float64)
    # rew_rat[context==0]=0.77
    # rew_rat[context==1]=1.3
    # fig=plt.figure(figsize=(3.5,2))
    # ax=fig.add_subplot(1,1,1)
    # ax.scatter(np.arange(len(context))[context==0],rew_rat[context==0],color='green',label='Context Left',marker='s',s=0.25)
    # ax.scatter(np.arange(len(context))[context==1],rew_rat[context==1],color='blue',label='Context Right',marker='s',s=0.25)
    # adjust_spines(ax,['left','bottom'])
    # ax.set_ylabel('Reward ratio (right/left)')
    # ax.set_xlabel('Trials')
    # ax.set_ylim([0.5,1.5])
    # ax.set_yticks([0.5,1,1.5])
    # plt.legend(loc='best')
    # fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/context_across_trials_sess_%i_%s.pdf'%(kk,monkey),dpi=500,bbox_inches='tight')
    
    # Psychometric curve
    # for i in range(len(coh_set_signed)):
    #     ind_coh=np.where(coh_signed==coh_set_signed[i])[0]
    #     ind_ct0=np.where(context[ind_coh]==0)[0]
    #     ind_ct1=np.where(context[ind_coh]==1)[0]
    #     psycho[kk,i,0]=np.mean(choice[ind_coh])
    #     psycho[kk,i,1]=np.mean(choice[ind_coh][ind_ct0])
    #     psycho[kk,i,2]=np.mean(choice[ind_coh][ind_ct1])

    # # Fit Psychometric
    # popt0,pcov0=curve_fit(log_curve,xx_coh,psycho[kk,:,0])
    # fit_psycho[kk,:,0]=log_curve(xx_coh,popt0[0],popt0[1])
    # popt1,pcov1=curve_fit(log_curve,xx_coh,psycho[kk,:,1])
    # fit_psycho[kk,:,1]=log_curve(xx_coh,popt1[0],popt1[1])
    # popt2,pcov2=curve_fit(log_curve,xx_coh,psycho[kk,:,2])
    # fit_psycho[kk,:,2]=log_curve(xx_coh,popt2[0],popt2[1])
       
    # Compute chronometric curves
    for i in range(len(coh_set_signed)):
        ind_coh=np.where(coh_signed==coh_set_signed[i])[0]
        ind_ct0=np.where(context[ind_coh]==0)[0]
        ind_ct1=np.where(context[ind_coh]==1)[0]
        chrono[kk,i,0]=np.nanmean(rt[ind_coh])
        chrono[kk,i,1]=np.nanmean(rt[ind_coh][ind_ct0])
        chrono[kk,i,2]=np.nanmean(rt[ind_coh][ind_ct1])
   
    # Fit Chronometric
    xx=np.array([100*coh_signed,choice]).T
    ind_fit0=(~np.isnan(rt))*(abs(coh_signed)<0.75)
    func_fit_chrono(ind_fit0,xx,rt,coh_signed,coh_set_signed,maxfev,p100,method)
    ff0=func_fit_chrono2(ind_fit0,xx,rt,coh_signed,coh_set_signed,maxfev,p00,method)
    fit_chrono[kk,:,0]=ff0[0]
    params[kk,:,0]=ff0[1]
    
    ind_fit1=(context==0)*(~np.isnan(rt))*(abs(coh_signed)<0.75)
    func_fit_chrono(ind_fit1,xx,rt,coh_signed,coh_set_signed,maxfev,p101,method)
    ff1=func_fit_chrono2(ind_fit1,xx,rt,coh_signed,coh_set_signed,maxfev,p01,method)
    fit_chrono[kk,:,1]=ff1[0]
    params[kk,:,1]=ff1[1]

    ind_fit2=(context==1)*(~np.isnan(rt))*(abs(coh_signed)<0.75)
    func_fit_chrono(ind_fit2,xx,rt,coh_signed,coh_set_signed,maxfev,p102,method)
    ff2=func_fit_chrono2(ind_fit2,xx,rt,coh_signed,coh_set_signed,maxfev,p02,method)
    fit_chrono[kk,:,2]=ff2[0]
    params[kk,:,2]=ff2[1]
   
    # plt.plot(xx_coh_pre,chrono[kk,:,0],color='black')
    # plt.plot(xx_coh_pre,fit_chrono[kk,:,0],color='black',linestyle='--')
    # plt.plot(xx_coh_pre,chrono[kk,:,1],color='green')
    # plt.plot(xx_coh_pre,fit_chrono[kk,:,1],color='green',linestyle='--')
    # plt.plot(xx_coh_pre,chrono[kk,:,2],color='blue')
    # plt.plot(xx_coh_pre,fit_chrono[kk,:,2],color='blue',linestyle='--')
    # plt.show()

#################################################################
# Plot Psychometric
psycho_m=np.mean(psycho,axis=0)
psycho_sem=sem(psycho,axis=0)
fit_psycho_m=np.mean(fit_psycho,axis=0)
fit_psycho_sem=sem(fit_psycho,axis=0)

fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(1,1,1)
adjust_spines(ax,['left','bottom'])
ax.plot(xx_coh,fit_psycho_m[:,0],color='black',label='All',alpha=0.5)
ax.fill_between(xx_coh,fit_psycho_m[:,0]-fit_psycho_sem[:,0],fit_psycho_m[:,0]+fit_psycho_sem[:,0],color='black',alpha=0.5)
ax.plot(xx_coh,fit_psycho_m[:,1],color='green',label='Context Left',linewidth=1,alpha=0.5)
ax.fill_between(xx_coh,fit_psycho_m[:,1]-fit_psycho_sem[:,1],fit_psycho_m[:,1]+fit_psycho_sem[:,1],color='green',alpha=0.5)
ax.plot(xx_coh,fit_psycho_m[:,2],color='blue',label='Context Right',linewidth=1,alpha=0.5)
ax.fill_between(xx_coh,fit_psycho_m[:,2]-fit_psycho_sem[:,2],fit_psycho_m[:,2]+fit_psycho_sem[:,2],color='blue',alpha=0.5)
ax.scatter(xx_coh,psycho_m[:,0],color='black',s=3)
ax.scatter(xx_coh,psycho_m[:,1],color='green',s=3)#,alpha=0.5)
ax.scatter(xx_coh,psycho_m[:,2],color='blue',s=3)#,alpha=0.5)
ax.plot(xx_coh,0.5*np.ones(len(xx_coh)),color='black',linestyle='--')
ax.set_ylabel('Probability Right Response')
ax.axvline(0,color='black',linestyle='--')
ax.set_xlabel('Evidence Right (% Coherence)')
ax.set_ylim([-0.05,1.05])
plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
#plt.legend(loc='best')
plt.xticks([-2.54,0,2.54],['-12.8','0','12.8'])
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/psychometric_monkey_%s.pdf'%monkey,dpi=500,bbox_inches='tight')

#########################################
# Plot Chronometric
chrono_m=np.mean(chrono,axis=0)
chrono_sem=sem(chrono,axis=0)
fit_chrono_m=np.mean(fit_chrono,axis=0)
fit_chrono_sem=sem(fit_chrono,axis=0)

fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(1,1,1)
adjust_spines(ax,['left','bottom'])
ax.plot(xx_coh_pre,fit_chrono_m[:,0],color='black',alpha=0.5)
ax.fill_between(xx_coh_pre,fit_chrono_m[:,0]-fit_chrono_sem[:,0],fit_chrono_m[:,0]+fit_chrono_sem[:,0],color='black',alpha=0.5)
ax.plot(xx_coh_pre,fit_chrono_m[:,1],color='green',linewidth=1,alpha=0.5)
ax.fill_between(xx_coh_pre,fit_chrono_m[:,1]-fit_chrono_sem[:,1],fit_chrono_m[:,1]+fit_chrono_sem[:,1],color='green',alpha=0.5)
ax.plot(xx_coh_pre,fit_chrono_m[:,2],color='blue',linewidth=1,alpha=0.5)
ax.fill_between(xx_coh_pre,fit_chrono_m[:,2]-fit_chrono_sem[:,2],fit_chrono_m[:,2]+fit_chrono_sem[:,2],color='blue',alpha=0.5)
ax.scatter(xx_coh_pre,chrono_m[:,0],color='black',s=3)
ax.scatter(xx_coh_pre,chrono_m[:,1],color='green',s=3)#,alpha=0.5)
ax.scatter(xx_coh_pre,chrono_m[:,2],color='blue',s=3)#,alpha=0.5)
ax.set_ylabel('Reaction Time (ms)')
ax.set_xlabel('Evidence Right (% Coherence)')
#plt.legend(loc='best')
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/chronometric_monkey_%s.pdf'%monkey,dpi=500,bbox_inches='tight')

