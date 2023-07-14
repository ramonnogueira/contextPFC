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

def log_curve(x,a,c):
    num=1+np.exp(-a*x+c)
    return 1/num

# def chrono_curve(x,a,c):
#     num=1+np.exp(-abs(a)*x+c)
#     return 1/num

#def chrono_curve(x,a,b,c,tl0,tr0):
#    return (b/(a*x+c))*np.tanh(a*x+c)+tl0*np.heaviside(-x,0)+tr0*np.heaviside(x,0)

def chrono_curve(x,a,b,c,t0):
  return (b/(a*x+c))*np.tanh(a*x+c)+t0

#def chrono_curve(x,b,t0):
#    return (b/x)*np.tanh(x)+t0

#def chrono_curve(x,a,c,t0):
#    return a*((x-c)**2)+t0

# def chrono_curve(x,a,c,t0):
#    fac1=1/(a*x+c)
#    fac2=np.tanh(a*x+c)
#    return fac1*fac2+t0

# def chrono_curve(x,a):
#    fac1=1/(a[0]*x+a[1])
#    fac2=np.tanh(a[0]*x+a[1])
#    return fac1*fac2+a[2]

#def chrono_curve(x,B,K,c,b,t0):
#    return B/(K*(c+b))*np.tanh(B*K*(c+b))+t0
    
#def chrono_curve(x,a,c,t0):
#    return np.exp(a*((x-c)**2))+t0

#################################################

monkey='Galileo'
phase='late'

abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/%s/%s/'%(phase,monkey) 
files=miscellaneous.order_files(np.array(os.listdir(abs_path)))

if monkey=='Niels':
    xx_coh_pre=np.array([-51.2,-25.6,-12.8,-6.4,-3.2,-1.6,0,1.6,3.2,6.4,12.8,25.6,51.2])
    xx_plot=np.array(['-51.2','-25.6','-12.8','-6.4','-3.2','-1.6','0','1.6','3.2','6.4','12.8','25.6','51.2'])
if monkey=='Galileo':
    xx_coh_pre=np.array([-51.2,-25.6,-12.8,-6.4,-4.5,-3.2,-1.6,0,1.6,3.2,4.5,6.4,12.8,25.6,51.2])
    xx_plot=np.array(['-51.2','-25.6','-12.8','-6.4','-4.5','-3.2','-1.6','0','1.6','3.2','4.5','6.4','12.8','25.6','51.2'])
xx_coh=np.log(abs(xx_coh_pre))
xx_coh[xx_coh<-100]=0
xx_coh[xx_coh_pre<0]=-1*xx_coh[xx_coh_pre<0]

psycho=nan*np.zeros((len(files),len(xx_coh),3))
chrono=nan*np.zeros((len(files),len(xx_coh),3))
fit_psycho=nan*np.zeros((len(files),len(xx_coh),3))
fit_chrono=nan*np.zeros((len(files),len(xx_coh),3))
    
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
    coh_log=np.log(abs(coh_signed))
    coh_log[coh_log<-100]=0
    coh_log[coh_signed<0]=-1*coh_log[coh_signed<0]
    coh_set_signed=np.unique(coh_signed)
    if monkey=='Niels':
        coh_set_signed=coh_set_signed[1:-1]
    
    context=beha['context']
    stimulus=beha['stimulus']
    choice=beha['choice']
    rt=beha['reaction_time']

    # Plot context per session
    rew_rat=np.array(context,dtype=np.float64)
    rew_rat[context==0]=0.77
    rew_rat[context==1]=1.3
    fig=plt.figure(figsize=(3.5,2))
    ax=fig.add_subplot(1,1,1)
    ax.scatter(np.arange(len(context))[context==0],rew_rat[context==0],color='green',label='Context Left',marker='s',s=0.25)
    ax.scatter(np.arange(len(context))[context==1],rew_rat[context==1],color='blue',label='Context Right',marker='s',s=0.25)
    adjust_spines(ax,['left','bottom'])
    ax.set_ylabel('Reward ratio (right/left)')
    ax.set_xlabel('Trials')
    ax.set_ylim([0.5,1.5])
    ax.set_yticks([0.5,1,1.5])
    plt.legend(loc='best')
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/context_across_trials_sess_%i_%s.pdf'%(kk,monkey),dpi=500,bbox_inches='tight')
    
    # Psychometric curve
    for i in range(len(coh_set_signed)):
        ind_coh=np.where(coh_signed==coh_set_signed[i])[0]
        ind_ct0=np.where(context[ind_coh]==0)[0]
        ind_ct1=np.where(context[ind_coh]==1)[0]
        psycho[kk,i,0]=np.mean(choice[ind_coh])
        psycho[kk,i,1]=np.mean(choice[ind_coh][ind_ct0])
        psycho[kk,i,2]=np.mean(choice[ind_coh][ind_ct1])

    # Fit Psychometric
    popt0,pcov0=curve_fit(log_curve,xx_coh,psycho[kk,:,0])
    fit_psycho[kk,:,0]=log_curve(xx_coh,popt0[0],popt0[1])
    popt1,pcov1=curve_fit(log_curve,xx_coh,psycho[kk,:,1])
    fit_psycho[kk,:,1]=log_curve(xx_coh,popt1[0],popt1[1])
    popt2,pcov2=curve_fit(log_curve,xx_coh,psycho[kk,:,2])
    fit_psycho[kk,:,2]=log_curve(xx_coh,popt2[0],popt2[1])
       
    # Compute chronometric curves
    for i in range(len(coh_set_signed)):
        ind_coh=np.where(coh_signed==coh_set_signed[i])[0]
        ind_ct0=np.where(context[ind_coh]==0)[0]
        ind_ct1=np.where(context[ind_coh]==1)[0]
        chrono[kk,i,0]=np.nanmean(rt[ind_coh])
        chrono[kk,i,1]=np.nanmean(rt[ind_coh][ind_ct0])
        chrono[kk,i,2]=np.nanmean(rt[ind_coh][ind_ct1])

    # Fit Chronometric
    # popt0,pcov0=curve_fit(chrono_curve,xx_coh,chrono[kk,:,0])#,p0=(1,1000))#,p0=(1,0,100,500))#,p0=(-1000,0,1000))
    # fit_chrono[kk,:,0]=chrono_curve(xx_coh,popt0[0],popt0[1],popt0[2],popt0[3])
    # popt1,pcov1=curve_fit(chrono_curve,xx_coh,chrono[kk,:,1])#,p0=(-1000,0,1000))
    # fit_chrono[kk,:,1]=chrono_curve(xx_coh,popt1[0],popt1[1],popt1[2],popt0[3])
    # popt2,pcov2=curve_fit(chrono_curve,xx_coh,chrono[kk,:,2])#,p0=(-1000,0,1000))
    # fit_chrono[kk,:,2]=chrono_curve(xx_coh,popt2[0],popt2[1],popt2[2],popt0[3])

    # popt0,pcov0=curve_fit(chrono_curve,xx_coh,np.log(rt))#,nan_policy='omit',p0=(1,0.1))
    # fit_chrono[kk,:,0]=np.exp(chrono_curve(xx_coh,popt0[0],popt0[1],popt0[2],popt0[3]))
    # print (popt0)
    # print (pcov0)
    # popt1,pcov1=curve_fit(chrono_curve,xx_coh[context==0],np.log(rt[context==0]))#,method='lm',nan_policy='omit')#,p0=(10,0.1,6))
    # fit_chrono[kk,:,1]=np.exp(chrono_curve(xx_coh,popt1[0],popt1[1],popt1[2],popt0[3]))
    # print (popt1)
    # print (pcov1)
    # popt2,pcov2=curve_fit(chrono_curve,xx_coh[context==1],np.log(rt[context==1]))#,method='lm',nan_policy='omit')#,p0=(10,0.1,6))
    # fit_chrono[kk,:,2]=np.exp(chrono_curve(xx_coh,popt2[0],popt2[1],popt2[2],popt0[3]))
    # print (popt2)
    # print (pcov2)
                      
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
ax.plot(xx_coh,fit_chrono_m[:,0],color='black',alpha=0.5)
ax.fill_between(xx_coh,fit_chrono_m[:,0]-fit_chrono_sem[:,0],fit_chrono_m[:,0]+fit_chrono_sem[:,0],color='black',alpha=0.5)
ax.plot(xx_coh,fit_chrono_m[:,1],color='green',linewidth=1,alpha=0.5)
ax.fill_between(xx_coh,fit_chrono_m[:,1]-fit_chrono_sem[:,1],fit_chrono_m[:,1]+fit_chrono_sem[:,1],color='green',alpha=0.5)
ax.plot(xx_coh,fit_chrono_m[:,2],color='blue',linewidth=1,alpha=0.5)
ax.fill_between(xx_coh,fit_chrono_m[:,2]-fit_chrono_sem[:,2],fit_chrono_m[:,2]+fit_chrono_sem[:,2],color='blue',alpha=0.5)
ax.scatter(xx_coh,chrono_m[:,0],color='black',s=3)
ax.scatter(xx_coh,chrono_m[:,1],color='green',s=3)#,alpha=0.5)
ax.scatter(xx_coh,chrono_m[:,2],color='blue',s=3)#,alpha=0.5)
ax.plot(xx_coh,chrono_m[:,0],color='black',label='All')
ax.fill_between(xx_coh,chrono_m[:,0]-chrono_sem[:,0],chrono_m[:,0]+chrono_sem[:,0],color='black',alpha=0.5)
ax.plot(xx_coh,chrono_m[:,1],color='green',label='Context Left')
ax.fill_between(xx_coh,chrono_m[:,1]-chrono_sem[:,1],chrono_m[:,1]+chrono_sem[:,1],color='green',alpha=0.5)
ax.plot(xx_coh,chrono_m[:,2],color='blue',label='Context Right')
ax.fill_between(xx_coh,chrono_m[:,2]-chrono_sem[:,2],chrono_m[:,2]+chrono_sem[:,2],color='blue',alpha=0.5)
ax.set_ylabel('Reaction Time (ms)')
ax.set_xlabel('Evidence Right (% Coherence)')
plt.legend(loc='best')
plt.xticks([-2.54,0,2.54],['-12.8','0','12.8'])
plt.yscale('log')
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/chronometric_monkey_%s.pdf'%monkey,dpi=500,bbox_inches='tight')


