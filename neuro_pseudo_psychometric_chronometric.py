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
from scipy.optimize import curve_fit
#from numba import jit
import miscellaneous

nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def log_curve(x,a,c):
    num=1+np.exp(-a*x+c)
    return 1/num
    
def log_curve_abs(x,a,c):
    num=1+np.exp(-a*abs(x)+c)
    return 1/num

###################################################3

monkey='Niels'#['Galileo']#'Niels']#,]

talig='dots_on' #'targ_on','dots_on'
dic_time=np.array([0,600,200,200]) # time pre, time post, bin size, step size
steps=int((dic_time[0]+dic_time[1])/dic_time[3])
xx=np.linspace(-dic_time[0]/1000,dic_time[1]/1000,steps,endpoint=False)

nt=100 #100 for coh signed, 200 for coh unsigned, 50 for coh signed with context
n_rand=50
n_shuff=0
perc_tr=0.8
thres=0
reg=1e2
n_coh=15
tpre_sacc=50

if monkey=='Niels':
    xx_coh_pre=np.array([-75,-51.2,-25.6,-12.8,-6.4,-3.2,-1.6,0,1.6,3.2,6.4,12.8,25.6,51.2,75])
    xx_plot=np.array(['-75','-51.2','-25.6','-12.8','-6.4','-3.2','-1.6','0','1.6','3.2','6.4','12.8','25.6','51.2','75'])
if monkey=='Galileo':
    xx_coh_pre=np.array([-51.2,-25.6,-12.8,-6.4,-4.5,-3.2,-1.6,0,1.6,3.2,4.5,6.4,12.8,25.6,51.2])
    xx_plot=np.array(['-51.2','-25.6','-12.8','-6.4','-4.5','-3.2','-1.6','0','1.6','3.2','4.5','6.4','12.8','25.6','51.2'])
xx_coh=np.log(abs(xx_coh_pre))
xx_coh[xx_coh<-100]=0
xx_coh[xx_coh_pre<0]=-1*xx_coh[xx_coh_pre<0]

print (xx_coh)

group_coh=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7])
    
abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/late/%s/'%(monkey)
files=os.listdir(abs_path)

chrono_neuro_pre=nan*np.zeros((steps,n_rand,n_coh,3))
psycho_neuro_pre=nan*np.zeros((steps,n_rand,n_coh,3))
fit_psycho_neuro_pre=nan*np.zeros((steps,n_rand,n_coh,3))
chrono_neuro_flat_pre=nan*np.zeros((n_rand,n_coh,3))
psycho_neuro_flat_pre=nan*np.zeros((n_rand,n_coh,3))
fit_psycho_neuro_flat_pre=nan*np.zeros((n_rand,n_coh,3))
for ii in range(n_rand):
    print (ii)
    # Careful! in this function I am only using correct trials so that choice and stimulus are the same
    pseudo=miscellaneous.pseudopop_coherence_context_correct(abs_path,files,talig,dic_time,steps,thres,nt,1,perc_tr,True,tpre_sacc,group_coh,shuff=False,learning=False) #This!
    pseudo_tr_pre=pseudo['pseudo_tr']
    pseudo_te_pre=pseudo['pseudo_te']
    neu_total=pseudo_tr_pre.shape[-1]
    clase_all_pre=pseudo['clase_all']
    clase_coh_pre=pseudo['clase_coh']
    indc0=np.where(clase_coh_pre==7)[0]
    clase_ctx_pre=pseudo['clase_ctx']
    stim_pre=nan*np.zeros(len(clase_coh_pre))
    stim_pre[clase_coh_pre>7]=1
    stim_pre[clase_coh_pre<7]=0
    stim_pre[indc0]=np.array(np.random.normal(0,1,len(indc0))>0,dtype=np.int16)
    
    ##########################################
    # Psychometric for all time steps together
    pseudo_tr_flat_pre=nan*np.zeros((pseudo_tr_pre.shape[2],steps*neu_total))
    pseudo_te_flat_pre=nan*np.zeros((pseudo_te_pre.shape[2],steps*neu_total))
    for pp in range(steps):
        pseudo_tr_flat_pre[:,pp*neu_total:(pp+1)*neu_total]=pseudo_tr_pre[pp,0]
        pseudo_te_flat_pre[:,pp*neu_total:(pp+1)*neu_total]=pseudo_te_pre[pp,0]

    sum_nan=np.sum(np.isnan(pseudo_tr_flat_pre),axis=1)
    index_nonan=np.where(sum_nan==0)[0]
    pseudo_tr_flat=pseudo_tr_flat_pre[index_nonan]
    pseudo_te_flat=pseudo_te_flat_pre[index_nonan]
    clase_all=clase_all_pre[index_nonan]
    clase_coh=clase_coh_pre[index_nonan]
    clase_ctx=clase_ctx_pre[index_nonan]
    stim=stim_pre[index_nonan]
                
    cl=LogisticRegression(C=1/reg,class_weight='balanced')
    cl.fit(pseudo_tr_flat,stim) # Cuidado con decode Stim y no Choice!
                    
    for j in range(n_coh):
        #print ('  ',j)
        try:
            ind_coh=np.where((clase_coh==j))[0]
            ind_coh0=np.where((clase_coh==j)&(clase_ctx==0))[0] # Left more rewarded
            ind_coh1=np.where((clase_coh==j)&(clase_ctx==1))[0] # Right more rewarded
            
            chrono_neuro_flat_pre[ii,j,0]=(1.0-cl.score(pseudo_te_flat[ind_coh],stim[ind_coh]))
            chrono_neuro_flat_pre[ii,j,1]=(1.0-cl.score(pseudo_te_flat[ind_coh0],stim[ind_coh0]))
            chrono_neuro_flat_pre[ii,j,2]=(1.0-cl.score(pseudo_te_flat[ind_coh1],stim[ind_coh1]))
            psycho_neuro_flat_pre[ii,j,1]=np.mean(cl.predict(pseudo_te_flat[ind_coh0]))
            psycho_neuro_flat_pre[ii,j,2]=np.mean(cl.predict(pseudo_te_flat[ind_coh1]))
            psycho_neuro_flat_pre[ii,j,0]=np.mean(cl.predict(pseudo_te_flat[ind_coh]))
        except:
            print ('rand ',ii,'coh ',j)
    
    indnan0=~np.isnan(psycho_neuro_flat_pre[ii,:,0])
    popt0,pcov0=curve_fit(log_curve,xx_coh[indnan0],psycho_neuro_flat_pre[ii,:,0][indnan0])
    fit_psycho_neuro_flat_pre[ii,indnan0,0]=log_curve(xx_coh[indnan0],popt0[0],popt0[1])
    
    indnan1=~np.isnan(psycho_neuro_flat_pre[ii,:,1])
    popt1,pcov1=curve_fit(log_curve,xx_coh[indnan1],psycho_neuro_flat_pre[ii,:,1][indnan1])
    fit_psycho_neuro_flat_pre[ii,indnan1,1]=log_curve(xx_coh[indnan1],popt1[0],popt1[1])
    
    indnan2=~np.isnan(psycho_neuro_flat_pre[ii,:,2])
    popt2,pcov2=curve_fit(log_curve,xx_coh[indnan2],psycho_neuro_flat_pre[ii,:,2][indnan2])
    fit_psycho_neuro_flat_pre[ii,indnan2,2]=log_curve(xx_coh[indnan2],popt2[0],popt2[1])

    ###########################################
    # psychometric as a function of time
    for hh in range(steps):
        print (hh)
        sum_nan=np.sum(np.isnan(pseudo_tr_pre[hh,0]),axis=1)
        index_nonan=np.where(sum_nan==0)[0]
        pseudo_tr=pseudo_tr_pre[hh,0][index_nonan]
        pseudo_te=pseudo_te_pre[hh,0][index_nonan]
        clase_all=clase_all_pre[index_nonan]
        clase_coh=clase_coh_pre[index_nonan]
        clase_ctx=clase_ctx_pre[index_nonan]
        stim=stim_pre[index_nonan]

        for j in range(n_coh):
            try:
                cl=LogisticRegression(C=1/reg,class_weight='balanced')
                cl.fit(pseudo_tr,stim) # Cuidado con decode Stim y no Choice!
            
                ind_coh=np.where((clase_coh==j))[0]
                ind_coh0=np.where((clase_coh==j)&(clase_ctx==0))[0] # Left more rewarded
                ind_coh1=np.where((clase_coh==j)&(clase_ctx==1))[0] # Right more rewarded
                
                #chrono_neuro_pre[hh,ii,j,0]=cl.score(pseudo_te[ind_coh],stim[ind_coh])
                #chrono_neuro_pre[hh,ii,j,1]=cl.score(pseudo_te[ind_coh0],stim[ind_coh0])
                #chrono_neuro_pre[hh,ii,j,2]=cl.score(pseudo_te[ind_coh1],stim[ind_coh1])
                psycho_neuro_pre[hh,ii,j,0]=np.mean(cl.predict(pseudo_te[ind_coh]))
                psycho_neuro_pre[hh,ii,j,1]=np.mean(cl.predict(pseudo_te[ind_coh0]))
                psycho_neuro_pre[hh,ii,j,2]=np.mean(cl.predict(pseudo_te[ind_coh1]))
            except:
                print ('rand ',ii,'step ',hh,'coh ',j)

        indnan0=~np.isnan(psycho_neuro_pre[hh,ii,:,0])
        popt0,pcov0=curve_fit(log_curve,xx_coh[indnan0],psycho_neuro_pre[hh,ii,:,0][indnan0])
        fit_psycho_neuro_pre[hh,ii,indnan0,0]=log_curve(xx_coh[indnan0],popt0[0],popt0[1])
        
        indnan1=~np.isnan(psycho_neuro_pre[hh,ii,:,1])
        popt1,pcov1=curve_fit(log_curve,xx_coh[indnan1],psycho_neuro_pre[hh,ii,:,1][indnan1])
        fit_psycho_neuro_pre[hh,ii,indnan1,1]=log_curve(xx_coh[indnan1],popt1[0],popt1[1])
            
        indnan2=~np.isnan(psycho_neuro_pre[hh,ii,:,2])
        popt2,pcov2=curve_fit(log_curve,xx_coh[indnan2],psycho_neuro_pre[hh,ii,:,2][indnan2])
        fit_psycho_neuro_pre[hh,ii,indnan2,2]=log_curve(xx_coh[indnan2],popt2[0],popt2[1])
            

################################################################
psycho_neuro_flat_m=np.nanmean(psycho_neuro_flat_pre,axis=(0))
psycho_neuro_flat_std=np.nanstd(psycho_neuro_flat_pre,axis=(0))
fit_psycho_neuro_flat_m=np.nanmean(fit_psycho_neuro_flat_pre,axis=(0))
fit_psycho_neuro_flat_std=np.nanstd(fit_psycho_neuro_flat_pre,axis=(0))

# Figure Psychometric
fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])

# Curve fit
ax.scatter(xx_coh,psycho_neuro_flat_m[:,0],color='black',s=3)
ax.plot(xx_coh,fit_psycho_neuro_flat_m[:,0],color='black')
ax.fill_between(xx_coh,fit_psycho_neuro_flat_m[:,0]-fit_psycho_neuro_flat_std[:,0],fit_psycho_neuro_flat_m[:,0]+fit_psycho_neuro_flat_std[:,0],color='black',alpha=0.5)
ax.scatter(xx_coh,psycho_neuro_flat_m[:,1],color='green',s=3)
ax.plot(xx_coh,fit_psycho_neuro_flat_m[:,1],color='green')
ax.fill_between(xx_coh,fit_psycho_neuro_flat_m[:,1]-fit_psycho_neuro_flat_std[:,1],fit_psycho_neuro_flat_m[:,1]+fit_psycho_neuro_flat_std[:,1],color='green',alpha=0.5)
ax.scatter(xx_coh,psycho_neuro_flat_m[:,2],color='blue',s=3)
ax.plot(xx_coh,fit_psycho_neuro_flat_m[:,2],color='blue')
ax.fill_between(xx_coh,fit_psycho_neuro_flat_m[:,2]-fit_psycho_neuro_flat_std[:,2],fit_psycho_neuro_flat_m[:,2]+fit_psycho_neuro_flat_std[:,2],color='blue',alpha=0.5)

ax.plot(xx_coh,0.5*np.ones(15),color='black',linestyle='--')
ax.axvline(0,color='black',linestyle='--')
ax.set_ylabel('Probability Right Response')
ax.set_xlabel('Evidence Right Choice (%)')
ax.set_ylim([-0.05,1.05])
plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
#plt.legend(loc='best')
plt.xticks([-2.54,0,2.54],['-12.8','0','12.8'])
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_neuro_pseudo_psychometric_%s_flat.pdf'%(monkey),dpi=500,bbox_inches='tight')
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_neuro_pseudo_psychometric_%s_flat.png'%(monkey),dpi=500,bbox_inches='tight')

#####################################
# Chrono
chrono_neuro_flat_m=np.nanmean(chrono_neuro_flat_pre,axis=(0))
chrono_neuro_flat_std=np.nanstd(chrono_neuro_flat_pre,axis=(0))

# Figure Psychometric
fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])

# Curve fit
ax.plot(xx_coh,chrono_neuro_flat_m[:,0],color='black')#,s=3)
#ax.fill_between(xx_coh,chrono_neuro_flat_m[:,0]-chrono_neuro_flat_std[:,0],chrono_neuro_flat_m[:,0]+chrono_neuro_flat_std[:,0],color='black',alpha=0.5)
ax.plot(xx_coh,chrono_neuro_flat_m[:,1],color='green')#,s=3)
#ax.fill_between(xx_coh,chrono_neuro_flat_m[:,1]-chrono_neuro_flat_std[:,1],chrono_neuro_flat_m[:,1]+chrono_neuro_flat_std[:,1],color='green',alpha=0.5)
ax.plot(xx_coh,chrono_neuro_flat_m[:,2],color='blue')#,s=3)
#ax.fill_between(xx_coh,chrono_neuro_flat_m[:,2]-chrono_neuro_flat_std[:,2],chrono_neuro_flat_m[:,2]+chrono_neuro_flat_std[:,2],color='blue',alpha=0.5)

ax.plot(xx_coh,0.5*np.ones(15),color='black',linestyle='--')
#ax.axvline(0,color='black',linestyle='--')
ax.set_ylim([0,0.6])
ax.set_ylabel('Probability Error')
ax.set_xlabel('Evidence Right Choice (%)')
plt.xticks(xx_coh,coh_plot[0])
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_neuro_pseudo_chronochometric_%s_flat.pdf'%(monkey),dpi=500,bbox_inches='tight')
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_neuro_pseudo_chronochometric_%s_flat.png'%(monkey),dpi=500,bbox_inches='tight')


########################################################
# Figure Psychometric time
psycho_neuro_m=np.nanmean(psycho_neuro_pre,axis=(1))
psycho_neuro_std=np.nanstd(psycho_neuro_pre,axis=(1))
fit_psycho_neuro_m=np.nanmean(fit_psycho_neuro_pre,axis=(1))
fit_psycho_neuro_std=np.nanstd(fit_psycho_neuro_pre,axis=(1))

for t_plot in range(steps):
    fig=plt.figure(figsize=(2.3,2))
    ax=fig.add_subplot(111)
    miscellaneous.adjust_spines(ax,['left','bottom'])

    # Curve fit
    ax.scatter(xx_coh,psycho_neuro_m[t_plot,:,0],color='black',s=3)
    ax.plot(xx_coh,fit_psycho_neuro_m[t_plot,:,0],color='black')
    ax.fill_between(xx_coh,fit_psycho_neuro_m[t_plot,:,0]-fit_psycho_neuro_std[t_plot,:,0],fit_psycho_neuro_m[t_plot,:,0]+fit_psycho_neuro_std[t_plot,:,0],color='black',alpha=0.5)
    ax.scatter(xx_coh,psycho_neuro_m[t_plot,:,1],color='green',s=3)
    ax.plot(xx_coh,fit_psycho_neuro_m[t_plot,:,1],color='green')
    ax.fill_between(xx_coh,fit_psycho_neuro_m[t_plot,:,1]-fit_psycho_neuro_std[t_plot,:,1],fit_psycho_neuro_m[t_plot,:,1]+fit_psycho_neuro_std[t_plot,:,1],color='green',alpha=0.5)
    ax.scatter(xx_coh,psycho_neuro_m[t_plot,:,2],color='blue',s=3)
    ax.plot(xx_coh,fit_psycho_neuro_m[t_plot,:,2],color='blue')
    ax.fill_between(xx_coh,fit_psycho_neuro_m[t_plot,:,2]-fit_psycho_neuro_std[t_plot,:,2],fit_psycho_neuro_m[t_plot,:,2]+fit_psycho_neuro_std[t_plot,:,2],color='blue',alpha=0.5)

    ax.plot(xx_coh,0.5*np.ones(15),color='black',linestyle='--')
    ax.axvline(0,color='black',linestyle='--')
    ax.set_ylim([0,1])
    ax.set_ylabel('Probability Right Response')
    ax.set_xlabel('Evidence Right Choice (%)')
    plt.xticks(xx_coh,coh_plot[0])
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_neuro_pseudo_psychometric_%s_t_%i.pdf'%(monkey,t_plot),dpi=500,bbox_inches='tight')
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_neuro_pseudo_psychometric_%s_t_%i.png'%(monkey,t_plot),dpi=500,bbox_inches='tight')


