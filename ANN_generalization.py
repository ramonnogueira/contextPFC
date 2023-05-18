import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from scipy.stats import sem
import scipy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
import pickle as pkl
import nn_pytorch
import miscellaneous_ANN
import miscellaneous
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.stats import norm
#torch.autograd.set_detect_anomaly(True)

dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def class_twovars(data,var1,var2):
    n_rand=10
    n_cv=5
    reg=1
    perf=nan*np.zeros((n_rand,n_cv,2))
    uq1=np.unique(var1)
    uq2=np.unique(var2)
    for i in range(n_rand):
        ind11=np.where((var1==uq1[0])&(var2==uq2[0]))[0]
        ind12=np.where((var1==uq1[0])&(var2==uq2[1]))[0]
        ind21=np.where((var1==uq1[1])&(var2==uq2[0]))[0]
        ind22=np.where((var1==uq1[1])&(var2==uq2[1]))[0]
        mint=np.min(np.array([len(ind11),len(ind12),len(ind21),len(ind22)]))
        ind_all=[ind11,ind12,ind21,ind22]
        class_all=np.array([[0,0],[0,1],[1,0],[1,1]])
        # Create dataset
        data_r=nan*np.zeros((4*mint,len(data[0])))
        clas_r=np.zeros((4*mint,2),dtype=np.int16)
        for ii in range(4):
            ind_r=np.random.choice(ind_all[ii],mint,replace=False)
            data_r[ii*(mint):(ii+1)*mint]=data[ind_r]
            clas_r[ii*(mint):(ii+1)*mint]=class_all[ii]
        # Decode Var1
        skf=StratifiedKFold(n_splits=n_cv)
        g=-1
        for train, test in skf.split(data_r,clas_r[:,0]):
            g=(g+1)
            cl=LogisticRegression(C=1/reg)
            cl.fit(data_r[train],clas_r[train][:,0])
            perf[i,g,0]=cl.score(data_r[test],clas_r[test][:,0])
        # Decode Var2
        skf=StratifiedKFold(n_splits=n_cv)
        g=-1
        for train, test in skf.split(data_r,clas_r[:,1]):
            g=(g+1)
            cl=LogisticRegression(C=1/reg)
            cl.fit(data_r[train],clas_r[train][:,1])
            perf[i,g,1]=cl.score(data_r[test],clas_r[test][:,1])
    return np.mean(perf,axis=(0,1))

# Input shape has to be: Batch x Steps x Input dim
# Target shape has to be: Batch x Steps (x Target dim)
# The output of the model concatenates all time steps from all trials
class nn_recurrent():
    def __init__(self,reg,lr,output_size,hidden_dim,wei_ctx):
        self.regularization=reg
        self.learning_rate=lr
        self.loss=torch.nn.CrossEntropyLoss(reduction='none')
        self.model=recurrent_noisy(output_size,hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)

    def fit(self,input_seq,target_seq,context,ctx_weights,input_seq_t,target_seq_t,context_t,ctx_weights_t,batch_size,n_epochs,sigma_noise,wei_ctx,learn_moments): 
        self.model.train()
        input_seq_np=np.array(input_seq,dtype=np.float32)
        context_np=np.array(context,dtype=np.float32)
        ctx_uq=np.unique(context_np)
        target_seq_np=np.array(target_seq,dtype=np.int16)
        ctx_wei_seq_np=np.array(ctx_weights,dtype=np.int16)
        input_seq_np_t=np.array(input_seq_t,dtype=np.float32)
        context_np_t=np.array(context_t,dtype=np.float32)
        target_seq_np_t=np.array(target_seq_t,dtype=np.int16)
        ctx_wei_seq_np_t=np.array(ctx_weights_t,dtype=np.int16)
        #
        input_seq_torch=Variable(torch.from_numpy(input_seq_np),requires_grad=False)
        context_torch=Variable(torch.from_numpy(context_np),requires_grad=False)
        target_seq_torch=Variable(torch.from_numpy(target_seq_np),requires_grad=False)
        ctx_wei_seq_torch=Variable(torch.from_numpy(ctx_wei_seq_np),requires_grad=False)
        input_seq_torch_t=Variable(torch.from_numpy(input_seq_np_t),requires_grad=False)
        context_torch_t=Variable(torch.from_numpy(context_np_t),requires_grad=False)
        target_seq_torch_t=Variable(torch.from_numpy(target_seq_np_t),requires_grad=False)
        ctx_wei_seq_torch_t=Variable(torch.from_numpy(ctx_wei_seq_np_t),requires_grad=False)
        #
        train_loader=DataLoader(torch.utils.data.TensorDataset(input_seq_torch,context_torch,ctx_wei_seq_torch,target_seq_torch),batch_size=batch_size,shuffle=True)

        net_units_vec=[]
        readout_units_vec=[]
        for t in range(n_epochs):
            net_units, readout_units = self.model(input_seq_torch,target_seq_torch,ctx_wei_seq_torch,sigma_noise)
            net_units_t, readout_units_t = self.model(input_seq_torch_t,target_seq_torch_t,ctx_wei_seq_torch_t,sigma_noise)

            if t in learn_moments:
                net_units_vec.append(net_units_t.detach().numpy())
                readout_units_vec.append(readout_units_t.detach().numpy())
            
            l_total=0
            for u in range(net_units.size(0)):
                ind11=(target_seq_torch[:,u]==0)*(context_torch[:,u]==ctx_uq[0])
                ind10=(target_seq_torch[:,u]==0)*(context_torch[:,u]==ctx_uq[1])
                ind01=(target_seq_torch[:,u]==1)*(context_torch[:,u]==ctx_uq[0])
                ind00=(target_seq_torch[:,u]==1)*(context_torch[:,u]==ctx_uq[1])
                l0_ct0=torch.mean(self.loss(readout_units[ind11][:,u,[0,1]],target_seq_torch[:,u][ind11].view(-1).long()))
                l0_ct1=torch.mean(self.loss(readout_units[ind10][:,u,[0,1]],target_seq_torch[:,u][ind10].view(-1).long()))
                l1_ct0=torch.mean(self.loss(readout_units[ind01][:,u,[0,1]],target_seq_torch[:,u][ind01].view(-1).long()))
                l1_ct1=torch.mean(self.loss(readout_units[ind00][:,u,[0,1]],target_seq_torch[:,u][ind00].view(-1).long()))
                loss_b=(wei_ctx[0]*(l0_ct0+l1_ct1)+wei_ctx[1]*(l1_ct0+l0_ct1))
                l_total=(l_total+loss_b)
            #if t==0 or t==(n_epochs-1):
            print (t,l_total.detach().numpy())
        
            for batch_idx, (data, contxt, ctx_wei, targets) in enumerate(train_loader):
                self.optimizer.zero_grad()
                nu, ru = self.model(data,targets,ctx_wei,sigma_noise)
                loss_t=0
                for u in range(net_units.size(0)):
                    ind11=(targets[:,u]==0)*(contxt[:,u]==ctx_uq[0])
                    ind10=(targets[:,u]==0)*(contxt[:,u]==ctx_uq[1])
                    ind01=(targets[:,u]==1)*(contxt[:,u]==ctx_uq[0])
                    ind00=(targets[:,u]==1)*(contxt[:,u]==ctx_uq[1])
                    loss0_ct0=torch.mean(self.loss(ru[ind11][:,u,[0,1]],targets[:,u][ind11].view(-1).long()))
                    loss0_ct1=torch.mean(self.loss(ru[ind10][:,u,[0,1]],targets[:,u][ind10].view(-1).long()))
                    loss1_ct0=torch.mean(self.loss(ru[ind01][:,u,[0,1]],targets[:,u][ind01].view(-1).long()))
                    loss1_ct1=torch.mean(self.loss(ru[ind00][:,u,[0,1]],targets[:,u][ind00].view(-1).long()))
                    loss=(wei_ctx[0]*(loss0_ct0+loss1_ct1)+wei_ctx[1]*(loss1_ct0+loss0_ct1))
                    loss_t=(loss_t+loss)
                loss_t.backward()
                self.optimizer.step()

        net_units_vec=np.array(net_units_vec)
        readout_units_vec=np.array(readout_units_vec)
        return net_units_vec,readout_units_vec

    
class recurrent_noisy(torch.nn.Module): # We always send the input with size batch x time steps x input dim
    def __init__(self, output_size, hidden_dim):
        super(recurrent_noisy, self).__init__()
        self.hidden_dim=hidden_dim
        self.output_size=output_size
        self.input_weights = torch.nn.Linear(1, hidden_dim)
        self.input_weights2 = torch.nn.Linear(1,hidden_dim)
        self.hidden_weights = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(self.hidden_dim, self.output_size)
        self.wei_ctx=wei_ctx

    def forward(self, input, target, ctx_weights, sigma_noise, hidden=None):
        # Initial state of the hidden units
        if hidden is None:
            hidden = torch.randn(input.size(0),self.hidden_dim).to(input.device)
            rew = torch.randn(input.size(0),1).to(input.device)
            
        # Function that determines the time evolution of the RNN
        def recurrence(input, hidden, rew):
            h_new = torch.tanh(self.input_weights(input) + self.hidden_weights(hidden) + self.input_weights2(rew) + sigma_noise*torch.randn(input.size(0),self.hidden_dim))
            return h_new

        # net units: activity of the RNN for all the time steps in the trial (trials, time, neurons). 
        net_units = torch.zeros(input.size(0),input.size(1),self.hidden_dim)
        readout_units = torch.zeros(input.size(0),input.size(1),2)
        steps = range(input.size(1))
        for i in steps:
            #print ('step ',i,input[:,i].size(),rew.size())
            hidden = recurrence(input[:,i], hidden, rew)
            ru = self.fc(hidden)
            dec = torch.softmax(ru,axis=1)
            rew=torch.sum(dec*ctx_weights[:,i],dim=1,keepdim=True)
            net_units[:,i]=hidden
            readout_units[:,i] = self.fc(hidden)
        return net_units, readout_units 

# Creates a correlated stream of classes across trials. 
# N trials: number of trials used
# Prob: probability of repeating class. 1 totally repeating, 0 totally alternating
def class_stream(n_trials,t_steps,prob):
    class_stream=nan*np.zeros((n_trials,t_steps))
    class_stream[:,0]=(np.random.normal(size=n_trials)>=0)
    for i in range(1,t_steps):
        sto=norm.cdf(np.random.normal(size=n_trials))
        class_stream[:,i]=abs(class_stream[:,i-1]-np.array(sto>prob,dtype=np.int16))
    return Variable(torch.from_numpy(np.array(2*class_stream-1,dtype=np.int16)),requires_grad=False)

# Creates a noisy input per trial. No time steps within trial. 
def create_input(n_trials,t_steps,coh_uq,input_noise,wei_ctx,scale_ctx,prob):
    dic={}
    input_vec_pre=nan*np.zeros((n_trials,t_steps*len(coh_uq),1))
    target_vec_pre=nan*np.zeros((n_trials,t_steps*len(coh_uq)))
    coherence_pre=nan*np.zeros((n_trials,t_steps*len(coh_uq)))

    # Input is random noise with different means (coherence). Last dimension is context, same for all time steps and +1 -1. 
    # Target is 1 when mean is positive and 0 otherwise
    # Create Stimulus
    for i in range(len(coh_uq)):
        input_vec_pre[:,i*t_steps:(i+1)*t_steps,0]=np.random.normal(loc=coh_uq[i],scale=input_noise,size=(n_trials,t_steps))
        target_vec_pre[:,i*t_steps:(i+1)*t_steps]=np.sign(coh_uq[i])
        coherence_pre[:,i*t_steps:(i+1)*t_steps]=coh_uq[i]
    target_vec_pre[target_vec_pre==0]=np.sign(np.random.normal(loc=0,scale=1,size=(n_trials*t_steps)))
    target_vec_pre[target_vec_pre==-1]=0

    # Shuffle indeces
    input_vec=nan*np.zeros((n_trials,t_steps*len(coh_uq),1))
    target_vec=nan*np.zeros((n_trials,t_steps*len(coh_uq)))
    coherence=nan*np.zeros((n_trials,t_steps*len(coh_uq)))
    for i in range(n_trials):
        index_def=np.random.permutation(np.arange(len(coh_uq)*t_steps))
        input_vec[i]=input_vec_pre[i][index_def]
        target_vec[i]=target_vec_pre[i][index_def]
        coherence[i]=coherence_pre[i][index_def]

    # Context
    ctx_vec=class_stream(n_trials,t_steps*len(coh_uq),prob)
    #
    ctx_wei_vec=nan*np.zeros((n_trials,t_steps*len(coh_uq),2))
    ctx_wei_vec[ctx_vec==-1]=np.array([wei_ctx[0],wei_ctx[1]])
    ctx_wei_vec[ctx_vec==1]=np.array([wei_ctx[1],wei_ctx[0]])

    # Return input and target in torch format
    dic['input_rec']=Variable(torch.from_numpy(np.array(input_vec,dtype=np.float32)),requires_grad=False)
    dic['target_vec']=Variable(torch.from_numpy(np.array(target_vec,dtype=np.int16)),requires_grad=False)
    dic['coherence']=Variable(torch.from_numpy(np.array(coherence,dtype=np.float32)),requires_grad=False)
    dic['context']=Variable(torch.from_numpy(np.array(ctx_vec,dtype=np.int16)),requires_grad=False)
    dic['context_wei']=Variable(torch.from_numpy(np.array(ctx_wei_vec,dtype=np.int16)),requires_grad=False)
    return dic

#######################################################
# Parameters       
n_trials_train=200
n_trials_test=200
t_steps=30
prob=0.99 #probability than on each trial the context is the same. Full random is when prob = 0.5.

batch_size=200
n_hidden=10
sigma_train=1
sigma_test=1
input_noise=1.5
scale_ctx=1

reg=1e-5
lr=0.001
n_epochs=200#1000
n_files=2

n_learn=10
learn_moments=np.array(n_epochs*np.linspace(0,1,n_learn),dtype=np.int16)
learn_moments[-1]=(learn_moments[-1]-1)

coh_uq=np.linspace(-2,2,11)
coh_uq_abs=coh_uq[coh_uq>=0]
wei_ctx=[10,1]

perf_task=nan*np.zeros((n_files,2,t_steps,len(coh_uq)))
psycho=nan*np.zeros((n_files,n_learn,t_steps,len(coh_uq),3))
for hh in range(n_files):
    print (hh)
    # Def variables
    all_train=create_input(n_trials_train,t_steps,coh_uq,input_noise,wei_ctx=wei_ctx,scale_ctx=scale_ctx,prob=prob)
    all_test=create_input(n_trials_test,t_steps,coh_uq,input_noise,wei_ctx=wei_ctx,scale_ctx=scale_ctx,prob=prob)
    #
    stimulus=all_test['target_vec'].detach().numpy()
    coherence=all_test['coherence'].detach().numpy()
    context=all_test['context'].detach().numpy()
    ctx_wei=all_test['context_wei'].detach().numpy()
    ctx_uq=np.unique(context)

    #ctx_ch=(context[:,1:]-context[:,0:-1])
    #ind_ch=np.where(abs(ctx_ch)==1)[0]
    #print (ind_ch,len(ind_ch))
        
    # Train RNN
    rec=nn_recurrent(reg=reg,lr=lr,output_size=2,hidden_dim=n_hidden,wei_ctx=wei_ctx)
    net_units_vec,readout_vec=rec.fit(input_seq=all_train['input_rec'],target_seq=all_train['target_vec'],context=all_train['context'],ctx_weights=all_train['context_wei'],input_seq_t=all_test['input_rec'],target_seq_t=all_test['target_vec'],context_t=all_test['context'],ctx_weights_t=all_test['context_wei'],batch_size=batch_size,n_epochs=n_epochs,sigma_noise=sigma_train,wei_ctx=wei_ctx,learn_moments=learn_moments)
    
    # Psychometric
    for u in range(n_learn):
        for j in range(t_steps):
            dec=np.argmax(readout_vec[u,:,j],axis=1)
            for i in range(len(coh_uq)):
                psycho[hh,u,j,i,0]=np.mean(dec[(coherence[:,j]==coh_uq[i])])
                psycho[hh,u,j,i,1]=np.mean(dec[(coherence[:,j]==coh_uq[i])&(context[:,j]==ctx_uq[0])])
                psycho[hh,u,j,i,2]=np.mean(dec[(coherence[:,j]==coh_uq[i])&(context[:,j]==ctx_uq[1])])
                
# ######################################################

ps=np.mean(psycho,axis=(0,2))
for i in range(n_learn):
    print (i)
    plt.ylim([0,1])
    plt.plot(ps[i,:,0],color='black')
    plt.plot(ps[i,:,1],color='green')
    plt.plot(ps[i,:,2],color='blue')
    plt.show()

# # Plot performance vs time for different coherences
# perf_abs_m=np.mean(perf_task_abs,axis=0)
# perf_abs_sem=sem(perf_task_abs,axis=0)

# fig=plt.figure(figsize=(2.3,2))
# ax=fig.add_subplot(111)
# miscellaneous.adjust_spines(ax,['left','bottom'])
# for i in range(len(coh_uq_abs)):
#     ax.plot(np.arange(t_steps),perf_abs_m[1,i],color='black',alpha=(i+1)/len(coh_uq_abs))
#     ax.fill_between(np.arange(t_steps),perf_abs_m[1,i]-perf_abs_sem[1,i],perf_abs_m[1,i]+perf_abs_sem[1,i],color='black',alpha=(i+1)/len(coh_uq_abs))
# ax.plot(np.arange(t_steps),0.5*np.ones(t_steps),color='black',linestyle='--')
# ax.set_ylim([0.4,1])
# ax.set_ylabel('Probability Correct')
# ax.set_xlabel('Time')
# if save_fig:
#     fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_rnn_prob_correct_coh_rr%i%i_new2.pdf'%(wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')
#     fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_rnn_prob_correct_coh_rr%i%i_new2.png'%(wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')

# ########################################################
# print (np.mean(perf_dec_ctx,axis=0))

# psycho_m=np.mean(psycho,axis=0)
# psycho_sem=sem(psycho,axis=0)
# perfbias_m=np.mean(perf_bias,axis=0)
# perfbias_sem=sem(perf_bias,axis=0)
# rt_m=np.mean(rt,axis=0)
# rt_sem=sem(rt,axis=0)

# # Plot Reaction Time
# fig=plt.figure(figsize=(2.3,2))
# ax=fig.add_subplot(111)
# miscellaneous.adjust_spines(ax,['left','bottom'])
# ax.plot(coh_uq,rt_m[:,0],color='black',label='All trials')
# ax.fill_between(coh_uq,rt_m[:,0]-rt_sem[:,0],rt_m[:,0]+rt_sem[:,0],color='black',alpha=0.5)
# ax.plot(coh_uq,rt_m[:,1],color='green',label='Context Left')
# ax.fill_between(coh_uq,rt_m[:,1]-rt_sem[:,1],rt_m[:,1]+rt_sem[:,1],color='green',alpha=0.5)
# ax.plot(coh_uq,rt_m[:,2],color='blue',label='Context Right')
# ax.fill_between(coh_uq,rt_m[:,2]-rt_sem[:,2],rt_m[:,2]+rt_sem[:,2],color='blue',alpha=0.5)
# ax.set_ylabel('Reaction time (steps)')
# ax.set_xlabel('Evidence Right Choice (%)')
# ax.set_ylim([1,20])
# if save_fig:
#     fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_reaction_time_rr%i%i_new2.pdf'%(wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')
#     fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_reaction_time_rr%i%i_new2.png'%(wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')

# for t_plot in range(t_steps):
#     # Figure psychometric
#     fig=plt.figure(figsize=(2.3,2))
#     ax=fig.add_subplot(111)
#     miscellaneous.adjust_spines(ax,['left','bottom'])
#     #print (psycho_m)
#     ax.plot(coh_uq*100,psycho_m[:,t_plot,0],color='black')
#     ax.fill_between(coh_uq*100,psycho_m[:,t_plot,0]-psycho_sem[:,t_plot,0],psycho_m[:,t_plot,0]+psycho_sem[:,t_plot,0],color='black',alpha=0.6)
#     ax.plot(coh_uq*100,psycho_m[:,t_plot,1],color='green')
#     ax.fill_between(coh_uq*100,psycho_m[:,t_plot,1]-psycho_sem[:,t_plot,1],psycho_m[:,t_plot,1]+psycho_sem[:,t_plot,1],color='green',alpha=0.6)
#     ax.plot(coh_uq*100,psycho_m[:,t_plot,2],color='blue')
#     ax.fill_between(coh_uq*100,psycho_m[:,t_plot,2]-psycho_sem[:,t_plot,2],psycho_m[:,t_plot,2]+psycho_sem[:,t_plot,2],color='blue',alpha=0.6)
#     ax.plot(coh_uq*100,0.5*np.ones(len(coh_uq)),color='black',linestyle='--')
#     ax.axvline(0,color='black',linestyle='--')
#     ax.set_ylim([0,1])
#     ax.set_ylabel('Probability Right Response')
#     ax.set_xlabel('Evidence Right Choice (%)')
#     if save_fig:
#         fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_rnn_psychometric_t%i_rr%i%i_new2.png'%(t_plot,wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')

#     # Probability Correct
#     fig=plt.figure(figsize=(2.3,2))
#     ax=fig.add_subplot(111)
#     miscellaneous.adjust_spines(ax,['left','bottom'])
#     ax.plot(coh_uq*100,perfbias_m[:,t_plot,0],color='black')
#     ax.fill_between(coh_uq*100,perfbias_m[:,t_plot,0]-perfbias_sem[:,t_plot,0],perfbias_m[:,t_plot,0]+perfbias_sem[:,t_plot,0],color='black',alpha=0.6)
#     ax.plot(coh_uq*100,perfbias_m[:,t_plot,1],color='green')
#     ax.fill_between(coh_uq*100,perfbias_m[:,t_plot,1]-perfbias_sem[:,t_plot,1],perfbias_m[:,t_plot,1]+perfbias_sem[:,t_plot,1],color='green',alpha=0.6)
#     ax.plot(coh_uq*100,perfbias_m[:,t_plot,2],color='blue')
#     ax.fill_between(coh_uq*100,perfbias_m[:,t_plot,2]-perfbias_sem[:,t_plot,2],perfbias_m[:,t_plot,2]+perfbias_sem[:,t_plot,2],color='blue',alpha=0.6)
#     ax.plot(coh_uq*100,0.5*np.ones(len(coh_uq)),color='black',linestyle='--')
#     ax.axvline(0,color='black',linestyle='--')
#     ax.set_ylim([0,1])
#     ax.set_ylabel('Probability Correct')
#     ax.set_xlabel('Evidence Right Choice (%)')
#     if save_fig:
#         fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_rnn_perf_bias_t%i_rr%i%i_new2.png'%(t_plot,wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')        
