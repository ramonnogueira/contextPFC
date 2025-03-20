import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import scipy
#import matplotlib.pylab as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
nan=float('nan')

dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# Input shape has to be: Batch x Steps x Input dim
# Target shape has to be: Batch x Steps (x Target dim)
# The output of the model concatenates all time steps from all trials

# class nn_recurrent():
#     def __init__(self,reg,lr,output_size,hidden_dim):
#         self.regularization=reg
#         self.learning_rate=lr
#         self.loss=torch.nn.CrossEntropyLoss(reduction='none')
#         self.model=recurrent_noisy(output_size,hidden_dim)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)

#     def fit(self,input_seq,target_seq,batch_size,n_epochs,sigma_noise,wei_ctx): 
#         thres_fit=1e-3
#         self.model.train()
#         input_seq_np=np.array(input_seq,dtype=np.float32)
#         context_np=np.array(input_seq[:,0,1],dtype=np.float32)
#         ctx_uq=np.unique(context_np)
#         target_seq_np=np.array(target_seq,dtype=np.int16)
#         input_seq_torch=Variable(torch.from_numpy(input_seq_np),requires_grad=False)
#         context_torch=Variable(torch.from_numpy(context_np),requires_grad=False)
#         target_seq_torch=Variable(torch.from_numpy(target_seq_np),requires_grad=False)
#         train_loader=DataLoader(torch.utils.data.TensorDataset(input_seq_torch,context_torch,target_seq_torch),batch_size=batch_size,shuffle=True)
        
#         index11=((target_seq_torch==0)&(context_torch==ctx_uq[0]))
#         index10=((target_seq_torch==0)&(context_torch==ctx_uq[1]))
#         index01=((target_seq_torch==1)&(context_torch==ctx_uq[0]))
#         index00=((target_seq_torch==1)&(context_torch==ctx_uq[1]))

#         for t in range(n_epochs):
#             output, hidden, net_units, read_out_units = self.model(input_seq_torch,sigma_noise)
#             l0c0=torch.mean(self.loss(output[index11][:,[0,1]],target_seq_torch[index11].view(-1).long()))
#             l0c1=torch.mean(self.loss(output[index10][:,[0,1]],target_seq_torch[index10].view(-1).long()))
#             l1c0=torch.mean(self.loss(output[index01][:,[0,1]],target_seq_torch[index01].view(-1).long()))
#             l1c1=torch.mean(self.loss(output[index00][:,[0,1]],target_seq_torch[index00].view(-1).long()))
#             l_total=(l0c0+l1c0+l0c1+l1c1)
#             if t==0 or t==(n_epochs-1):
#                 print (t,l_total.detach().numpy())
#             #if (l0c0+l1c0+l0c1+l1c1)<thres_fit:
#             #    break
#             for batch_idx, (data, contxt, targets) in enumerate(train_loader):
#                 ind11=(targets==0)*(contxt==ctx_uq[0])
#                 ind10=(targets==0)*(contxt==ctx_uq[1])
#                 ind01=(targets==1)*(contxt==ctx_uq[0])
#                 ind00=(targets==1)*(contxt==ctx_uq[1])
#                 self.optimizer.zero_grad()
#                 output, hidden, net_units, read_out_units = self.model(data,sigma_noise)
#                 loss0_ct0=torch.mean(self.loss(output[ind11][:,[0,1]],targets[ind11].view(-1).long()))
#                 loss0_ct1=torch.mean(self.loss(output[ind10][:,[0,1]],targets[ind10].view(-1).long()))
#                 loss1_ct0=torch.mean(self.loss(output[ind01][:,[0,1]],targets[ind01].view(-1).long()))
#                 loss1_ct1=torch.mean(self.loss(output[ind00][:,[0,1]],targets[ind00].view(-1).long()))
#                 loss=(wei_ctx[0]*(loss0_ct0+loss1_ct1)+wei_ctx[1]*(loss1_ct0+loss0_ct1))
#                 #loss_ctx=(torch.mean(self.loss(output[:,[2,3]],contxt.view(-1).long())))
#                 #loss=(wei_task[0]*loss_rdm+wei_task[1]*loss_ctx)
#                 loss.backward()
#                 self.optimizer.step()
#         return self.model.state_dict()

############################################
def sparsity_loss(data,p):
    loss=torch.mean(torch.pow(abs(data),p),axis=(0,1,2))
    return loss
    
class nn_recurrent_sparse():
    def __init__(self,reg,lr,output_size,hidden_dim):
        self.regularization=reg
        self.learning_rate=lr
        self.loss=torch.nn.CrossEntropyLoss(reduction='none')
        self.model=recurrent_noisy(output_size,hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)

    def fit(self,input_seq,target_seq,context,batch_size,n_epochs,sigma_noise,wei_ctx,beta,b_exp): 
        thres_fit=1e-3
        self.model.train()
        input_seq_np=np.array(input_seq,dtype=np.float32)
        context_np=np.array(context,dtype=np.float32)
        ctx_uq=np.unique(context_np)
        target_seq_np=np.array(target_seq,dtype=np.int16)
        input_seq_torch=Variable(torch.from_numpy(input_seq_np),requires_grad=False)
        context_torch=Variable(torch.from_numpy(context_np),requires_grad=False)
        target_seq_torch=Variable(torch.from_numpy(target_seq_np),requires_grad=False)
        train_loader=DataLoader(torch.utils.data.TensorDataset(input_seq_torch,context_torch,target_seq_torch),batch_size=batch_size,shuffle=True)
        
        index11=((target_seq_torch==0)&(context_torch==ctx_uq[0]))
        index10=((target_seq_torch==0)&(context_torch==ctx_uq[1]))
        index01=((target_seq_torch==1)&(context_torch==ctx_uq[0]))
        index00=((target_seq_torch==1)&(context_torch==ctx_uq[1]))
    
        for t in range(n_epochs):
            output, hidden, net_units, read_out_units = self.model(input_seq_torch,sigma_noise)
            l0c0=torch.mean(self.loss(output[index11][:,[0,1]],target_seq_torch[index11].view(-1).long()))
            l0c1=torch.mean(self.loss(output[index10][:,[0,1]],target_seq_torch[index10].view(-1).long()))
            l1c0=torch.mean(self.loss(output[index01][:,[0,1]],target_seq_torch[index01].view(-1).long()))
            l1c1=torch.mean(self.loss(output[index00][:,[0,1]],target_seq_torch[index00].view(-1).long()))
            l_total=(l0c0+l1c0+l0c1+l1c1)
            l_sparse=sparsity_loss(net_units,b_exp)
            #if t==0 or t==(n_epochs-1):
            print (t,l_total.detach().numpy(),l_sparse.detach().numpy())
           
            for batch_idx, (data, contxt, targets) in enumerate(train_loader):
                ind11=(targets==0)*(contxt==ctx_uq[0])
                ind10=(targets==0)*(contxt==ctx_uq[1])
                ind01=(targets==1)*(contxt==ctx_uq[0])
                ind00=(targets==1)*(contxt==ctx_uq[1])
                
                self.optimizer.zero_grad()
                output, hidden, net_units, read_out_units = self.model(data,sigma_noise)
                loss0_ct0=torch.mean(self.loss(output[ind11][:,[0,1]],targets[ind11].view(-1).long()))
                loss0_ct1=torch.mean(self.loss(output[ind10][:,[0,1]],targets[ind10].view(-1).long()))
                loss1_ct0=torch.mean(self.loss(output[ind01][:,[0,1]],targets[ind01].view(-1).long()))
                loss1_ct1=torch.mean(self.loss(output[ind00][:,[0,1]],targets[ind00].view(-1).long()))
                loss=(wei_ctx[0]*(loss0_ct0+loss1_ct1)+wei_ctx[1]*(loss1_ct0+loss0_ct1))
                l_sp=sparsity_loss(net_units,b_exp)
                loss_t=(loss+beta*l_sp)
                loss_t.backward()
                self.optimizer.step()
        return self.model.state_dict()
    
class recurrent_noisy(torch.nn.Module): # We always send the input with size batch x time steps x input dim
    def __init__(self, output_size, hidden_dim):
        super(recurrent_noisy, self).__init__()
        self.hidden_dim=hidden_dim
        self.output_size=output_size
        self.input_weights = torch.nn.Linear(2, hidden_dim)
        self.hidden_weights = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, input, sigma_noise, hidden=None):
        if hidden is None:
            hidden = torch.randn(input.size(0),self.hidden_dim).to(input.device)
            #hidden = torch.zeros(input.size(0),self.hidden_dim).to(input.device)
        
        # def recurrence(input, hidden):
        #     #h_new = torch.relu(self.input_weights(input) + self.hidden_weights(hidden) + sigma_noise*torch.randn(input.size(0),self.hidden_dim))
        #     h_new = torch.tanh(self.input_weights(input) + self.hidden_weights(hidden) + sigma_noise*torch.randn(input.size(0),self.hidden_dim))
        #     #h_new = torch.sigmoid(self.input_weights(input) + self.hidden_weights(hidden) + sigma_noise*torch.randn(input.size(0),self.hidden_dim))
        #     return h_new        
    
        # net_units = torch.zeros(input.size(0),input.size(1),self.hidden_dim)
        # steps = range(input.size(1))
        # for i in steps:
        #     hidden = recurrence(input[:,i], hidden)
        #     #hidden = recurrence_lin(input[:,i], hidden)
        #     net_units[:,i]=hidden

        model_gru=torch.nn.GRU(input_size=2,hidden_size=self.hidden_dim,batch_first=True)
        net_units, hidden  = model_gru(input)
        print (net_units.shape, hidden.shape)
            
        #hidden = hidden.detach()
        out = net_units[:,-1].contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        read_out_units = self.fc(net_units)
        return out, hidden, net_units, read_out_units # out: choice of readout unit last time step, hidden: state network last time step, 
            









