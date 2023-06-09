import time
import torch
import numpy as np

# open a file for output information in iterations
fout=open('nn.err','w')
#======general setup===========================================
table_init=0                   # 1: a pretrained or restart  
force_table=True
ratio=0.9                      # ratio for vaildation
find_unused = False
Epoch=50000                    # total numbers of epochs for fitting 
queue_size=10
dtype='float32'   #float32/float64
# batchsize: the most import setup for efficiency
batchsize=64  # batchsize for each process
init_weight=[1.0, 5.0]
final_weight=[1.0, 0.5]

#========================parameters for optim=======================
start_lr=0.01                  # initial learning rate
end_lr=1e-5                    # final learning rate
re_ceff=0.0                    # L2 normalization cofficient
decay_factor=0.5               # Factor by which the learning rate will be reduced. new_lr = lr * factor.      
patience_epoch=100             # patience epoch  Number of epochs with no improvement after which learning rate will be reduced. 
#=======================parameters for local environment========================
maxneigh=100000
cutoff = 4.0
max_l=2.0
nwave=8
#==================================data floder=============================
floder="./"

#===============================embedded NN structure==========
emb_nblock=1
emb_nl=[8,8]

#=========radial nn=================================================
r_nblock = 1                     # the number of resduial NN blocks
r_nl=[8,8]                   # NN structure
#===========params for MPNN ===============================================
iter_loop = 2
iter_nblock = 1             # neural network architecture   
iter_nl = [64,64]
iter_dropout_p=[0.0,0.0,0.0,0.0]
iter_table_norm=False

#======== parameters for final output nn=================================================
nblock = 1                     # the number of resduial NN blocks
nl=[64,64]                   # NN structure
dropout_p=[0.0,0.0]            # dropout probability for each hidden layer
table_norm = False

#======================read input=================================================================
with open('para/input','r') as f1:
   while True:
      tmp=f1.readline()
      if not tmp: break
      string=tmp.strip()
      if len(string)!=0:
          if string[0]=='#':
              pass
          else:
              m=string.split('#')
              exec(m[0])

# torch and numpy dtype
if dtype=='float64':
    torch_dtype=torch.float64
    np_dtype=np.float64
else:
    torch_dtype=torch.float32
    np_dtype=np.float32

torch.set_default_dtype(torch_dtype)

# parallel process the variable  
#=====================environment for select the GPU in free=================================================
local_rank = int(os.environ.get("LOCAL_RANK"))
local_size = int(os.environ.get("LOCAL_WORLD_SIZE"))

gpu_sel()

# set the backend for DDP comunication
if torch.cuda.is_available():
    DDP_backend="nccl"
else:
    DDP_backend="gloo"

world_size = int(os.environ.get("WORLD_SIZE"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu",local_rank)
dist.init_process_group(backend=DDP_backend)

rank=dist.get_rank()
datafloder=floder+str(rank)+"/"
#==============================train data loader===================================
init_weight=troch.tensor(init_weight).to(torch_dtype).to(device)
final_weight=troch.tensor(final_weight).to(torch_dtype).to(device)
if force_table:
    nprop=2
else:
    nprop=1
    init_weight=init_weight[0:1]
    final_weight=final_weight[0:1]
