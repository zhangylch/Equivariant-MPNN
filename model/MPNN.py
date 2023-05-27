import torch
from torch import nn
from torch import Tensor
import numpy as np
import low_level.MLP as MLP
import low_level.sph_cal as sph_cal
from collections import OrderedDict
from low_level.activate import Relu_like as actfun

class MPNN(torch.nn.Module):
    def __init__(self,initpot,max_l=2,nwave=8,cutoff=4.0,norbital=32,emb_nblock=1,emb_nl=[24,24],emb_layernorm=True,iter_loop=3,iter_nblock=1,iter_nl=[64,64],iter_dropout_p=[0.0,0.0],iter_layernorm=True,nblock=1,nl=[64,64],dropout_p=[0.0,0.0],layernorm=True,device=torch.device("cpu"),Dtype=torch.float32):
        super(MPNN,self).__init__()
        self.nwave=nwave
        self.max_l=max_l
        self.cutoff=cutoff
        self.norbital=norbital
        self.rmaxl=self.max_l+1
        self.nangular=self.rmaxl*self.rmaxl
        # add the input neuron for each neuron
        emb_nl.insert(0,1)
        iter_nl.insert(0,self.norbital)
        nl.insert(0,self.norbital)

        self.contracted_coeff=nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.randn(self.rmaxl,nwave,norbital)))

        self.index_l=torch.zeros(self.nangular,device=device,dtype=torch.long)
        for l in range(self.rmaxl):
            self.index_l[l*l:(l+1)*(l+1)]=l           
 
        initbias=torch.ones(nwave,device=device,dtype=Dtype)
        alpha=torch.ones(nwave,device=device,dtype=Dtype)
        rs=(torch.rand(nwave)*cutoff).to(device).to(Dtype)
        initbias=torch.hstack((initbias,alpha,rs))
        # embedded nn
        self.embnn=MLP.NNMod(self.nwave*3,emb_nblock,emb_nl,np.array([0]),actfun,initbias=initbias,layernorm=emb_layernorm)

        # instantiate the nn radial function and disable the dropout 
        self.sph_cal=sph_cal.SPH_CAL(max_l,device=device,Dtype=Dtype)
        itermod=OrderedDict()
        for i in range(iter_loop):
            initbias=torch.ones(nwave,device=device,dtype=Dtype)
            f_iter="memssage_"+str(i)
            itermod[f_iter]= MLP.NNMod(nwave,iter_nblock,iter_nl,iter_dropout_p,actfun,initbias=initbias,layernorm=iter_layernorm)
        self.itermod= torch.nn.ModuleDict(itermod)
        self.outnn=MLP.NNMod(1,nblock,nl,dropout_p,actfun,initbias=torch.tensor(np.array([initpot])),layernorm=layernorm)

    def forward(self,cart,neighlist,shifts,center_factor,neigh_factor,species):
        distvec=cart[neighlist[1]]-cart[neighlist[0]]+shifts
        distances=torch.linalg.norm(distvec,dim=1)
        center_coeff=self.embnn(species)
        full_list=center_coeff[neighlist]
        neigh_emb=torch.einsum("ij, ij -> ji",full_list[0],full_list[1])
        cut_distances=neigh_factor*self.cutoff_cosine(distances)  
        contracted_coeff=self.contracted_coeff[self.index_l]
        # for the efficiency of traditional ANN, we do the first calculation of density mannually.
        radial_func=self.radial_func(distances,neigh_emb[self.nwave:self.nwave*2],neigh_emb[self.nwave*2:])
        sph=self.sph_cal(distvec.T/self.cutoff)
        orbital=torch.einsum("i,ji,ji,ki->ikj",cut_distances,radial_func,neigh_emb[:self.nwave],sph)
        center_orbital=cart.new_zeros((cart.shape[0],self.nangular,self.nwave),dtype=cart.dtype,device=cart.device)
        center_orbital=torch.index_add(center_orbital,0,neighlist[0],orbital)
        contracted_orbital=torch.einsum("ikj,kjm->ikm",center_orbital,contracted_coeff)
        density=torch.einsum("ikm,ikm->im",contracted_orbital,contracted_orbital)
        for iter_loop, (_, m) in enumerate(self.itermod.items()):
            iter_coeff=m(density)
            iter_density,center_orbital=self.density(orbital,cut_distances,iter_coeff,neighlist[0],neighlist[1],contracted_coeff,center_orbital)
            # here cente_coeff is for discriminating for the different center atoms.
            density=density+iter_density
        output=self.outnn(density)
        return torch.einsum("ij,i ->",output,center_factor)

    def density(self,orbital,cut_distances,iter_coeff,index_center,index_neigh,contracted_coeff,center_orbital):
        weight_orbital=torch.einsum("ij,ikj -> ikj",iter_coeff[index_neigh],orbital)+torch.einsum("ikj,i->ikj",center_orbital[index_neigh],cut_distances)
        center_orbital=torch.index_add(center_orbital,0,index_center,weight_orbital)
        contracted_orbital=torch.einsum("ikj,kjm->ikm",center_orbital,contracted_coeff)
        density=torch.einsum("ikm,ikm->im",contracted_orbital,contracted_orbital)
        return density,center_orbital
     
    def cutoff_cosine(self,distances):
        tmp=0.5 * torch.cos(distances * (np.pi / self.cutoff)) + 0.5
        return tmp*tmp

    def radial_func(self,distances,alpha,rs):
        return torch.exp(-torch.square(alpha*(distances-rs)))
