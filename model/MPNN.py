import torch
from torch import nn
from torch import Tensor
import numpy as np
import low_level.MLP as MLP
import low_level.sph_cal as sph_cal
from collections import OrderedDict
from low_level.activate import Relu_like as actfun

class MPNN(torch.nn.Module):
    def __init__(self,maxneigh,initpot,max_l=2,nwave=8,cutoff=4.0,norbital=32,emb_nblock=1,emb_nl=[8,8],emb_layernorm=True,iter_loop=3,iter_nblock=1,iter_nl=[64,64],iter_dropout_p=[0.0,0.0],iter_layernorm=True,nblock=1,nl=[64,64],dropout_p=[0.0,0.0],layernorm=True,device=torch.device("cpu"),Dtype=torch.float32):
        super(MPNN,self).__init__()
        self.nwave=nwave
        self.max_l=max_l
        self.cutoff=cutoff
        self.norbital=norbital
        self.nangular=(self.max_l+1)*(self.max_l+1)
        # add the input neuron for each neuron
        self.index_l=torch.zeros(self.nangular,device=device,dtype=torch.long)
        for l in range(max_l+1):
            self.index_l[l*l:(l+1)*(l+1)]=l
        emb_nl.insert(0,1)
        iter_nl.insert(0,self.norbital)
        nl.insert(0,self.norbital)

        self.contracted_coeff=nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.randn(max_l+1,nwave,norbital)))

        init_coeff=torch.ones(nwave,device=device,dtype=Dtype)
        alpha=torch.ones(nwave,device=device,dtype=Dtype)
        rs=(torch.rand(nwave)*np.sqrt(cutoff)).to(device)
        initbias=torch.hstack((init_coeff,rs,alpha))
        # embedded nn
        self.embnn=MLP.NNMod(self.nwave*3,emb_nblock,emb_nl,np.array([0]),actfun,initbias=initbias,layernorm=layernorm)

        # instantiate the nn radial function and disable the dropout 
        self.sph_cal=sph_cal.SPH_CAL(max_l,device=device,Dtype=Dtype)
        itermod=OrderedDict()
        for i in range(iter_loop):
            f_iter="memssage_"+str(i)
            itermod[f_iter]= MLP.NNMod(nwave,iter_nblock,iter_nl,iter_dropout_p,actfun,initbias=torch.ones(nwave),layernorm=iter_layernorm)
        self.itermod= torch.nn.ModuleDict(itermod)
        self.outnn=MLP.NNMod(1,nblock,nl,dropout_p,actfun,initbias=torch.tensor(np.array([initpot])),layernorm=layernorm)

    def forward(self,cart,neighlist,shifts,center_factor,neigh_factor,species):
        distvec=cart[neighlist[1]]-cart[neighlist[0]]+shifts
        distances=torch.linalg.norm(distvec,dim=1)
        center_coeff=self.embnn(species)
        full_center_list=center_coeff[neighlist[0]]
        neigh_emb=(full_center_list*center_coeff[neighlist[1]]).T
        cut_distances=neigh_factor*self.cutoff_cosine(distances)  
        distvec=torch.einsum("ik,i ->ik",distvec,cut_distances)
        contracted_coeff=self.contracted_coeff[self.index_l]
        # for the efficiency of traditional ANN, we do the first calculation of density mannually.
        weight_distvec=torch.einsum("ji,ji,ik->ikj",neigh_emb[:self.nwave],self.radial_func(distances,neigh_emb[self.nwave:self.nwave*2],neigh_emb[self.nwave*2:self.nwave*3]),distvec)
        wc_distvec=cart.new_zeros((cart.shape[0],3,self.nwave))
        wc_distvec=torch.index_add(wc_distvec,0,neighlist[0],weight_distvec)
        sph=self.sph_cal(wc_distvec.permute(1,0,2))
        contracted_sph=torch.einsum("kij,kjm->ikm",sph,contracted_coeff)
        density=torch.einsum("ikm,ikm->im",contracted_sph,contracted_sph)
        for iter_loop, (_, m) in enumerate(self.itermod.items()):
            iter_coeff=m(density)[neighlist[1]]
            iter_density,wc_distvec=self.density(weight_distvec,cut_distances,iter_coeff,neighlist[0],neighlist[1],contracted_coeff,wc_distvec)
            # here cente_coeff is for discriminating for the different center atoms.
            density=density+iter_density
        output=self.outnn(density)
        return torch.einsum("ij,i ->",output,center_factor)

    def density(self,weight_distvec,cut_distances,iter_coeff,index_center,index_neigh,contracted_coeff,wc_distvec):
        wn_distvec=torch.einsum("i,ikj -> ikj",cut_distances,wc_distvec[index_neigh])
        weight_distvec=torch.einsum("ij,ikj -> ikj",iter_coeff,weight_distvec)+wn_distvec
        wc_distvec=torch.index_add(wc_distvec,0,index_center,weight_distvec)
        sph=self.sph_cal(wc_distvec.permute(1,0,2))
        contracted_sph=torch.einsum("kij,kjm->ikm",sph,contracted_coeff)
        density=torch.einsum("ikm,ikm->im",contracted_sph,contracted_sph)
        return density,wc_distvec
     
    def cutoff_cosine(self,distances):
        tmp=0.5 * torch.cos(distances * (np.pi / self.cutoff)) + 0.5
        return tmp*tmp

    def radial_func(self,distances,rs,alpha):
        return torch.exp(-torch.square(alpha*(distances-rs)))
