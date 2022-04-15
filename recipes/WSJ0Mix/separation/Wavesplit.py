import torch
from torch import nn


class SpeakerStack(nn.Module):
    #the firsrt stack convert the mixture into a representation of each speaker
    #input x = [batchsize,Timestep]
    #output h = [batch_size,num_spks,Timestep,dims of repre vectors(hyper)]
    #this part is to model a latent of speakers,which means,vecotr h1 represents bob now,but may represents mike later
    #That will be solved after clustering(inference),when training,Speaker centroids is derived from known data
    pass


class SeparationStack(nn.Module):
    #the second stack transforms the input mixture into multiple isolated recordings conditioned on the speaker representation
    #maps the mixture x and speaker centroids into N-channel signal y
    pass