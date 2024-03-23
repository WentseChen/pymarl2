import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Mixer(nn.Module):
    def __init__(self, args, abs=True, scheme=None):
        super(Mixer, self).__init__()

        self.args = args
        self.abs = abs
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_state_dim = self.state_dim = int(np.prod(args.state_shape)) 
        
        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_state_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.input_state_dim, self.embed_dim))
        
        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(self.input_state_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.input_state_dim, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, 1))
        
        # hyper w3 b3
        self.hyper_w3 = nn.Sequential(nn.Linear(self.input_state_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.n_agents))
        self.hyper_b3 = nn.Sequential(nn.Linear(self.input_state_dim, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, 1))
        
        # linear transformation
        self.hyper_w5 = nn.Sequential(
            nn.Linear(self.state_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.n_agents)
        )
        self.hyper_b5 = nn.Sequential(
            nn.Linear(self.state_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.n_agents)
        )
    
    def beta(self, states):
        
        state_shape = states.shape
        
        states = states.reshape(-1, states.shape[-1])
        w = self.hyper_w3(states).view(-1, self.n_agents)
        if self.abs:
            w = w.abs()
            
        w = w.view(state_shape[:-1] + (self.n_agents, ))
        
        return w
    
        
    # def mbpb(self, qvals, states):
        
    #     qval_shape = qvals.shape
    #     states = states.reshape(-1, self.state_dim)
        
    #     if qval_shape[-2] == self.n_agents:
    #         self.dim_idx = -3
    #         qvals = qvals.reshape(-1, self.n_agents, qvals.shape[-1])
    #         w = self.hyper_w5(states).view(-1, self.n_agents, 1)
    #         b = self.hyper_b5(states).view(-1, self.n_agents, 1)
    #     if qval_shape[-1] == self.n_agents:
    #         self.dim_idx = -2
    #         qvals = qvals.reshape(-1, self.n_agents)
    #         w = self.hyper_w5(states).view(-1, self.n_agents)
    #         b = self.hyper_b5(states).view(-1, self.n_agents)
            
    #     if self.abs:
    #         w = w.abs()
        
    #     y = qvals * w + b 
        
    #     return y.reshape(qval_shape)
    
    # multiply beta and add bias
    def mbpb(self, qvals, states):
        
        qval_shape = qvals.shape
        
        if qval_shape[-2] == self.n_agents:
            qvals = qvals.reshape(-1, self.n_agents, qvals.shape[-1])
            states = states.reshape(-1, states.shape[-1])
            w = self.hyper_w3(states).view(-1, self.n_agents, 1)
            b = self.hyper_b3(states).view(-1, 1, 1) 
        if qval_shape[-1] == self.n_agents:
            qvals = qvals.reshape(-1, self.n_agents)
            states = states.reshape(-1, states.shape[-1])
            w = self.hyper_w3(states).view(-1, self.n_agents)
            b = self.hyper_b3(states).view(-1, 1)
        
        if self.abs:
            w = w.abs()
        
        y = qvals * w + b / self.n_agents
        
        return y.reshape(qval_shape)

    def forward(self, qvals, states):
        
        # reshape
        b, t, _ = qvals.size()
        
        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        # First layer
        w1 = self.hyper_w1(states).view(-1, self.n_agents, self.embed_dim) # b * t, n_agents, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)
        
        # Second layer
        w2 = self.hyper_w2(states).view(-1, self.embed_dim, 1) # b * t, emb, 1
        b2= self.hyper_b2(states).view(-1, 1, 1)
        
        if self.abs:
            w1 = w1.abs()
            w2 = w2.abs()
            
        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1) # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2 # b * t, 1, 1
        
        return y.view(b, t, -1)
    
