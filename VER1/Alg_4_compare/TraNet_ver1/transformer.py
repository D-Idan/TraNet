import torch
import torch.nn as nn
from collections import deque
import torch.nn.functional as func
from filing_paths import path_model
from torch.nn import TransformerEncoder, TransformerEncoderLayer,\
    TransformerDecoderLayer, TransformerDecoder
from utilities import *
from params import *
# from NN_parameters import nGRU

import sys

sys.path.insert(1, path_model)
from KalmanNet.ERRCOV_ICASSP22.Simulations.Lorenz_Atractor.model import getJacobian

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")


##########################################
## Building the net
##########################################

# parameters controlling the size of the MEMORY (number of samples and outs to remember)
in_mem_t = 3
out_mem_t = 3

class KalmanTransformer(torch.nn.Module):
    '''Neural Network using hybrid model
    KalmanNet with substitute the deep RNN network to transformer'''

    ###################
    ### Constructor ###
    ###################
    def __init__(self,d_model_enc, d_model_dec, in_mem_t=in_mem_t, out_mem_t=out_mem_t):
        super().__init__()
        self.in_mem_t = in_mem_t
        self.out_mem_t = out_mem_t
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model=d_model_enc, dropout=dropout)
        encoder_layers = TransformerEncoderLayer(d_model=d_model_enc, nhead=nhead,
                                                 dim_feedforward=d_hid, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers,
                                                      num_layers=nlayers)
        decoder_layers = TransformerDecoderLayer(d_model=d_model_dec, nhead=nhead,
                                               dim_feedforward=dim_feedforward_dec,
                                                 dropout=dropout)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layers,
                                                      num_layers=nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model_enc = d_model_enc
        self.d_model_dec = d_model_dec
        self.m = m
        self.n = n
        # fcLayers = [nn.Linear(d_model_dec * k, self.m*self.n) for k in range(1,15)]
        # self.FC_Layers = nn.ModuleList(fcLayers).to(dev)
        # self.fc3 = nn.Linear(d_model_dec*self.out_mem_t, self.m*self.n)
        dropout01 = nn.Dropout(0.1)
        flatten = nn.Flatten(0,-1)
        fcMaxMem = nn.Linear(6 * max_mem_len , 32) # 72 - > 32
        fc2KG = nn.Linear(32 , self.m * self.n)  # 32 - > 6(KG)
        self.linearTrans_Out2KG = nn.Sequential(flatten, fcMaxMem, dropout01, nn.ELU(), fc2KG)

    ############################################
    ### Initialize KalmanTransformer Network ###
    ############################################
    def InitKGainNet_FIRST_TRAIN(self):

        # # initiate weights
        # initrange = 0.1
        # self.fc3.weight.data.uniform_(-initrange, initrange)
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        pass

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, f, h, m, n, R, infoString='fullInfo'):

        if (infoString == 'partialInfo'):
            self.fString = 'ModInacc'
            self.hString = 'ObsInacc'
        else:
            self.fString = 'ModAcc'
            self.hString = 'ObsAcc'

        # Set State Evolution Function
        self.f = f
        self.m = m

        # Set Observation Function
        self.h = h
        self.n = n

        # set measurement noise (only needed for covariance estimate)
        self.R = R

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, M2_0, T, mem_t=12, calculate_covariance=False):
        # Collect the results and measurements
        self.in_mem_t = self.out_mem_t = mem_t

        self.mem_in_q = deque()
        self.mem_out_q = deque()
        self.mem_in_q.append(torch.zeros([M1_0.size(0) * 2]))
        self.mem_out_q.append(torch.zeros([M1_0.size(0) * 2]))
        # for _ in range(mem_t):
        #     self.mem_in_q.append(torch.zeros([M1_0.size(0)*2]))
        #     self.mem_out_q.append(torch.zeros([M1_0.size(0)*2]))

        # print(f'$$$$$$$$$$$$$$$$$$$$$$$$ in_mem_t = {self.in_mem_t} $$$$$$$$$$$$$$$$$$$$$$$$$$$')
        # print(f'$$$$$$$$$$$$$$$$$$$$$$$$ in_mem_t = {self.out_mem_t} $$$$$$$$$$$$$$$$$$$$$$$$$$$')
        # self.mem_in = torch.zeros([self.d_model_enc, self.in_mem_t]).to(dev)
        # self.mem_out = torch.zeros([self.d_model_dec, self.out_mem_t]).to(dev)
        # self.mem_inx = 0

        self.T = T
        self.x_out = torch.empty(self.m, T).to(dev, non_blocking=True)

        self.calculate_covariance = calculate_covariance

        self.m1x_posterior = M1_0.to(dev, non_blocking=True)
        self.m1x_posterior_previous = self.m1x_posterior.to(dev, non_blocking=True)
        self.m1x_prior_previous = self.m1x_posterior.to(dev, non_blocking=True)
        self.y_previous = self.h(self.m1x_posterior).to(dev, non_blocking=True)

        # KGain saving
        self.i = 0
        # self.KGain_array = self.KG_array = torch.zeros((self.T,self.m,self.n)).to(dev, non_blocking=True)
        self.KGain_array = torch.zeros((self.T, self.m, self.n)).to(dev, non_blocking=True)
        if self.calculate_covariance:
            self.P_array = torch.empty((self.T, self.n, self.n))

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior)

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):

        # calculating and normalizing all the features
        obs_diff = y - torch.squeeze(self.y_previous)
        obs_innov_diff = y - torch.squeeze(self.m1y)
        fw_evol_diff = torch.squeeze(self.m1x_posterior, dim=-1) - torch.squeeze(self.m1x_posterior_previous, dim=-1)
        fw_update_diff = torch.squeeze(self.m1x_posterior, dim=-1) - torch.squeeze(self.m1x_prior_previous, dim=-1)

        obs_diff = func.normalize(obs_diff, p=2, dim=0, eps=1e-12, out=None).to(dev)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=0, eps=1e-12, out=None).to(dev)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=0, eps=1e-12, out=None).to(dev)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=0, eps=1e-12, out=None).to(dev)

        # Kalman Gain Network Step
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.m, self.n))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):

        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Save KGain in array
        self.KGain_array[self.i] = self.KGain

        # Innovation
        y_obs = torch.unsqueeze(y, 1)
        dy = y_obs - self.m1y

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV

        # self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior

        # update y_prev
        self.y_previous = y

        # calculate the covariance
        if self.calculate_covariance:
            self.P_array[self.i, :, :] = self.covariance()
        self.i += 1

        # return
        return torch.squeeze(self.m1x_posterior)

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        self.seq_len_input = 1
        self.batch_size = 1
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, 0, :] = x
            return expanded

        in_tensor = torch.cat((obs_diff, obs_innov_diff), 0)
        out_tensor = torch.cat((fw_evol_diff, fw_update_diff), 0)

        self.mem_in_q.append(in_tensor)
        self.mem_out_q.append(out_tensor)

        if len(self.mem_in_q) > self.in_mem_t :
            self.mem_in_q.popleft()
        if len(self.mem_out_q) > self.out_mem_t:
            self.mem_out_q.popleft()

        self.mem_in = torch.stack(list(self.mem_in_q)).T
        self.mem_out = torch.stack(list(self.mem_out_q)).T

        # self.mem_inx += 1
        # if self.mem_inx < self.in_mem_t + 1:
        #     self.mem_in[:,-self.mem_inx] = in_tensor
        #     self.mem_out[:,-self.mem_inx] = out_tensor
        #
        #     if self.mem_inx == self.in_mem_t:
        #         self.mem_inx = 0

        # obs_diff = expand_dim(obs_diff)
        # obs_innov_diff = expand_dim(obs_innov_diff)
        # fw_evol_diff = expand_dim(fw_evol_diff)
        # fw_update_diff = expand_dim(fw_update_diff)



        ####################
        ### Forward Flow ###
        ####################

        # Measurements will go through the encoder.
        # Prior out states will go through the decoder.
        # Encoder outputs will go to the Decoder too.
        # Decoder out will get it fully connected layer
        # output will be the model kalman gain

        # POSITION ENCODING
        mem_in = self.pos_encoder(self.mem_in)
        mem_out = self.pos_encoder(self.mem_out)

        encoder_output = self.transformer_encoder.to(dev)(mem_in.T)
        decoder_output = self.decoder.to(dev)(mem_out.T, encoder_output)
        # Transpose for clarity and delete position encodings:
        # 6 features in each measurement and state
        decoder_output = decoder_output.T[-6: ,:]
        # Padding : Max memrory length in params.py
        padding = max_mem_len - decoder_output.size(1)
        trans_out_padded = func.pad(decoder_output, (0, padding, 0, 0))

        # decoder_output1D = torch.reshape(trans_out_padded,[1,-1])



        net_output = self.linearTrans_Out2KG(trans_out_padded)
        return net_output
        # return self.fc3.to(dev)(decoder_output)

    ##################################
    ### calculate state covariance ###
    ##################################
    def covariance(self):
        H = getJacobian(self.m1x_prior, self.h)

        """
        HTHi = torch.inverse(torch.matmul(H.T, H))
        A = torch.matmul(HTHi, H.T)
        HKi = torch.inverse(torch.matmul(H, self.KGain))
        B = torch.inverse(HKi - torch.eye(self.n))
        C = torch.matmul(self.R, torch.matmul(H, HTHi))

        P_prior = torch.matmul(A, torch.matmul(B,C))
        P_posterior = torch.matmul((torch.eye(self.n) - torch.matmul(self.KGain, H)), P_prior)"""

        Lambda = torch.inverse(torch.matmul(H.T, H))
        G = torch.matmul(H, self.KGain)

        A = torch.matmul(Lambda, H.T)
        B = torch.inverse(torch.eye(self.n) - G)
        C = torch.matmul(G, torch.matmul(H, Lambda))

        P_prior = torch.matmul(A, torch.matmul(B, C))
        D = torch.eye(self.m) - torch.matmul(self.KGain, H)
        P_posterior = torch.matmul(D, P_prior)

        return P_posterior

    ###############
    ### Forward ###
    ###############
    def forward(self, y):
        y = y.to(dev, non_blocking=True)

        '''
        for t in range(0, self.T):
            self.x_out[:, t] = self.KNet_step(y[:, t])
        '''
        self.x_out = self.KNet_step(y)

        return self.x_out



