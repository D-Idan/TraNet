from TranNet_nn_ver1 import KalmanTransformer
import torch

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")

def NNBuild(SysModel,d_model_enc, d_model_dec):

    Model = KalmanTransformer(d_model_enc, d_model_dec)

    Model.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n, SysModel.R, infoString = "partialInfo")
    Model.InitSequence(SysModel.m1x_0, SysModel.m2x_0, SysModel.T)

    # Model.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S)

    return Model

