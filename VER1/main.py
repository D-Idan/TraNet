# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from TranNet_nn_ver1 import KalmanTransformer
from params import *

import time
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from filing_paths import path_model

KalmanNet_path = Path('Alg_4_compare/KalmanNet').absolute()
pathList = [KalmanNet_path, Path(path_model).absolute()]
for path in pathList:
    if path not in sys.path:
        sys.path.insert(1, str(path))

from plots import plotCovariance
from Alg_4_compare.KalmanNet import KalmanNet_sysmdl
# from KalmanNet.ERRCOV_ICASSP22.src import KalmanNet_sysmdl, plotCovariance
from data.Lorenz_Atractor.model import *
from data.Lorenz_Atractor.parameters import *
# from KalmanNet.ERRCOV_ICASSP22.Simulations.Lorenz_Atractor.model import *
# from KalmanNet.ERRCOV_ICASSP22.Simulations.Lorenz_Atractor.parameters import *
from data.DataGenerator import load_data
from Alg_4_compare.EKF.EKF_test import EKFTest
from Alg_4_compare.KalmanNet.KalmanNet_test import NNTest
from plots.plotCovariance import empirical_error, plot_error_evolution_trace

from TransformerNet_build import NNBuild as trans_NNBuild
from transformet_train import NNTrain as trans_NNTrain
from plots.Plots import plot3D_1part, count_parameters
from utilities import calc_trace
import TraNet_sysmdl


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    device = torch.device("cpu")
print(device)

R = torch.eye(n)
T = 100

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    tmp_data_measurements = torch.rand((3))
    # tmp_data_measurements = torch.Tensor((1,1,1))
    # Dynamical model
    sys_model = TraNet_sysmdl.TraNet_SystemModel(f_interpolate, Q_mod, h, R_mod, T)
    sys_model.InitSequence(m1x_0_design, m2x_0_design)
    sys_model_EKF = KalmanNet_sysmdl.SystemModel(f_interpolate, Q_mod, h, R, T)
    sys_model_EKF.InitSequence(m1x_0_design, m2x_0_design)
    #
    # # # Build and train KalmanNet
    Model = trans_NNBuild(sys_model,d_model_enc=8, d_model_dec=8)


    out1 = Model(tmp_data_measurements)
    out = Model(tmp_data_measurements/2)
    print(out)
    print("FINISH First TEST !!")




    # load
    [test_target, test_input, test_IC, _, _, _, _, _, _, train_target, train_input, train_IC, CV_target, CV_input,
     CV_IC] = load_data("identity", 0, process_noise=None)

    R = torch.eye(n)

    # Dynamical model
    sys_model_KalmanNet = KalmanNet_sysmdl.SystemModel(f_interpolate, Q_mod, h, R_mod, T)
    sys_model_KalmanNet.InitSequence(m1x_0_design, m2x_0_design)
    sys_model_EKF = KalmanNet_sysmdl.SystemModel(f_interpolate, Q_mod, h, R, T)
    sys_model_EKF.InitSequence(m1x_0_design, m2x_0_design)
    sys_model_Transformer = TraNet_sysmdl.TraNet_SystemModel(f_interpolate, Q_mod, h, R, T)
    sys_model_Transformer.InitSequence(m1x_0_design, m2x_0_design)


    train_TransNet = False
    LoadPreTrainedModel = True

    if train_TransNet:
        # define paths to save results
        path_base = Path('Results_Transformer').absolute()
        os.makedirs(path_base, exist_ok=True)
        results_file = Path(path_base , "results.pt")
        path_results = path_base


        if LoadPreTrainedModel == True:
            Trans_Model = torch.load(Path('Results_Transformer/best-model.pt').absolute())
        else:
            # Build and train KalmanNet
            Trans_Model = trans_NNBuild(sys_model,d_model_enc=8, d_model_dec=8)
            # initiate weights
            initrange = 0.1
            for p in Trans_Model.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)

        [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = trans_NNTrain(
            sys_model_Transformer, Trans_Model, CV_input, CV_target, train_input, train_target, path_base, sequential_training=False, train_IC=train_IC, CV_IC=CV_IC)

    if train_TransNet != True:
        time_EKF = time.time()
        # testing EKF
        [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out, P_array_EKF, _] = EKFTest(
            sys_model_EKF, test_input, test_target, init_cond=test_IC, calculate_covariance=True)
        time_EKF = time.time() - time_EKF


        # Evaluate KalmanNet
        path_base = Path('Alg_4_compare/KalmanNet').absolute()
        kn_model_path = Path('Alg_4_compare/KalmanNet/best-model.pt').absolute()
        kn_model = torch.load(kn_model_path)
        KN_model_parameters = count_parameters(kn_model)
        time_KalmanNet = time.time()
        [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KGain_array,
         x_out_array, P_array, _] = NNTest(
            sys_model, test_input, test_target, path_base, IC=test_IC, calculate_covariance=True)
        time_KalmanNet = time.time() - time_KalmanNet


        # Evaluate TransformerNet
        path_base = Path('Results_Transformer').absolute()
        transNet_model_path = Path('Results_Transformer/best-model.pt').absolute()
        transNet_model = torch.load(transNet_model_path)
        transNet_model_parameters = count_parameters(transNet_model)
        time_TransformerNet = time.time()
        [TransMSE_test_linear_arr, TransMSE_test_linear_avg, TransMSE_test_dB_avg, TransKGain_array,
         Transx_out_array, TransP_array, _] = NNTest(
            sys_model_Transformer, test_input, test_target, path_base, IC=test_IC, calculate_covariance=True)
        time_TransformerNet = time.time() - time_TransformerNet

        # Print results and generate plots
        emp_err_EKF = empirical_error(test_target, EKF_out.to('cpu'))
        emp_err_knet = empirical_error(test_target, x_out_array.to('cpu'))
        emp_err_transnet = empirical_error(test_target, Transx_out_array.to('cpu'))

        print(f"EKF: MSE={MSE_EKF_dB_avg} [dB], Runing time={time_EKF}")
        print(f"KalmanNet: MSE={MSE_test_dB_avg} [dB], Parameters={KN_model_parameters},"
              f" Runing time={time_KalmanNet}")
        print(f"TransformerNet: MSE={TransMSE_test_dB_avg} [dB], Parameters={transNet_model_parameters},"
              f"Runing time={time_TransformerNet}")

        avg_cov_EKF = torch.mean(P_array_EKF.to('cpu'), 0)
        avg_cov_knet = torch.mean(P_array.to('cpu'), 0).detach()
        avg_cov_transnet = torch.mean(TransP_array.to('cpu'), 0).detach()

        trace_EKF = calc_trace(avg_cov_EKF.to('cpu'))
        trace_knet = calc_trace(avg_cov_knet.to('cpu'))
        trace_transnet = calc_trace(avg_cov_transnet.to('cpu'))

        path_results = Path('Results_Transformer').absolute()
        # plot_error_evolution_trace(trace_EKF, trace_knet, emp_err_EKF, emp_err_knet, T, path_results)
        plot_error_evolution_trace(trace_transnet, trace_knet, emp_err_transnet, emp_err_knet, T, path_results)

        MSE = torch.norm(test_target - x_out_array.to('cpu'), dim=1) ** 2
        MSE = torch.flatten(MSE).detach().numpy()
        transMSE = torch.norm(test_target - Transx_out_array.to('cpu'), dim=1) ** 2
        transMSE = torch.flatten(transMSE).detach().numpy()

        plt.close()
        plt.hist(MSE, bins=100, density=True)
        plt.xlabel("MSE")
        plt.ylabel("probability")
        plt.savefig(Path.joinpath(path_results, Path('histogram.png')), dpi=300)
        plt.show()

        plt.close()
        plt.hist(transMSE, bins=100, density=True)
        plt.xlabel("MSE")
        plt.ylabel("probability")
        plt.savefig(Path.joinpath(path_results, Path('histogram.png')), dpi=300)
        plt.show()

        # plot3D_1part(Transx_out_array, x_out_array,test_target)

        plot3D_1part(10,test_target, test_input,Transx_out_array)


