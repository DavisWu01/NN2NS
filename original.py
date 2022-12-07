"""
@author: LXA
 Created on: 2022 年 10 月 15 日
"""
import os
import sys
import platform
import shutil
import time
import numpy as np
import matplotlib
import torch
import torch.nn as tn
import torch.nn.functional as tnf
from torch.nn.parameter import Parameter


import DNN_base
import dataUtilizer2torch
import Load_data2Mat
import saveData
import plotData
import DNN_Log_Print

# Reference article: 1、Physics-Informed Deep Neural Networks for Learning Parameters and Constitutive
#                    Relationships in Subsurface Flow Problems
#                    2、Deep-Learning-Based Inverse Modeling Approaches: A Subsurface Flow Example
#                    3、Physics constrained learning for data-driven inverse modeling from sparse observations
#                    4、Physics-informed Karhunen-Loéve and neural network approximations for solving
#                    inverse differential equation problems


class DNN2Inverse(tn.Module):
    def __init__(self, input_dim=1, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='relu',
                 name2actHidden='relu', name2actOut='linear',  input_dim2Para=1, out_dim2Para=1, hidden_layer2Para=None,
                 Model_Name2Para='DNN', name_actIn2Para='tanh', name_actHidden2Para='relu', name_actOut2Para='linear',
                 freq2Solu=None, sFourier2Solu=1.0, repeat_highFreq2Solu=False, freq2Para=None,  sFourier2Para=1.0,
                 repeat_highFreq2Para=False, opt2regular_WB='L2',  type2numeric='float32', use_gpu=False, No2GPU=0):
        super(DNN2Inverse, self).__init__()
        if 'DNN' == str.upper(Model_name):
            self.DNN2Solu = DNN_base.Pure_DenseNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name, actName2in=name2actIn,
                actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric, scope2W='UW', scope2B='BU',
                to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'SCALE_DNN' == str.upper(Model_name) or 'DNN_SCALE' == str.upper(Model_name):
            self.DNN2Solu = DNN_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name, actName2in=name2actIn,
                actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric, scope2W='UW', scope2B='BU',
                repeat_Highfreq=repeat_highFreq2Solu, to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'FOURIER_DNN' == str.upper(Model_name) or 'DNN_FOURIERBASE' == str.upper(Model_name):
            self.DNN2Solu = DNN_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name, actName2in=name2actIn,
                actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric, scope2W='UW', scope2B='BU',
                repeat_Highfreq=repeat_highFreq2Solu, to_gpu=use_gpu, gpu_no=No2GPU)

        if 'DNN' == str.upper(Model_Name2Para):
            self.DNN2Para = DNN_base.Pure_DenseNet(
                indim=input_dim2Para, outdim=out_dim2Para, hidden_units=hidden_layer2Para, name2Model=Model_Name2Para,
                actName2in=name_actIn2Para, actName=name_actHidden2Para, actName2out=name_actOut2Para,
                type2float=type2numeric, scope2W='UK', scope2B='BK', to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'SCALE_DNN' == str.upper(Model_Name2Para) or 'DNN_SCALE' == str.upper(Model_Name2Para):
            self.DNN2Para = DNN_base.Dense_ScaleNet(
                indim=input_dim2Para, outdim=out_dim2Para, hidden_units=hidden_layer2Para, name2Model=Model_Name2Para,
                actName2in=name_actIn2Para, actName=name_actHidden2Para, actName2out=name_actOut2Para,
                type2float=type2numeric, scope2W='UK', scope2B='BK', repeat_Highfreq=repeat_highFreq2Para,
                to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'FOURIER_DNN' == str.upper(Model_Name2Para) or 'DNN_FOURIERBASE' == str.upper(Model_Name2Para):
            self.DNN2Para = DNN_base.Dense_FourierNet(
                indim=input_dim2Para, outdim=out_dim2Para, hidden_units=hidden_layer2Para, name2Model=Model_Name2Para,
                actName2in=name_actIn2Para, actName=name_actHidden2Para, actName2out=name_actOut2Para,
                type2float=type2numeric, scope2W='UK', scope2B='BK', repeat_Highfreq=repeat_highFreq2Para,
                to_gpu=use_gpu, gpu_no=No2GPU)

        if type2numeric == 'float32':
            self.float_type = torch.float32
        elif type2numeric == 'float64':
            self.float_type = torch.float64
        elif type2numeric == 'float16':
            self.float_type = torch.float16

        self.freq2Solu = freq2Solu
        self.freq2ParaModel = freq2Para

        self.sFourier2Solu = sFourier2Solu
        self.sFourier2Para = sFourier2Para

        self.opt2regular_WB = opt2regular_WB

        self.use_gpu = use_gpu
        if use_gpu:
            self.opt2device = 'cuda:' + str(No2GPU)
        else:
            self.opt2device = 'cpu'

        if use_gpu:
            self.mat2X = torch.tensor([[1, 0]], dtype=self.float_type, device=self.opt2device)  # 1 行 2 列
            self.mat2U = torch.tensor([[0, 1]], dtype=self.float_type, device=self.opt2device)  # 1 行 2 列
            self.mat2T = torch.tensor([[0, 1]], dtype=self.float_type, device=self.opt2device)  # 1 行 2 列
        else:
            self.mat2X = torch.tensor([[1, 0]], dtype=self.float_type)  # 1 行 2 列
            self.mat2U = torch.tensor([[0, 1]], dtype=self.float_type)  # 1 行 2 列
            self.mat2T = torch.tensor([[0, 1]], dtype=self.float_type)  # 1 行 2 列

    def loss_linear(self, X=None,
                    fside=None,
                    if_lambda2fside=True,
                    loss_type='l2_loss',
                    scale2lncosh=0.5):
        # -div(K(x)grad U(x)) = f(x)
        # U(xb) = g(xb)
        # Reference article: Physics-Informed Deep Neural Networks for Learning Parameters and Constitutive
        #                    Relationships in Subsurface Flow Problems
        assert (X is not None)
        assert (fside is not None)

        shape2X = X.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2fside:
            force_side = fside(X)
        else:
            force_side = fside

        UNN = self.DNN2Solu(X, scale=self.factor2freq, sFourier=self.sFourier)
        KNN = self.DNN2Para(X, scale=self.freq2ParaModel, sFourier=self.sFourier)
        grad2UNNx = torch.autograd.grad(UNN, X, grad_outputs=torch.ones_like(X),
                                        create_graph=True, retain_graph=True)
        dUNN2X = grad2UNNx[0]

        if str.lower(loss_type) == 'l2_loss':
            grad2UNNxx = torch.autograd.grad(dUNN2X, X, grad_outputs=torch.ones(X.shape),
                                             create_graph=True, retain_graph=True)
            dUNNxx = grad2UNNxx[0]

            grad2KNNx = torch.autograd.grad(KNN, X, grad_outputs=torch.ones(X.shape), create_graph=True,
                                            retain_graph=True)
            dKNN2X = grad2KNNx[0]

            # -div(K(x)grad U(x)) = f(x) --->  -[KxUx+KUxx]=f
            loss_it_temp = torch.mul(dKNN2X, dUNN2X) + torch.mul(KNN, dUNNxx) + torch.reshape(force_side, shape=[-1, 1])
            square_loss_it = torch.square(loss_it_temp)
            loss_it = torch.mean(square_loss_it)
            return UNN, KNN, loss_it
        elif str.lower(loss_type) == 'ritz_loss':
            # -div(K(x)grad U(x)) = f(x) --->  0.5*K*|gradU|^2 - f*U
            square_norm2dUNN = torch.square(dUNN2X)
            ritz2dUNN = 0.5*torch.mul(KNN, square_norm2dUNN) - torch.mul(force_side, UNN)
            loss_it = torch.mean(ritz2dUNN)
            return UNN, KNN, loss_it
        elif str.lower(loss_type) == 'lncosh' or str.lower(loss_type) == 'lncosh_loss':
            grad2UNNxx = torch.autograd.grad(dUNN2X, X, grad_outputs=torch.ones(X.shape),
                                             create_graph=True, retain_graph=True)
            dUNNxx = grad2UNNxx[0]

            grad2KNNx = torch.autograd.grad(KNN, X, grad_outputs=torch.ones(X.shape), create_graph=True,
                                            retain_graph=True)
            dKNN2X = grad2KNNx[0]
            # -div(K(x)grad U(x)) = f(x) --->  -[KxUx+KUxx]=f
            loss_it_temp = torch.mul(dKNN2X, dUNN2X) + torch.mul(KNN, dUNNxx) + torch.reshape(force_side, shape=[-1, 1])
            logcosh_loss_it = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss_it_temp))
            loss_it = torch.mean(logcosh_loss_it)
            return UNN, KNN, loss_it

    def loss_nonlinear(self, X=None,
                       fside=None,
                       if_lambda2fside=True,
                       loss_type='l2_loss',
                       scale2lncosh=0.5):
        # -div(K(x,U)grad U(x)) = f(x)
        # U(xb) = g(xb)
        # Reference article: Physics-Informed Deep Neural Networks for Learning Parameters and Constitutive
        #                    Relationships in Subsurface Flow Problems
        assert (X is not None)
        assert (fside is not None)

        shape2X = X.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2fside:
            force_side = fside(X)
        else:
            force_side = fside

        UNN = self.DNN2Solu(X, scale=self.factor2freq, sFourier=self.sFourier)

        XU = torch.matmul(X, self.mat2X) + torch.matmul(UNN, self.mat2U)
        KNN = self.DNN2Para(XU, scale=self.freq2ParaModel, sFourier=self.sFourier)

        grad2UNNx = torch.autograd.grad(UNN, X, grad_outputs=torch.ones_like(X),
                                        create_graph=True, retain_graph=True)
        dUNN2X = grad2UNNx[0]

        if str.lower(loss_type) == 'l2_loss':
            grad2UNNxx = torch.autograd.grad(dUNN2X, X, grad_outputs=torch.ones(X.shape),
                                             create_graph=True, retain_graph=True)
            dUNNxx = grad2UNNxx[0]

            grad2KNNx = torch.autograd.grad(KNN, X, grad_outputs=torch.ones(X.shape),
                                            create_graph=True, retain_graph=True)
            dKNN2X = grad2KNNx[0]
            # -div(K(x,U)grad U(x)) = f(x) --->  -[KxUx+KyUy+K(Uxx+Uyy)]=f
            loss_it_temp = torch.mul(dKNN2X, dUNN2X) + torch.mul(KNN, dUNNxx) + torch.reshape(force_side, shape=[-1, 1])
            square_loss_it = torch.square(loss_it_temp)
            loss_it = torch.mean(square_loss_it)
            return UNN, KNN, loss_it
        elif str.lower(loss_type) == 'ritz_loss':
            square_norm2dUNN = torch.square(dUNN2X)
            ritz2dUNN = 0.5 * torch.mul(KNN, square_norm2dUNN) - torch.mul(force_side, UNN)
            loss_it = torch.mean(ritz2dUNN)
            return UNN, KNN, loss_it
        elif str.lower(loss_type) == 'lncosh' or str.lower(loss_type) == 'lncosh_loss':
            grad2UNNxx = torch.autograd.grad(dUNN2X, X, grad_outputs=torch.ones(X.shape),
                                             create_graph=True, retain_graph=True)
            dUNNxx = grad2UNNxx[0]

            grad2KNNx = torch.autograd.grad(KNN, X, grad_outputs=torch.ones(X.shape), create_graph=True,
                                            retain_graph=True)
            dKNN2X = grad2KNNx[0]
            # -div(K(x)grad U(x)) = f(x) --->  -[KxUx+KUxx]=f
            loss_it_temp = torch.mul(dKNN2X, dUNN2X) + torch.mul(KNN, dUNNxx) + torch.reshape(force_side, shape=[-1, 1])
            logcosh_loss_it = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss_it_temp))
            loss_it = torch.mean(logcosh_loss_it)
            return UNN, KNN, loss_it

    # 0 阶导数边界条件(Dirichlet 边界)
    def loss_bd2dirichlet(self, X_bd=None,
                          Ubd_exact=None,
                          if_lambda2Ubd=True,
                          loss_type='l2_loss',
                          scale2lncosh=0.5):
        assert (X_bd is not None)
        assert (Ubd_exact is not None)

        shape2X = np.shape(X_bd)
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd)
        else:
            Ubd = Ubd_exact

        UNN_bd = self.DNN2Solu(X_bd, scale=self.factor2freq, sFourier=self.sFourier)
        diff_bd = UNN_bd - Ubd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd_square = torch.square(diff_bd)
            loss_bd = torch.mean(loss_bd_square)
            return loss_bd
        elif str.lower(loss_type) == 'lncosh' or str.lower(loss_type) == 'lncosh_loss':
            loss_bd_lncosh = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * diff_bd))
            loss_bd = torch.mean(loss_bd_lncosh)
            return loss_bd

    # 1 阶导数边界条件(Dirichlet 边界)
    def loss_bd2neumann(self, X_bd=None,
                        Ubd_exact=None,
                        if_lambda2Ubd=True,
                        name2bd='left_bd',
                        loss_type='l2_loss',
                        scale2lncosh=0.5):
        assert (X_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = np.shape(X_bd)
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 1)

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd)
        else:
            Ubd = Ubd_exact

        UNN_bd = self.DNN2Solu(X_bd, scale=self.factor2freq, sFourier=self.sFourier)
        grad2UNNx = torch.autograd.grad(UNN_bd, X_bd, grad_outputs=torch.ones_like(X_bd.shape),
                                        create_graph=True, retain_graph=True)
        dUNN_bd = grad2UNNx[0]

        diff_bd = dUNN_bd - Ubd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd_square = torch.square(diff_bd)
            loss_bd = torch.mean(loss_bd_square)
            return loss_bd
        elif str.lower(loss_type) == 'lncosh':
            loss_bd_lncosh = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * diff_bd))
            loss_bd = torch.mean(loss_bd_lncosh)
            return loss_bd

    def loss_init(self, X=None,
                  tinit=None,
                  Uinit_exact=None,
                  if_lambda2Uinit=True,
                  loss_type='l2_loss',
                  scale2lncosh=0.5):
        assert (X is not None)
        assert (tinit is not None)
        assert (Uinit_exact is not None)

        shape2X = X.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2Uinit:
            Uinit = Uinit_exact(X)
        else:
            Uinit = Uinit_exact

        XT = torch.matmul(X, self.mat2X) + torch.matmul(tinit, self.mat2T)
        UNN_init = self.DNN2Solu(XT, scale=self.factor2freq, sFourier=self.sFourier)
        diif_init = UNN_init - Uinit

        if str.lower(loss_type) == 'l2_loss':
            loss_init_square = torch.square(diif_init)
            loss_init = torch.mean(loss_init_square)
            return loss_init
        elif str.lower(loss_type) == 'lncosh' or str.lower(loss_type) == 'lncosh_loss':
            loss_init_lncosh = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * diif_init))
            loss_init = torch.mean(loss_init_lncosh)
            return loss_init

    def loss_exactSolu(self, X=None,
                       Uexact=None,
                       if_lambda2Uexact=True,
                       loss_type='l2_loss',
                       scale2lncosh=0.5):
        assert (X is not None)
        assert (Uexact is not None)

        shape2X = np.shape(X)
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2Uexact:
            Usolu = Uexact(X)
        else:
            Usolu = Uexact
        UNN_exact = self.DNN2Solu(X, scale=self.factor2freq, sFourier=self.sFourier)
        diff_exact = UNN_exact - Usolu
        if str.lower(loss_type) == 'l2_loss':
            loss_exact_square = torch.square(diff_exact)
            loss_exact = torch.mean(loss_exact_square)
            return loss_exact
        elif str.lower(loss_type) == 'lncosh' or str.lower(loss_type) == 'lncosh_loss':
            loss_exact_lncosh = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh*diff_exact))
            loss_exact = torch.mean(loss_exact_lncosh)
            return loss_exact

    def loss_exactPara(self, X=None,
                       Kexact=None,
                       if_lambda2Kexact=True,
                       loss_type='l2_loss',
                       scale2lncosh=0.5):
        assert (X is not None)
        assert (Kexact is not None)

        shape2X = np.shape(X)
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2Kexact:
            Kpara = Kexact(X)
        else:
            Kpara = Kexact
        KNN_exact = self.DNN2Para(X, scale=self.freq2ParaModel, sFourier=self.sFourier)
        diff_exact = KNN_exact - Kpara
        if str.lower(loss_type) == 'l2_loss':
            loss_exact_square = torch.square(diff_exact)
            loss_exact = torch.mean(loss_exact_square)
            return loss_exact
        elif str.lower(loss_type) == 'lncosh' or str.lower(loss_type) == 'lncosh_loss':
            loss_exact_lncosh = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * diff_exact))
            loss_exact = torch.mean(loss_exact_lncosh)
            return loss_exact

    def loss_exactPara2NonLinear(self, X=None,
                                 Kexact=None,
                                 if_lambda2Kexact=True,
                                 loss_type='l2_loss',
                                 scale2lncosh=0.5):
        assert (X is not None)
        assert (Kexact is not None)

        shape2X = np.shape(X)
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2Kexact:
            Kpara = Kexact(X)
        else:
            Kpara = Kexact

        UNN = self.DNN2Solu(X, scale=self.factor2freq, sFourier=self.sFourier)
        XU = torch.matmul(X, self.mat2X) + torch.matmul(UNN, self.mat2U)
        KNN_exact = self.DNN2Para(XU, scale=self.freq2ParaModel, sFourier=self.sFourier)
        diff_exact = KNN_exact - Kpara
        if str.lower(loss_type) == 'l2_loss':
            loss_exact_square = torch.square(diff_exact)
            loss_exact = torch.mean(loss_exact_square)
            return loss_exact
        elif str.lower(loss_type) == 'lncosh' or str.lower(loss_type) == 'lncosh_loss':
            loss_exact_lncosh = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * diff_exact))
            loss_exact = torch.mean(loss_exact_lncosh)
            return loss_exact

    def get_regularSum2WB(self):
        sum2WB_Solu = self.DNN2Solu.get_regular_sum2WB(self.opt2regular_WB)
        sum2WB_Para = self.DNN2Para.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB_Solu + sum2WB_Para

    def evalue_DNN2Linear_Inverse(self, X_points=None):
        assert (X_points is not None)
        UNN = self.DNN2Solu(X_points, scale=self.factor2freq, sFourier=self.sFourier)
        KNN = self.DNN2Para(X_points, scale=self.freq2ParaModel, sFourier=self.sFourier)
        return UNN, KNN

    def evalue_DNN2NonLinear_Inverse(self, X_points=None):
        assert (X_points is not None)
        UNN = self.DNN2Solu(X_points, scale=self.factor2freq, sFourier=self.sFourier)

        XYU = torch.matmul(X_points, self.mat2X) + torch.matmul(UNN, self.mat2U)
        KNN = self.DNN2Para(XYU, scale=self.freq2ParaModel, sFourier=self.sFourier)
        return UNN, KNN


def solve_PDE(R):
    log_out_path = R['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Log_Print.dictionary2Inverse_porblem(R, log_fileout)

    # 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']
    batchsize_obser = R['batch_size2observe']
    batchsize_test = R['batch_size2test']

    bd_penalty_init = R['init_boundary_penalty']  # Regularization parameter for boundary conditions
    solu_penalty_init = R['init_solu_penalty']
    penalty2WB = R['penalty2weight_biases']  # Regularization parameter for weights and biases
    # lr_decay = R['learning_rate_decay']
    init_lr = R['learning_rate']
    act_func = R['name2act_hidden']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    region_l = 0.0
    region_r = 1.0
    if R['PDE_type'] == 'Linear_diffusion':
        # -div(K(x)grad U(x)) = f(x)
        # U(x) = g   在边界上, x 为边界点
        if R['equa_name'] == 'Linear_diffusion1':
            lam1 = 1
            lam2 = 5
            mu1 = 2
            mu2 = 4
            u_true = lambda x: torch.sin(lam1*np.pi * x) + 0.05*torch.sin(lam2*np.pi * x)
            k_true = lambda x: 2.0 + torch.sin(mu1 * np.pi * x) + torch.cos(mu2 * np.pi * x)

            u_left = lambda x: torch.sin(lam1*np.pi * region_l) + 0.05*torch.sin(lam2*np.pi * region_l)
            u_right = lambda x:  torch.sin(lam1*np.pi * region_r) + 0.05*torch.sin(lam2*np.pi * region_r)
            f = lambda x: -(mu1*np.pi*torch.cos(mu1*np.pi*x) - mu2*np.pi*torch.sin(mu2*np.pi*x))*\
                             (lam1*np.pi*torch.cos(lam1*np.pi*x)+0.05*lam2*np.pi*torch.cos(lam2*np.pi*x)) - \
                             (2.0 + torch.sin(mu1 * np.pi * x) + torch.cos(mu2 * np.pi * x))*\
                             (-lam1*lam1*np.pi*np.pi*torch.sin(lam1*np.pi*x)-0.05*lam2*lam2*np.pi*np.pi*torch.sin(lam2*np.pi*x))

    inverse_dnn = DNN2Inverse(input_dim=R['input_dim'], out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
                              Model_name=R['model2NN'], name2actIn=R['name2act_in'], name2actHidden=R['name2act_hidden'],
                              name2actOut=R['name2act_out'], opt2regular_WB='L0', type2numeric='float32',
                              factor2freq=R['freq'], sFourier=R['sfourier'], input_dim2Para=R['input_dim2Para'],
                              out_dim2Para=R['output_dim2Para'], hiddes2Para=R['hiddens2Para'],
                              ModelName2Para=R['Model2Para'], name_actIn2Para=R['nameIn2Para'],
                              name_actHidden2Para=R['nameHidden2Para'], name_actOut2Para=R['nameOut2Para'],
                              freq2Para=R['freq2Para'], repeat_highFreq2Solu=R['repeat_high_freq2solu'],
                              repeat_highFreq2Para=R['repeat_high_freq2para'])

    if True == R['use_gpu']:
        inverse_dnn = inverse_dnn.cuda(device='cuda:'+str(R['gpuNo']))

    params2Net = inverse_dnn.DNN.parameters()

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    optimizer = torch.optim.Adam(params2Net, lr=init_lr)                      # Adam

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.995)

    if R['testData_model'] == 'random_generate':
        x_it2test_batch = np.reshape(np.linspace(region_l, region_r, num=batchsize_test), [-1, 1])
        saveData.save_testData_or_solus2mat(x_it2test_batch, dataName='testX', outPath=R['FolderName'])
    else:
        mat_mesh2data_path = 'data2mesh'
        x_it2test_batch = Load_data2Mat.get_meshData(mesh_number=R['mesh_num'], data_path=mat_mesh2data_path)
        saveData.save_testData_or_solus2mat(x_it2test_batch, dataName='testX', outPath=R['FolderName'])

    if R['observeData_model'] == 'random_generate':
        # x_obsrve = DNN_data.rand_it(batchsize_obser, input_dim, region_a=region_l, region_b=region_r)
        x_obsrve = np.reshape(np.linspace(region_l, region_r, num=batchsize_obser), [-1, 1])
        saveData.save_testData_or_solus2mat(x_obsrve, dataName='Xobserve', outPath=R['FolderName'])
    else:
        observe_mat2data_path = 'data2observe'
        x_obsrve = Load_data2Mat.get_observeData(num3observe=R['batch_size2observe'], data_path=observe_mat2data_path)
        saveData.save_testData_or_solus2mat(x_obsrve, dataName='Xobserve', outPath=R['FolderName'])

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, loss_solu_all, loss_para_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    train_mse2Para_all, train_rel2Para_all, train_mse2Solu_all, train_rel2Solu_all = [], [], [], []
    test_mse2Para_all, test_rel2Para_all, test_mse2Solu_all, test_rel2Solu_all = [], [], [], []
    test_epoch = []

    Ktrue2test = k_true(x_it2test_batch)
    Utrue2test = u_true(x_it2test_batch)

    for i_epoch in range(R['max_epoch'] + 1):
        x_it_batch = dataUtilizer2torch.rand_it(batchsize_it, input_dim, region_a=region_l, region_b=region_r)
        xl_bd_batch, xr_bd_batch = dataUtilizer2torch.rand_bd_1D(batchsize_bd, input_dim,
                                                                 region_a=region_l, region_b=region_r)
        if R['activate_penalty2bd_increase'] == 1:
            if i_epoch < int(R['max_epoch'] / 10):
                temp_penalty_bd = bd_penalty_init
            elif i_epoch < int(R['max_epoch'] / 5):
                temp_penalty_bd = 10 * bd_penalty_init
            elif i_epoch < int(R['max_epoch'] / 4):
                temp_penalty_bd = 50 * bd_penalty_init
            elif i_epoch < int(R['max_epoch'] / 2):
                temp_penalty_bd = 100 * bd_penalty_init
            elif i_epoch < int(3 * R['max_epoch'] / 4):
                temp_penalty_bd = 200 * bd_penalty_init
            else:
                temp_penalty_bd = 500 * bd_penalty_init
        else:
            temp_penalty_bd = bd_penalty_init

        if R['activate_penalty2solu_increase'] == 1:
            if i_epoch < int(R['max_epoch'] / 10):
                temp_penalty_solu = solu_penalty_init
            elif i_epoch < int(R['max_epoch'] / 5):
                temp_penalty_solu = 10 * solu_penalty_init
            elif i_epoch < int(R['max_epoch'] / 4):
                temp_penalty_solu = 50 * solu_penalty_init
            elif i_epoch < int(R['max_epoch'] / 2):
                temp_penalty_solu = 100 * solu_penalty_init
            elif i_epoch < int(3 * R['max_epoch'] / 4):
                temp_penalty_solu = 200 * solu_penalty_init
            else:
                temp_penalty_solu = 500 * solu_penalty_init
        else:
            temp_penalty_solu = solu_penalty_init

        if R['PDE_type'] == 'Linear_diffusion':
            UNN2train, KNN2train, loss_it = inverse_dnn.loss_linear(
                X=x_it_batch, fside=f, loss_type=R['loss_type'])
        elif R['PDE_type'] == 'NonLinear_diffusion':
            UNN2train, KNN2train, loss_it = inverse_dnn.loss_nonlinear(
                X=x_it_batch, fside=f, loss_type=R['loss_type'])

        loss_bd2left = inverse_dnn.loss_bd2dirichlet(X_bd=xl_bd_batch, Ubd_exact=u_left, loss_type=R['loss_type'])
        loss_bd2right = inverse_dnn.loss_bd2dirichlet(X_bd=xr_bd_batch, Ubd_exact=u_right, loss_type=R['loss_type'])
        loss_bd = loss_bd2left + loss_bd2right

        loss2exact_solu = inverse_dnn.loss_exactSolu(X=x_obsrve, Uexact=u_true, loss_type=R['loss_type'])

        if R['PDE_type'] == 'Linear_diffusion':
            loss2exact_para = inverse_dnn.loss_exactPara(X=x_obsrve, Kexact=k_true, loss_type=R['loss_type'])
        elif R['PDE_type'] == 'NonLinear_diffusion':
            loss2exact_para = inverse_dnn.loss_exactPara2NonLinear(X=x_obsrve, Kexact=k_true, loss_type=R['loss_type'])

        regularSum2WB = inverse_dnn.get_regularSum2WB()
        PWB = penalty2WB * regularSum2WB

        loss = loss_it + temp_penalty_bd * loss_bd + temp_penalty_solu*(loss2exact_solu + loss2exact_para) + PWB

        loss_all.append(loss.item())
        loss_it_all.append(loss_it.item())
        loss_solu_all.append(loss2exact_solu.item())
        loss_para_all.append(loss2exact_para.item())
        loss_bd_all.append(loss_bd.item())

        optimizer.zero_grad()                        # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()                              # 对loss关于Ws和Bs求偏导
        optimizer.step()                             # 更新参数Ws和Bs
        scheduler.step()

        # 训练上的真解值和训练结果的误差
        Ktrue2train = k_true(x_it_batch)
        train_MSE2Para = torch.mean(torch.square(Ktrue2train - KNN2train))
        train_REL2Para = train_MSE2Para / torch.mean(torch.square(Ktrue2train))

        Utrue2train = u_true(x_it_batch)
        train_MSE2Solu = torch.mean(torch.square(Utrue2train - UNN2train))
        train_REL2Solu = train_MSE2Solu / torch.mean(torch.square(Utrue2train))

        train_mse2Para_all.append(train_MSE2Para.item())
        train_rel2Para_all.append(train_REL2Para.item())

        train_mse2Solu_all.append(train_MSE2Solu.item())
        train_rel2Solu_all.append(train_REL2Solu.item())

        if i_epoch % 1000 == 0:
            run_times = time.time() - t0
            tmp_lr = optimizer.param_groups[0]['lr']
            DNN_Log_Print.print_and_log_train_one_epoch2Inverse(
                i_epoch, run_times, tmp_lr, temp_penalty_bd, temp_penalty_solu, PWB.item(), loss_it.item(),
                loss_bd.item(), loss2exact_solu.item(), loss2exact_para.item(), loss.item(), train_MSE2Solu.item(),
                train_REL2Para.item(), train_MSE2Solu.item(), train_REL2Solu.item(), log_out=log_fileout)

            # ---------------------------   test network ----------------------------------------------
            if R['PDE_type'] == 'Linear_diffusion':
                UNN2test, KNN2test = inverse_dnn.evalue_DNN2Linear_Inverse(X_points=x_it2test_batch)
            elif R['PDE_type'] == 'NonLinear_diffusion':
                UNN2test, KNN2test = inverse_dnn.evalue_DNN2NonLinear_Inverse(X_points=x_it2test_batch)
            test_epoch.append(i_epoch / 1000)

            point_square_error2Para = torch.square(Ktrue2test - KNN2test)
            mse2para_test = torch.mean(point_square_error2Para)
            test_mse2Para_all.append(mse2para_test)
            rel2para_test = mse2para_test / torch.mean(torch.square(Ktrue2test))
            test_rel2Para_all.append(rel2para_test)

            point_square_error2Solu = torch.square(Utrue2test - UNN2test)
            mse2solu_test = torch.mean(point_square_error2Solu)
            test_mse2Solu_all.append(mse2solu_test)
            rel2solu_test = mse2solu_test / torch.mean(torch.square(Utrue2test))
            test_rel2Solu_all.append(rel2solu_test)

            DNN_Log_Print.print_and_log2test_Inverse(mse2para_test.item(), rel2para_test.item(),
                                                     mse2solu_test.item(), rel2solu_test.item(),
                                                     log_out=log_fileout)

    # -----------------------  save training results to mat files, then plot them ---------------------------------
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=act_func,
                                         outPath=R['FolderName'])
    saveData.save_train_MSE_REL2mat(train_mse2Para_all, train_rel2Para_all, actName='Para', outPath=R['FolderName'])

    saveData.save_train_MSE_REL2mat(train_mse2Solu_all, train_rel2Solu_all, actName='Solu', outPath=R['FolderName'])

    if R['loss_type'] == 'L2_loss':
        plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'],
                                          outPath=R['FolderName'], yaxis_scale=True)
    else:
        plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'],
                                          outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)

    plotData.plotTrain_MSE_REL_1act_func(train_mse2Para_all, train_rel2Para_all, actName='Para', seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    plotData.plotTrain_MSE_REL_1act_func(train_mse2Solu_all, train_rel2Solu_all, actName='Solu', seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
    saveData.save_2testSolus2mat(Ktrue2test, KNN2test, actName='Para', actName1='ParaNN',
                                 outPath=R['FolderName'])

    saveData.save_2testSolus2mat(Utrue2test, UNN2test, actName='Solu', actName1='SoluNN',
                                 outPath=R['FolderName'])
    saveData.save_testMSE_REL2mat(test_mse2Para_all, test_rel2Para_all, actName='Para', outPath=R['FolderName'])
    saveData.save_testMSE_REL2mat(test_mse2Solu_all, test_rel2Solu_all, actName='Solu', outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse2Para_all, test_rel2Para_all, test_epoch, actName='Para',
                              seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2Solu_all, test_rel2Solu_all, test_epoch, actName='Solu',
                              seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 0
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # ------------------------------------------- 文件保存路径设置 ----------------------------------------

    store_file = 'LDiffusion1D'
    # store_file = 'NLDiffusion1D'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])  # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        # tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # ---------------------------- Setup of laplace equation ------------------------------
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    # step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    # R['activate_stop'] = int(step_stop_flag)
    R['activate_stop'] = 1
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    # R['max_epoch'] = 10000
    R['max_epoch'] = 50000
    # if 0 != R['activate_stop']:
    #     epoch_stop = input('please input a stop epoch:')
    #     R['max_epoch'] = int(epoch_stop)

    if store_file == 'LDiffusion1D':
        R['PDE_type'] = 'Linear_diffusion'
        R['equa_name'] = 'Linear_diffusion1'
        # R['equa_name'] = 'Linear_diffusion2'
    elif store_file == 'NLDiffusion1D':
        R['PDE_type'] = 'NoLinear_Diffusion'
        R['equa_name'] = 'NoLinear_Diffusion1'
        # R['equa_name'] = 'NoLinear_Diffusion2'

    R['epsilon'] = 0.1
    R['order2pLaplace_operator'] = 2

    R['input_dim'] = 1  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    if store_file == 'LDiffusion1D':
        R['input_dim2Para'] = R['input_dim']
    elif store_file == 'NLDiffusion1D':
        R['input_dim2Para'] = R['input_dim'] + 1

    R['output_dim2Para'] = 1
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置(内部和边界)
    R['batch_size2interior'] = 2500                # 内部训练数据的批大小
    R['batch_size2boundary'] = 500                 # 边界训练数据大小
    R['batch_size2test'] = 2000

    # R['batch_size2observe'] = 10                # 10 个观测点，不收敛
    # R['batch_size2observe'] = 11                # 11 个观测点，不收敛
    R['batch_size2observe'] = 15                  # 收敛，但精度只有1e-5
    # R['batch_size2observe'] = 16                # 收敛，但精度只有1e-6
    # R['batch_size2observe'] = 18                # 收敛，但精度只有1e-6
    # R['batch_size2observe'] = 20
    # R['batch_size2observe'] = 25
    # R['batch_size2observe'] = 50
    # R['batch_size2observe'] = 100
    # R['batch_size2observe'] = 150
    # R['batch_size2observe'] = 200

    # 装载测试数据模式
    # R['testData_model'] = 'loadData'
    R['testData_model'] = 'random_generate'

    # R['observeData_model'] = 'loadData'
    R['observeData_model'] = 'random_generate'

    # 装载测试数据模式和画图
    R['plot_ongoing'] = 0
    R['subfig_type'] = 1

    R['loss_type'] = 'L2_loss'                            # l2 loss
    # R['loss_type'] = 'lncosh'                           # lncosh loss

    R['optimizer_name'] = 'Adam'                          # 优化器
    R['learning_rate'] = 2e-4                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['train_model'] = 'union_training'
    # R['train_model'] = 'group2_training'
    # R['train_model'] = 'group3_training'
    # R['train_model'] = 'group4_training'

    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    R['penalty2weight_biases'] = 0.000                    # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001                  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                 # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    # R['activate_penalty2bd_increase'] = 0
    R['activate_penalty2bd_increase'] = 1

    if R['activate_penalty2bd_increase'] == 0:
        R['init_boundary_penalty'] = 1000                 # Regularization parameter for boundary conditions
    else:
        R['init_boundary_penalty'] = 10                   # Regularization parameter for boundary conditions

    # 边界的惩罚处理方式,以及边界的惩罚因子
    # R['activate_penalty2solu_increase'] = 0
    R['activate_penalty2solu_increase'] = 1

    if R['activate_penalty2solu_increase'] == 0:
        R['init_solu_penalty'] = 1000                     # Regularization parameter for boundary conditions
    else:
        R['init_solu_penalty'] = 10                       # Regularization parameter for boundary conditions

    # 网络的频率范围设置
    # R['freq'] = np.arange(1, 31)
    # R['freq'] = np.arange(1, 51)
    R['freq'] = np.arange(1, 101)
    # R['freq'] = np.random.normal(1, 120, 100)

    # R['freq2Para'] = np.arange(1, 31)
    # R['freq2Para'] = np.arange(1, 51)
    R['freq2Para'] = np.arange(1, 101)
    # R['freq2Para'] = np.random.normal(1, 120, 100)

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model2NN'] = 'DNN'
    # R['model2NN'] = 'Scale_DNN'
    # R['model2NN'] = 'Adapt_scale_DNN'
    R['model2NN'] = 'Fourier_DNN'
    # R['model2NN'] = 'Wavelet_DNN'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model2NN'] == 'Fourier_DNN':
        R['hidden_layers'] = (125, 80, 60, 60, 40)  # 1*125+250*100+100*80+80*80+80*60+60*1= 44385 个参数
        # R['hidden_layers'] = (125, 120, 80, 80, 80)  # 1*125+250*120+120*80+80*80+80*80+80*1= 52605 个参数
        # R['hidden_layers'] = (125, 150, 100, 100, 80)  # 1*125+250*150+150*100+100*100+100*80+80*1= 70705 个参数
    else:
        R['hidden_layers'] = (250, 80, 60, 60, 40)  # 1*250+250*100+100*80+80*80+80*60+60*1= 44510 个参数
        # R['hidden_layers'] = (250, 120, 80, 80, 80)  # 1*250+250*120+120*80+80*80+80*80+80*1= 52730 个参数
        # R['hidden_layers'] = (250, 150, 100, 100, 80)  # 1*250+250*150+150*100+100*100+100*80+80*1= 70830 个参数

    # R['Model2Para'] = 'DNN'
    # R['Model2Para'] = 'Scale_DNN'
    R['Model2Para'] = 'Fourier_DNN'

    if R['Model2Para'] == 'Fourier_DNN':
        R['hiddens2Para'] = (125, 100, 50, 50)
    else:
        R['hiddens2Para'] = (250, 100, 50, 50)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['name2act_in'] = 'relu'
    # R['name2act_in'] = 'elu'
    # R['name2act_in'] = 'gelu'
    # R['name2act_in'] = 'mgelu'
    # R['name2act_in'] = 'mish'
    # R['name2act_in'] = 'gauss'
    # R['name2act_in'] = 'gcu'
    R['name2act_in'] = 'tanh'
    # R['name2act_in'] = 's2relu'
    # R['name2act_in'] = 'sin'
    # R['name2act_in'] = 'sinADDcos'

    # R['name2act_hidden'] = 'relu'
    # R['name2act_hidden'] = 'elu'
    # R['name2act_hidden'] = 'gelu'
    # R['name2act_hidden'] = 'mgelu'
    # R['name2act_hidden'] = 'mish'
    # R['name2act_hidden'] = 'gauss'
    # R['name2act_hidden'] = 'gcu'
    R['name2act_hidden'] = 'tanh'
    # R['name2act_hidden'] = 'srelu'
    # R['name2act_hidden'] = 's2relu'
    # R['name2act_hidden'] = 'sin'
    # R['name2act_hidden'] = 'sinADDcos'
    # R['name2act_hidden'] = 'phi'

    R['name2act_out'] = 'linear'

    R['nameIn2Para'] = 'tanh'
    # R['nameIn2Para'] = 'gcu'
    # R['nameIn2Para'] = 'gelu'
    # R['nameIn2Para'] = 'mgelu'
    # R['nameIn2Para'] = 'mish'
    # R['nameIn2Para'] = 'gauss'
    # R['nameIn2Para'] = 'sin'

    R['nameHidden2Para'] = 'tanh'
    # R['nameHidden2Para'] = 'gcu'
    # R['nameHidden2Para'] = 'gelu'
    # R['nameHidden2Para'] = 'mgelu'
    # R['nameHidden2Para'] = 'mish'
    # R['nameHidden2Para'] = 'gauss'
    # R['nameHidden2Para'] = 's2relu'
    # R['nameHidden2Para'] = 'sin'
    # R['nameHidden2Para'] = 'sinADDcos'

    R['nameOut2Para'] = 'linear'

    R['sfourier'] = 1.0
    if R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'tanh':
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 's2relu':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'sinADDcos':
        R['sfourier'] = 0.5
        # R['sfourier'] = 1.0
    else:
        R['sfourier'] = 1.0

    R['repeat_high_freq2solu'] = True
    R['repeat_high_freq2para'] = False
    solve_PDE(R)

    # Ritz loss 不收敛
    # lncosh loss 不收敛
    # Fourier(1.0) + tanh 才有效果
    # 对于多尺度震荡问题，尺度因子要选的大一些
