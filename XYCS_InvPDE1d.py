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
import itertools
import torch.nn.functional as tnf
from torch.nn.parameter import Parameter


import DNN_base
import dataUtilizer2torch
import Load_data2Mat
import MS_StokeEqs
import saveData
import plotData
import DNN_Log_Print

'''
Reference: Generalized finite difference method for solving stationary 2D and 3D Stokes equations with a mixed boundary condition
'''
class DNN2NS(tn.Module):
    def __init__(self, input_dim=2, out_dim=2, hidden_layer=None, Model_name='DNN', name2actIn='relu',
                 name2actHidden='relu', name2actOut='linear',  input_dim2P=2, out_dim2P=1, hidden_layer2P=None,
                 Model_Name2P='DNN', name_actIn2P='tanh', name_actHidden2P='relu', name_actOut2P='linear',
                 freq2Solu=None, sFourier2Solu=1.0, repeat_highFreq2Solu=False, freq2P=None,  sFourier2P=1.0,
                 repeat_highFreq2P=False, opt2regular_WB='L2',  type2numeric='float32', use_gpu=False, No2GPU=0):
        super(DNN2NS, self).__init__()
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

        if 'DNN' == str.upper(Model_Name2P):
            self.DNN2P = DNN_base.Pure_DenseNet(
                indim=input_dim2P, outdim=out_dim2P, hidden_units=hidden_layer2P, name2Model=Model_Name2P,
                actName2in=name_actIn2P, actName=name_actHidden2P, actName2out=name_actOut2P,
                type2float=type2numeric, scope2W='UK', scope2B='BK', to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'SCALE_DNN' == str.upper(Model_Name2P) or 'DNN_SCALE' == str.upper(Model_Name2P):
            self.DNN2P = DNN_base.Dense_ScaleNet(
                indim=input_dim2P, outdim=out_dim2P, hidden_units=hidden_layer2P, name2Model=Model_Name2P,
                actName2in=name_actIn2P, actName=name_actHidden2P, actName2out=name_actOut2P,
                type2float=type2numeric, scope2W='UK', scope2B='BK', repeat_Highfreq=repeat_highFreq2P,
                to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'FOURIER_DNN' == str.upper(Model_Name2P) or 'DNN_FOURIERBASE' == str.upper(Model_Name2P):
            self.DNN2P = DNN_base.Dense_FourierNet(
                indim=input_dim2P, outdim=out_dim2P, hidden_units=hidden_layer2P, name2Model=Model_Name2P,
                actName2in=name_actIn2P, actName=name_actHidden2P, actName2out=name_actOut2P,
                type2float=type2numeric, scope2W='UK', scope2B='BK', repeat_Highfreq=repeat_highFreq2P,
                to_gpu=use_gpu, gpu_no=No2GPU)

        if type2numeric == 'float32':
            self.float_type = torch.float32
        elif type2numeric == 'float64':
            self.float_type = torch.float64
        elif type2numeric == 'float16':
            self.float_type = torch.float16

        self.freq2Solu = freq2Solu
        self.freq2PModel = freq2P

        self.sFourier2Solu = sFourier2Solu
        self.sFourier2P = sFourier2P

        self.opt2regular_WB = opt2regular_WB

        self.use_gpu = use_gpu
        if use_gpu:
            self.opt2device = 'cuda:' + str(No2GPU)
        else:
            self.opt2device = 'cpu'

        if use_gpu:
            self.mat2U1 = torch.tensor([[1], [0]], dtype=self.float_type, device=self.opt2device, requires_grad=False)  # 2 行 1 列
            self.mat2U2 = torch.tensor([[0], [1]], dtype=self.float_type, device=self.opt2device, requires_grad=False)  # 2 行 1 列
            self.mat2U1U2 = torch.tensor([[0], [1]], dtype=self.float_type, device=self.opt2device, requires_grad=False)  # 2 行 1 列
        else:
            self.mat2U1 = torch.tensor([[1], [0]], dtype=self.float_type, requires_grad=False)  # 2 行 1 列
            self.mat2U2 = torch.tensor([[0], [1]], dtype=self.float_type, requires_grad=False)  # 2 行 1 列
            self.mat2U1U2 = torch.tensor([[0], [1]], dtype=self.float_type, requires_grad=False)  # 2 行 1 列

    def loss_it(self, XY=None,
                      fside1=None,
                      fside2=None,
                      if_lambda2fside=True,
                      loss_type='l2_loss'):
        assert (XY is not None)
        assert (fside1 is not None)
        assert (fside2 is not None)
        shape2XY = XY.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)
        X = torch.reshape(XY[:, 0], shape=[-1, 1])
        Y = torch.reshape(XY[:, 1], shape=[-1, 1])
        if if_lambda2fside:
            force_side1 = fside1(X,Y)
            force_side2 = fside2(X,Y)
        else:
            force_side1 = fside1
            force_side2 = fside2
        UNN = self.DNN2Solu(XY, scale=self.freq2Solu, sFourier=self.sFourier2Solu) #二维 out:n*2
        UNN1 = torch.matmul(UNN, self.mat2U1)
        UNN2 = torch.matmul(UNN, self.mat2U2)
        KNN = self.DNN2P(XY, scale=self.freq2PModel, sFourier=self.sFourier2P)
        grad2UNN1 = torch.autograd.grad(UNN1, XY, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True) #输出的是张量 out:n*2
        dUNN1 = grad2UNN1[0] # out: n*2
        grad2UNN2 = torch.autograd.grad(UNN2, XY, grad_outputs=torch.ones_like(X),create_graph=True, retain_graph=True)  # 输出的是张量 out:n*2
        dUNN2 = grad2UNN2[0]  # out: n*2
        if str.lower(loss_type) == 'l2_loss':
            dUNN1x = torch.reshape(dUNN1[:,0], shape=[-1,1])
            dUNN1y = torch.reshape(dUNN1[:,1], shape=[-1, 1])
            dUNN2x = torch.reshape(dUNN2[:,0], shape=[-1, 1])
            dUNN2y = torch.reshape(dUNN2[:,1], shape=[-1, 1])
            grad_UNN1x = torch.autograd.grad(dUNN1x, XY, grad_outputs=torch.ones_like(X),
                                             create_graph=True, retain_graph=True)
            d2UNN1xxy = grad_UNN1x[0]
            grad_UNN1y = torch.autograd.grad(dUNN1y, XY, grad_outputs=torch.ones_like(X),
                                             create_graph=True, retain_graph=True)
            d2UNN1yxy = grad_UNN1y[0]
            grad_UNN2x = torch.autograd.grad(dUNN2x, XY, grad_outputs=torch.ones_like(X),
                                             create_graph=True, retain_graph=True)
            d2UNN2xxy = grad_UNN2x[0]
            grad_UNN2y = torch.autograd.grad(dUNN2y, XY, grad_outputs=torch.ones_like(X),
                                             create_graph=True, retain_graph=True)
            d2UNN2yxy = grad_UNN2y[0]
            grad2KNN = torch.autograd.grad(KNN, XY, grad_outputs=torch.ones_like(X),
                                            create_graph=True, retain_graph=True)
            dUNN1xx = torch.reshape(d2UNN1xxy[:,0], shape=[-1,1])
            dUNN1yy = torch.reshape(d2UNN1yxy[:,1], shape=[-1,1])
            dUNN2xx = torch.reshape(d2UNN2xxy[:,0], shape=[-1, 1])
            dUNN2yy = torch.reshape(d2UNN2yxy[:,1], shape=[-1, 1])
            dKNN = grad2KNN[0]
            dKNN2X = torch.reshape(dKNN[:, 0], shape=[-1, 1])
            dKNN2Y = torch.reshape(dKNN[:, 1], shape=[-1, 1])

            loss_it_Eq11 = torch.reshape(force_side1, shape=[-1, 1]) + torch.add(dUNN1xx, dUNN1yy) - dKNN2X #u1的第一个方程
            loss_it_Eq12 = torch.reshape(force_side2, shape=[-1, 1]) + torch.add(dUNN2xx, dUNN2yy) - dKNN2Y #u2
            loss_it_Eq21 = torch.sum(dUNN1, dim=-1) #make sure the size is n*1
            loss_it_Eq22 = torch.sum(dUNN2, dim=-1)
            square_loss_it11 = torch.square(loss_it_Eq11)
            square_loss_it12 = torch.square(loss_it_Eq12)
            square_loss_it21 = torch.square(loss_it_Eq21)
            square_loss_it22 = torch.square(loss_it_Eq22)
            loss_it11 = torch.mean(square_loss_it11)
            loss_it12 = torch.mean(square_loss_it12)
            loss_it21 = torch.mean(square_loss_it21)
            loss_it22 = torch.mean(square_loss_it22)
            loss_in = loss_it11 + loss_it12
            loss_in2div = loss_it21 + loss_it22
            return UNN, UNN1, UNN2, KNN, loss_in, loss_in2div

    def loss_bd(self, XY_bd=None,
                Ubd_exact1=None,
                Ubd_exact2=None,
                if_lambda2Ubd=True,
                loss_type='l2_loss',
                scale2lncosh=0.5):
        assert (XY_bd is not None)
        shape2X = np.shape(XY_bd)
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 2)
        X_bd = torch.reshape(XY_bd[:, 0], shape=[-1, 1])
        Y_bd = torch.reshape(XY_bd[:, 1], shape=[-1, 1])
        if if_lambda2Ubd:
            Ubd1 = Ubd_exact1(X_bd, Y_bd)
            Ubd2 = Ubd_exact2(X_bd, Y_bd)
        else:
            Ubd1 = Ubd_exact1
            Ubd2 = Ubd_exact2
        UNN_bd = self.DNN2Solu(XY_bd, scale=self.freq2Solu, sFourier=self.sFourier2Solu)
        UNN1_bd = torch.matmul(UNN_bd, self.mat2U1)
        UNN2_bd = torch.matmul(UNN_bd, self.mat2U2)
        diff1_bd = UNN1_bd - Ubd1
        diff2_bd = UNN2_bd - Ubd2
        if str.lower(loss_type) == 'l2_loss':
            loss1_bd_square = torch.square(diff1_bd)
            loss1_bd = torch.mean(loss1_bd_square)
            loss2_bd_square = torch.square(diff2_bd)
            loss2_bd = torch.mean(loss2_bd_square)
            loss_bd = loss1_bd + loss2_bd
            return loss_bd

    #p是（内部+边界）上积分之和为0
    def loss_p(self, XY_bd_l=None,
                  XY_bd_r=None,
                  XY_bd_t=None,
                  XY_bd_b=None,
                  XY_it=None
                  ):
        assert (XY_bd_l is not None)
        assert (XY_bd_r is not None)
        assert (XY_bd_t is not None)
        assert (XY_bd_b is not None)
        assert (XY_it is not None)
        shape2XY_l = np.shape(XY_bd_l)
        lenght2XY_l_shape = len(shape2XY_l)
        assert (lenght2XY_l_shape == 2)
        assert (shape2XY_l[-1] == 2)
        KNN_bd_l = self.DNN2P(XY_bd_l, scale=self.freq2PModel, sFourier=self.sFourier2P)
        KNN_bd_r = self.DNN2P(XY_bd_r, scale=self.freq2PModel, sFourier=self.sFourier2P)
        KNN_bd_t = self.DNN2P(XY_bd_t, scale=self.freq2PModel, sFourier=self.sFourier2P)
        KNN_bd_b = self.DNN2P(XY_bd_b, scale=self.freq2PModel, sFourier=self.sFourier2P)
        KNN_it = self.DNN2P(XY_it, scale=self.freq2PModel, sFourier=self.sFourier2P)
        KNN_bd_p = KNN_bd_l + KNN_bd_r + KNN_bd_t + KNN_bd_b
        loss2P_bd_temp = torch.sum(KNN_bd_p, dim=-1)
        loss2P_it_temp = torch.sum(KNN_it, dim=-1)
        loss2P_bd = torch.mean(loss2P_bd_temp)
        loss2P_it = torch.mean(loss2P_it_temp)
        loss2P = loss2P_bd + loss2P_it
        return loss2P

    def loss_exact(self, XY=None,
                       U1exact=None,
                       U2exact=None,
                       Pexact=None,
                       if_lambda2Kexact=True,
                       loss_type='l2_loss',
                       scale2lncosh=0.5):
        assert (XY is not None)
        assert (Pexact is not None)
        assert (U1exact is not None)
        assert (U2exact is not None)
        shape2XY = np.shape(XY)
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)
        X_p = torch.reshape(XY[:, 0], shape=[-1, 1])
        Y_p = torch.reshape(XY[:, 1], shape=[-1, 1])

        if if_lambda2Kexact:
            Kp = Pexact(X_p, Y_p)
            U1ture = U1exact(X_p, Y_p)
            U2ture = U2exact(X_p, Y_p)
        else:
            Kp = Pexact
            U1ture = U1exact
            U2ture = U2exact
        U_exact = self.DNN2Solu(XY,scale=self.freq2Solu, sFourier=self.sFourier2Solu)
        U1_exact = torch.matmul(U_exact, self.mat2U1)
        U2_exact = torch.matmul(U_exact, self.mat2U2)
        KNN_exact = self.DNN2P(XY, scale=self.freq2PModel, sFourier=self.sFourier2P)
        diff_exact_p = KNN_exact - Kp
        diff_exact_u1 = U1_exact - U1ture
        diff_exact_u2 = U2_exact - U2ture
        if str.lower(loss_type) == 'l2_loss':
            loss_exact_square_p = torch.square(diff_exact_p)
            loss_exact_p = torch.mean(loss_exact_square_p)
            loss_exact_square_u1 = torch.square(diff_exact_u1)
            loss_exact_u1 = torch.mean(loss_exact_square_u1)
            loss_exact_square_u2 = torch.square(diff_exact_u2)
            loss_exact_u2 = torch.mean(loss_exact_square_u2)
            return loss_exact_u1, loss_exact_u2, loss_exact_p

    def get_regularSum2WB(self):
        sum2WB_Solu = self.DNN2Solu.get_regular_sum2WB(self.opt2regular_WB)
        sum2WB_Para = self.DNN2P.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB_Solu + sum2WB_Para

    def evalue_DNN2NS(self, XY_points=None):
        assert (XY_points is not None)
        UNN = self.DNN2Solu(XY_points, scale=self.freq2Solu, sFourier=self.sFourier2Solu)
        KNN = self.DNN2P(XY_points, scale=self.freq2PModel, sFourier=self.sFourier2P)
        UNNX = torch.matmul(UNN, self.mat2U1)
        UNNY = torch.matmul(UNN, self.mat2U2)
        return UNN, UNNX, UNNY, KNN


def solve_PDE(R):
    log_out_path = R['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    #DNN_Log_Print.dictionary2Inverse_porblem(R, log_fileout)

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
    out_dim = R['out_dim']
    input_dim2P = R['inputdim2P']
    out_dim2P = R['outdim2P']

    region_l = -1.0
    region_r = 1.0
    if R['PDE_type'] == 'Navier_Stoke':
        f1 = lambda x, y: torch.zeros_like(x)
        f2 = lambda x, y: torch.zeros_like(x)
        u1_true = lambda x, y: 20 * x * (y ** 3)
        u2_true = lambda x, y: 5 * (x ** 4 - y ** 4)
        p_true = lambda x, y: 60 * (x ** 2) * y - 20 * (y ** 3)
        u1x_left = lambda x, y: 20 * x * (y ** 3)
        u1x_right = lambda x, y: 20 * x * (y ** 3)
        u1y_bottom = lambda x, y: 20 * x * (y ** 3)
        u1y_top = lambda x, y: 20 * x * (y ** 3)
        u2x_left = lambda x, y: 5 * (x ** 4 - y ** 4)
        u2x_right = lambda x, y: 5 * (x ** 4 - y ** 4)
        u2y_bottom = lambda x, y: 5 * (x ** 4 - y ** 4)
        u2y_top = lambda x, y: 5 * (x ** 4 - y ** 4)
    NS_dnn = DNN2NS(input_dim=R['input_dim'], out_dim=R['out_dim'], hidden_layer=R['hidden_layers'], Model_name=R['model2NN'], name2actIn=R['name2act_in'],
                 name2actHidden=R['name2act_hidden'], name2actOut=R['name2act_out'], input_dim2P=R['inputdim2P'], out_dim2P=R['outdim2P'], hidden_layer2P=R['hiddens2P'],
                 Model_Name2P=R['Model2P'], name_actIn2P=R['nameIn2P'], name_actHidden2P=R['nameHidden2P'], name_actOut2P=R['nameOut2P'],
                 freq2Solu=R['freq'], sFourier2Solu=R['sfourier'], repeat_highFreq2Solu=R['repeat_high_freq2solu'], freq2P=R['freq2P'],  sFourier2P=R['sfourier2p'],
                 repeat_highFreq2P=R['repeat_high_freq2p'], opt2regular_WB=R['regular_wb_model'],  type2numeric='float32', use_gpu=R['use_gpu'], No2GPU=R['gpuNo'])

    if True == R['use_gpu']:
        NS_dnn = NS_dnn.cuda(device='cuda:'+str(R['gpuNo']))

    params2Net_1 = NS_dnn.DNN2Solu.parameters()
    params2Net_2 = NS_dnn.DNN2P.parameters()

    params2Net = itertools.chain(params2Net_1, params2Net_2)

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
        xy_it2test_batch = dataUtilizer2torch.rand_it(batchsize_test, R['input_dim'], region_a=region_l, region_b=region_r,
                                                     to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'],
                                                     use_grad2x=True)
        saveData.save_testData_or_solus2mat(xy_it2test_batch, dataName='testX', outPath=R['FolderName'])
    else:
        mat_mesh2data_path = 'data2mesh'
        xy_it2test_batch = Load_data2Mat.get_meshData(mesh_number=R['mesh_num'], data_path=mat_mesh2data_path)
        saveData.save_testData_or_solus2mat(xy_it2test_batch, dataName='testX', outPath=R['FolderName'])

    if R['observeData_model'] == 'random_generate':
        # x_obsrve = DNN_data.rand_it(batchsize_obser, input_dim, region_a=region_l, region_b=region_r)
        x_obsrve = dataUtilizer2torch.rand_it(batchsize_obser, R['input_dim'], region_a=region_l, region_b=region_r,
                                                     to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'],
                                                     use_grad2x=True)
        saveData.save_testData_or_solus2mat(x_obsrve, dataName='Xobserve', outPath=R['FolderName'])
    else:
        observe_mat2data_path = 'data2observe'
        x_obsrve = Load_data2Mat.get_observeData(num3observe=R['batch_size2observe'], data_path=observe_mat2data_path)
        saveData.save_testData_or_solus2mat(x_obsrve, dataName='Xobserve', outPath=R['FolderName'])

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, loss_solu_all, loss_p_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    train_mse2P_all, train_rel2P_all, train_mse2Solu_all_1, train_rel2Solu_all_1, train_mse2Solu_all_2, train_rel2Solu_all_2 = [], [], [], [], [], []
    test_mse2P_all, test_rel2P_all, test_mse2Solu_all, test_rel2Solu_all, test_mse2Solu_all_2, test_rel2Solu_all_2 = [], [], [], [], [], []
    test_epoch = []
    U1true2test = u1_true(torch.reshape(xy_it2test_batch[:, 0], shape=[-1, 1]),
                                  torch.reshape(xy_it2test_batch[:, 1], shape=[-1, 1]))
    U2true2test = u2_true(torch.reshape(xy_it2test_batch[:, 0], shape=[-1, 1]),
                                  torch.reshape(xy_it2test_batch[:, 1], shape=[-1, 1]))
    Ptrue2test = p_true(torch.reshape(xy_it2test_batch[:, 0], shape=[-1, 1]),
                                  torch.reshape(xy_it2test_batch[:, 1], shape=[-1, 1]))
    for i_epoch in range(R['max_epoch'] + 1):
        xy_it_batch = dataUtilizer2torch.rand_it(batchsize_it, input_dim, region_a=region_l, region_b=region_r, to_torch=True,
                                                to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'], use_grad2x=True)
        xl_bd_batch, xr_bd_batch, xt_bd_batch, xb_bd_batch = dataUtilizer2torch.rand_bd_2D(batchsize_bd, input_dim,
                                                                 region_a=region_l, region_b=region_r, to_torch=True, to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'])
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

        if R['PDE_type'] == 'Navier_Stoke':
            UNN2train, UNN2trainx, UNN2trainy, KNN2train, loss_it, loss_it2div = NS_dnn.loss_it(
                XY=xy_it_batch, fside1=f1, fside2=f2, loss_type=R['loss_type'])

        loss_in = loss_it + loss_it2div
        loss_bd2left = NS_dnn.loss_bd(XY_bd=xl_bd_batch, Ubd_exact1=u1x_left, Ubd_exact2=u2x_left, loss_type=R['loss_type'])
        loss_bd2right = NS_dnn.loss_bd(XY_bd=xr_bd_batch, Ubd_exact1=u1x_right, Ubd_exact2=u2x_right, loss_type=R['loss_type'])
        loss_bd2top = NS_dnn.loss_bd(XY_bd=xt_bd_batch, Ubd_exact1=u1y_top, Ubd_exact2=u2y_top, loss_type=R['loss_type'])
        loss_bd2bottom = NS_dnn.loss_bd(XY_bd=xb_bd_batch, Ubd_exact1=u1y_bottom, Ubd_exact2=u2y_bottom, loss_type=R['loss_type'])
        loss_2P = NS_dnn.loss_p(XY_bd_l=xl_bd_batch, XY_bd_r=xr_bd_batch, XY_bd_t=xt_bd_batch, XY_bd_b=xb_bd_batch, XY_it=xy_it_batch)

        loss_bd = loss_bd2left + loss_bd2right + loss_bd2top + loss_bd2bottom
        loss_exact2u1, loss_exact2u2, loss_exact2p = NS_dnn.loss_exact(XY=xy_it_batch, U1exact=u1_true, U2exact=u2_true, Pexact=p_true)


        regularSum2WB = NS_dnn.get_regularSum2WB()
        PWB = penalty2WB * regularSum2WB

        loss = loss_in + temp_penalty_bd * loss_bd + PWB + loss_2P + temp_penalty_solu * (loss_exact2p)

        loss_all.append(loss.item())
        loss_it_all.append(loss_in.item())
        loss_bd_all.append(loss_bd.item())

        optimizer.zero_grad()                        # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()                              # 对loss关于Ws和Bs求偏导
        optimizer.step()                             # 更新参数Ws和Bs
        scheduler.step()

        # 训练上的真解值和训练结果的误差
        Ktrue2train = p_true(torch.reshape(xy_it_batch[:, 0], shape=[-1, 1]),
                                  torch.reshape(xy_it_batch[:, 1], shape=[-1, 1]))
        train_MSE2P = torch.mean(torch.square(Ktrue2train - KNN2train))
        train_REL2P = train_MSE2P / torch.mean(torch.square(Ktrue2train))

        U1true2train = u1_true(torch.reshape(xy_it_batch[:, 0], shape=[-1, 1]),
                                  torch.reshape(xy_it_batch[:, 1], shape=[-1, 1]))
        U2true2train = u2_true(torch.reshape(xy_it_batch[:, 0], shape=[-1, 1]),
                                  torch.reshape(xy_it_batch[:, 1], shape=[-1, 1]))
        train_MSE2Solu_1 = torch.mean(torch.square(U1true2train - UNN2trainx))
        train_REL2Solu_1 = train_MSE2Solu_1 / torch.mean(torch.square(U1true2train))
        train_MSE2Solu_2 = torch.mean(torch.square(U2true2train - UNN2trainy))
        train_REL2Solu_2 = train_MSE2Solu_2 / torch.mean(torch.square(U2true2train))

        train_mse2P_all.append(train_MSE2P.item())
        train_rel2P_all.append(train_REL2P.item())

        train_mse2Solu_all_1.append(train_MSE2Solu_1.item())
        train_rel2Solu_all_1.append(train_REL2Solu_1.item())
        train_mse2Solu_all_2.append(train_MSE2Solu_2.item())
        train_rel2Solu_all_2.append(train_REL2Solu_2.item())

        if i_epoch % 1000 == 0:
            run_times = time.time() - t0
            tmp_lr = optimizer.param_groups[0]['lr']
            DNN_Log_Print.print_and_log_train_one_epoch2NS(
                i_epoch, run_times, tmp_lr, temp_penalty_bd, PWB.item(), loss_it.item(),
                loss_bd.item(), loss.item(), train_MSE2P.item(),train_REL2P.item(), train_MSE2Solu_1.item(),
                train_REL2Solu_1.item(), train_MSE2Solu_2.item(), train_REL2Solu_2.item(),log_out=log_fileout)

            # ---------------------------   test network ----------------------------------------------
            UNN2test, UNN2testx, UNN2testy, KNN2test = NS_dnn.evalue_DNN2NS(XY_points=xy_it2test_batch)
            test_epoch.append(i_epoch / 1000)

            point_square_error2Para = torch.square(Ptrue2test - KNN2test)
            mse2p_test = torch.mean(point_square_error2Para)
            test_mse2P_all.append(mse2p_test)
            rel2p_test = mse2p_test / torch.mean(torch.square(Ptrue2test))
            test_rel2P_all.append(rel2p_test)

            point_square_error2Solu = torch.square(U1true2test - UNN2testx)
            mse2solu_test = torch.mean(point_square_error2Solu)
            test_mse2Solu_all.append(mse2solu_test)
            rel2solu_test = mse2solu_test / torch.mean(torch.square(U1true2test))
            test_rel2Solu_all.append(rel2solu_test)

            point_square_error2Solu_2 = torch.square(U2true2test - UNN2testy)
            mse2solu_test_2 = torch.mean(point_square_error2Solu_2)
            test_mse2Solu_all_2.append(mse2solu_test_2)
            rel2solu_test_2 = mse2solu_test_2 / torch.mean(torch.square(U2true2test))
            test_rel2Solu_all_2.append(rel2solu_test_2)



            DNN_Log_Print.print_and_log_test_one_epoch2NS(mse2p_test.item(), rel2p_test.item(),
                                                     mse2solu_test.item(), rel2solu_test.item(),
                                                     mse2solu_test_2.item(), rel2solu_test_2.item(),
                                                     log_out=log_fileout)

    # -----------------------  save training results to mat files, then plot them ---------------------------------
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=act_func,
                                         outPath=R['FolderName'])
    saveData.save_train_MSE_REL2mat(train_mse2P_all, train_rel2P_all, actName='P', outPath=R['FolderName'])

    saveData.save_train_MSE_REL2mat(train_mse2Solu_all_1, train_rel2Solu_all_1, actName='Solu_1', outPath=R['FolderName'])

    saveData.save_train_MSE_REL2mat(train_mse2Solu_all_2, train_rel2Solu_all_2, actName='Solu_2',outPath=R['FolderName'])

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

    plotData.plotTrain_MSE_REL_1act_func(train_mse2P_all, train_rel2P_all, actName='P', seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    plotData.plotTrain_MSE_REL_1act_func(train_mse2Solu_all_1, train_rel2Solu_all_1, actName='Solu_1', seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    plotData.plotTrain_MSE_REL_1act_func(train_mse2Solu_all_2, train_rel2Solu_all_2, actName='Solu_2', seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)
    # ----------------------  save testing results to mat files, then plot them --------------------------------
    saveData.save_2testSolus2mat(Ptrue2test, KNN2test, actName='P', actName1='ParaNN',
                                 outPath=R['FolderName'])

    saveData.save_2testSolus2mat(U1true2test, UNN2test[0], actName='Solu_1', actName1='SoluNN',
                                 outPath=R['FolderName'])

    saveData.save_2testSolus2mat(U2true2test, UNN2test[1], actName='Solu_1', actName1='SoluNN',
                                 outPath=R['FolderName'])
    saveData.save_testMSE_REL2mat(test_mse2P_all, test_rel2P_all, actName='P', outPath=R['FolderName'])
    saveData.save_testMSE_REL2mat(test_mse2Solu_all, test_rel2Solu_all, actName='Solu_1', outPath=R['FolderName'])
    saveData.save_testMSE_REL2mat(test_mse2Solu_all_2, test_rel2Solu_all_2, actName='Solu_2', outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse2P_all, test_rel2P_all, test_epoch, actName='P',
                              seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2Solu_all, test_rel2Solu_all, test_epoch, actName='Solu_1',
                              seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2Solu_all_2, test_rel2Solu_all_2, test_epoch, actName='Solu_2',
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

    store_file = 'Navier_Stoke'
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

    R['PDE_type'] = 'Navier_Stoke'
    R['equa_name'] = 'Navier_Stoke'
    # R['equa_name'] = 'Linear_diffusion2'

    R['epsilon'] = 0.1
    R['order2pLaplace_operator'] = 2

    R['input_dim'] = 2  # 输入维数，即问题的维数(几元问题)
    R['out_dim'] = 2  # 输出维数

    R['inputdim2P'] = 2
    R['outdim2P'] = 1

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
    R['learning_rate'] = 0.01                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['train_model'] = 'union_training'
    # R['train_model'] = 'group2_training'
    # R['train_model'] = 'group3_training'
    # R['train_model'] = 'group4_training'

    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    #R['regular_wb_model'] = 'L2'
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
    R['freq'] = np.arange(1, 51)
    #R['freq'] = np.arange(1, 101)
    # R['freq'] = np.random.normal(1, 120, 100)

    # R['freq2P'] = np.arange(1, 31)
    R['freq2P'] = np.arange(1, 51)
    #R['freq2P'] = np.arange(1, 101)
    # R['freq2P'] = np.random.normal(1, 120, 100)

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
    R['Model2P'] = 'Fourier_DNN'

    if R['Model2P'] == 'Fourier_DNN':
        R['hiddens2P'] = (125, 100, 50, 50)
    else:
        R['hiddens2P'] = (250, 100, 50, 50)

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

    R['nameIn2P'] = 'tanh'
    # R['nameIn2Para'] = 'gcu'
    # R['nameIn2Para'] = 'gelu'
    # R['nameIn2Para'] = 'mgelu'
    # R['nameIn2Para'] = 'mish'
    # R['nameIn2Para'] = 'gauss'
    # R['nameIn2Para'] = 'sin'

    R['nameHidden2P'] = 'tanh'
    # R['nameHidden2Para'] = 'gcu'
    # R['nameHidden2Para'] = 'gelu'
    # R['nameHidden2Para'] = 'mgelu'
    # R['nameHidden2Para'] = 'mish'
    # R['nameHidden2Para'] = 'gauss'
    # R['nameHidden2Para'] = 's2relu'
    # R['nameHidden2Para'] = 'sin'
    # R['nameHidden2Para'] = 'sinADDcos'

    R['nameOut2P'] = 'linear'

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

    R['sfourier2p'] = 1.0
    if R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'tanh':
        R['sfourier2p'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 's2relu':
        # R['sfourier'] = 0.5
        R['sfourier2p'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'sinADDcos':
        R['sfourier2p'] = 0.5
        # R['sfourier'] = 1.0
    else:
        R['sfourier2p'] = 1.0

    R['repeat_high_freq2solu'] = True
    R['repeat_high_freq2p'] = False

    R['use_gpu'] = True
    solve_PDE(R)

    # Ritz loss 不收敛
    # lncosh loss 不收敛
    # Fourier(1.0) + tanh 才有效果
    # 对于多尺度震荡问题，尺度因子要选的大一些
