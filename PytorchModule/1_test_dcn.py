
import datetime
import json
from collections import deque
import paddle.fluid as fluid
import paddle.fluid.layers as P
import sys
import time
import shutil
import math
import copy
import random
import threading
import numpy as np
import os
from paddle.fluid.param_attr import ParamAttr

from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import Constant
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.optimizer import ExponentialMovingAverage
import paddle

import torch
from model.custom_layers import Conv2dUnit, DCNv2

from collections import OrderedDict


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0, bias=True)
        self.act1 = torch.nn.LeakyReLU(0.1)

        self.dcnv2 = DCNv2(8, 512, filter_size=3, stride=2, padding=1, bias_attr=False)

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.act1(x)

        x = self.dcnv2(x)
        return x




if __name__ == '__main__':
    paddle.enable_static()
    use_gpu = False

    lr = 0.1

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs = P.data(name='input_1', shape=[-1, 3, 32, 32], append_batch_size=False, dtype='float32')
            conv01_out_tensor = fluid.layers.conv2d(input=inputs, num_filters=8, filter_size=1, stride=1, padding=0,
                                                    param_attr=ParamAttr(name="conv01_weights"),
                                                    bias_attr=ParamAttr(name="conv01_bias", initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)))
            act01_out_tensor = fluid.layers.leaky_relu(conv01_out_tensor, alpha=0.1)


            filter_size = 3
            filters = 512
            stride = 2
            padding = 1
            conv_name = 'dcnv2'
            offset_mask = fluid.layers.conv2d(
                input=act01_out_tensor,
                num_filters=filter_size * filter_size * 3,
                filter_size=filter_size,
                stride=stride,
                padding=padding,
                act=None,
                # param_attr=ParamAttr(initializer=Constant(0.0), name=conv_name + "_conv_offset.w_0"),
                param_attr=ParamAttr(name=conv_name + "_conv_offset.w_0"),
                bias_attr=ParamAttr(initializer=Constant(0.0), name=conv_name + "_conv_offset.b_0"),
                name=conv_name + "_conv_offset")
            offset = offset_mask[:, :filter_size**2 * 2, :, :]
            mask = offset_mask[:, filter_size**2 * 2:, :, :]
            mask = fluid.layers.sigmoid(mask)
            conv02_out_tensor = fluid.layers.deformable_conv(input=act01_out_tensor, offset=offset, mask=mask,
                                             num_filters=filters,
                                             filter_size=filter_size,
                                             stride=stride,
                                             padding=padding,
                                             groups=1,
                                             deformable_groups=1,
                                             im2col_step=1,
                                             param_attr=ParamAttr(name=conv_name + "_weights"),
                                             bias_attr=False,
                                             name=conv_name + ".conv2d.output.1")


            # ??????????????????
            y_true = P.data(name='y_true', shape=[-1, 1, 16, 16], append_batch_size=False, dtype='float32')
            # ????????????????????????????????????P.pow()??????op???????????????python???????????????**???
            mseloss = P.pow(y_true - conv02_out_tensor, 2)
            mseloss = P.reduce_mean(mseloss)       # ??????????????????mse????????????

            # ?????????
            optimizer = fluid.optimizer.SGD(learning_rate=lr)
            optimizer.minimize(mseloss)


    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            # ?????????????????????????????????????????????????????????????????????
            inputs = P.data(name='input_1', shape=[-1, 3, 32, 32], append_batch_size=False, dtype='float32')
            conv01_out_tensor = fluid.layers.conv2d(input=inputs, num_filters=8, filter_size=1, stride=1, padding=0,
                                                    param_attr=ParamAttr(name="conv01_weights"),
                                                    bias_attr=ParamAttr(name="conv01_bias", initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)))
            act01_out_tensor = fluid.layers.leaky_relu(conv01_out_tensor, alpha=0.1)


            filter_size = 3
            filters = 512
            stride = 2
            padding = 1
            conv_name = 'dcnv2'
            offset_mask = fluid.layers.conv2d(
                input=act01_out_tensor,
                num_filters=filter_size * filter_size * 3,
                filter_size=filter_size,
                stride=stride,
                padding=padding,
                act=None,
                # param_attr=ParamAttr(initializer=Constant(0.0), name=conv_name + "_conv_offset.w_0"),
                param_attr=ParamAttr(name=conv_name + "_conv_offset.w_0"),
                bias_attr=ParamAttr(initializer=Constant(0.0), name=conv_name + "_conv_offset.b_0"),
                name=conv_name + "_conv_offset")
            offset = offset_mask[:, :filter_size**2 * 2, :, :]
            mask = offset_mask[:, filter_size**2 * 2:, :, :]
            mask = fluid.layers.sigmoid(mask)
            conv02_out_tensor = fluid.layers.deformable_conv(input=act01_out_tensor, offset=offset, mask=mask,
                                             num_filters=filters,
                                             filter_size=filter_size,
                                             stride=stride,
                                             padding=padding,
                                             groups=1,
                                             deformable_groups=1,
                                             im2col_step=1,
                                             param_attr=ParamAttr(name=conv_name + "_weights"),
                                             bias_attr=False,
                                             name=conv_name + ".conv2d.output.1")
            eval_fetch_list = [conv02_out_tensor]
    eval_prog = eval_prog.clone(for_test=True)
    # ???????????????
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    compiled_eval_prog = fluid.compiler.CompiledProgram(eval_prog)


    # pytorch?????????????????????????????????????????????paddle????????????????????????????????????paddle???????????????
    # 1.?????????
    paddle_conv01_weights = np.array(fluid.global_scope().find_var('conv01_weights').get_tensor())
    paddle_conv01_bias = np.array(fluid.global_scope().find_var('conv01_bias').get_tensor())
    # 2.?????????
    # 3.?????????
    paddle_conv02_weights = np.array(fluid.global_scope().find_var('dcnv2_conv_offset.w_0').get_tensor())
    paddle_conv02_bias = np.array(fluid.global_scope().find_var('dcnv2_conv_offset.b_0').get_tensor())
    paddle_conv02_dcn_weights = np.array(fluid.global_scope().find_var('dcnv2_weights').get_tensor())

    # 6.?????????
    # 7.?????????????????????????????????

    #  pytorch?????????????????????
    myNet = MyNet()
    # myNet = myNet.cuda()
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    # loss_fn = loss_fn.cuda()
    optimizer2 = torch.optim.SGD(filter(lambda p: p.requires_grad, myNet.parameters()), lr=lr)   # requires_grad==True ???????????????????????????
    # ??????????????????????????????
    myNet.conv1.weight.data = torch.Tensor(np.copy(paddle_conv01_weights))
    myNet.conv1.bias.data = torch.Tensor(np.copy(paddle_conv01_bias))
    myNet.dcnv2.conv_offset.weight.data = torch.Tensor(np.copy(paddle_conv02_weights))
    myNet.dcnv2.conv_offset.bias.data = torch.Tensor(np.copy(paddle_conv02_bias))
    myNet.dcnv2.dcn_weight.data = torch.Tensor(np.copy(paddle_conv02_dcn_weights))


    myNet.train()  # ?????????????????????


    # ?????????8???
    for step in range(8):
        print('------------------ step %d ------------------' % step)
        # ==================== train ====================
        batch_data = np.random.normal(loc=0, scale=1, size=(2, 3, 32, 32)).astype(np.float32)
        y_true_arr = np.random.normal(loc=0, scale=1, size=(2, 512, 16, 16)).astype(np.float32)

        paddle_mseloss_out, paddle_conv01_out, paddle_conv02_out = exe.run(train_prog, feed={"input_1": batch_data, "y_true": y_true_arr, },
                                                                       fetch_list=[mseloss, conv01_out_tensor, conv02_out_tensor])

        print('train_forward:')


        # python??????????????????????????????paddle????????????????????????????????????????????????????????????
        batch_data = torch.Tensor(batch_data)
        y_true_arr = torch.Tensor(y_true_arr)
        my_act02_out = myNet(batch_data)
        my_mseloss_out = loss_fn(my_act02_out,  y_true_arr)

        # ????????????
        optimizer2.zero_grad()  # ???????????????????????????????????????
        my_mseloss_out.backward()  # ??????????????????, ?????????????????????
        optimizer2.step()  # ??????????????????????????? net ??? parameters ???


        _my_mseloss_out = my_mseloss_out.cpu().data.numpy()

        diff_mseloss_out = np.sum((paddle_mseloss_out - _my_mseloss_out)**2)
        print('diff_mseloss_out=%.6f' % diff_mseloss_out)   # ??????0???????????????????????????PaddlePaddle bn??????????????????


        print('\nbackward:')
        # ????????????????????????????????????
        paddle_conv01_weights = np.array(fluid.global_scope().find_var('conv01_weights').get_tensor())
        paddle_conv01_bias = np.array(fluid.global_scope().find_var('conv01_bias').get_tensor())

        paddle_conv02_weights = np.array(fluid.global_scope().find_var('dcnv2_conv_offset.w_0').get_tensor())
        paddle_conv02_bias = np.array(fluid.global_scope().find_var('dcnv2_conv_offset.b_0').get_tensor())
        paddle_conv02_dcn_weights = np.array(fluid.global_scope().find_var('dcnv2_weights').get_tensor())


        diff_conv01_weights = np.sum((paddle_conv01_weights - myNet.conv1.weight.data.numpy())**2)
        print('diff_conv01_weights=%.6f' % diff_conv01_weights)   # ??????0???????????????????????????????????????
        diff_conv01_bias = np.sum((paddle_conv01_bias - myNet.conv1.bias.data.numpy())**2)
        print('diff_conv01_bias=%.6f' % diff_conv01_bias)   # ??????0???????????????????????????????????????


        print('\nDCNv2:')
        diff_conv02_weights = np.sum((paddle_conv02_weights - myNet.dcnv2.conv_offset.weight.data.numpy())**2)
        print('diff_conv02_weights=%.6f' % diff_conv02_weights)   # ??????0???????????????????????????????????????
        diff_conv02_bias = np.sum((paddle_conv02_bias - myNet.dcnv2.conv_offset.bias.data.numpy())**2)
        print('diff_conv02_bias=%.6f' % diff_conv02_bias)   # ??????0???????????????????????????????????????
        diff_conv02_dcn_weights = np.sum((paddle_conv02_dcn_weights - myNet.dcnv2.dcn_weight.data.numpy())**2)
        print('diff_conv02_dcn_weights=%.6f' % diff_conv02_dcn_weights)   # ??????0???????????????????????????????????????

        # ==================== test ====================
        test_data = np.random.normal(loc=0, scale=1, size=(2, 3, 32, 32)).astype(np.float32)
        _conv02_out_tensor, = exe.run(compiled_eval_prog, feed={"input_1": test_data, }, fetch_list=eval_fetch_list)
        # ???????????????test
        print('\ntest_forward:')
        myNet.eval()  # ?????????????????????
        test_data = torch.Tensor(test_data)
        my_test_conv02_out_ = myNet(test_data)
        my_test_conv02_out = my_test_conv02_out_.cpu().data.numpy()
        myNet.train()  # ?????????????????????
        d1 = np.sum((_conv02_out_tensor - my_test_conv02_out)**2)
        print('d1=%.6f' % d1)   # ??????0?????????????????????????????????



