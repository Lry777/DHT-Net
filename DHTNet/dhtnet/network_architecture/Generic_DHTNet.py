#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy
import os
from dhtnet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from dhtnet.network_architecture.initialization import InitWeights_He
from dhtnet.network_architecture.neural_network import SegmentationNetwork
# from dhtnet.network_architecture.transformer_block import *
from dhtnet.network_architecture.Dynamic_Hierarchical_TransformerBlock import DHTransformer
from dhtnet.network_architecture.all_attention import *
from batchgenerators.utilities.file_and_folder_operations import *
from dhtnet.paths import network_training_output_dir_base
import torch.nn.functional
import copy as cp


class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        x = self.blocks(x)
        return x


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class Generic_OursV2(SegmentationNetwork):
    def __init__(self,img_size, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv3d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False,
                 embed_dims = [64, 128, 256, 320]):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_OursV2, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True, 'kernel_size': [3, 3, 3], 'padding': [1, 1, 1]}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        ##############################################ours##############################################################
        ############################# 编码部分 #######################################
        self.downblock_channal = [32, 64, 128, 256, 512, 512]
        self.start0 = nn.Sequential(
            StackedConvLayers(input_channels, self.downblock_channal[0], num_conv_per_stage,
                              self.conv_op, self.conv_kwargs, self.norm_op,
                              self.norm_op_kwargs, self.dropout_op,
                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                              first_stride=None, basic_block=basic_block))
        self.start1 = nn.Sequential(
            StackedConvLayers(self.downblock_channal[0], self.downblock_channal[1], num_conv_per_stage,
                              self.conv_op, self.conv_kwargs, self.norm_op,
                              self.norm_op_kwargs, self.dropout_op,
                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                              first_stride=[2,2,2], basic_block=basic_block))
        self.start2 = nn.Sequential(
            StackedConvLayers(self.downblock_channal[1], self.downblock_channal[2], num_conv_per_stage,
                              self.conv_op, self.conv_kwargs, self.norm_op,
                              self.norm_op_kwargs, self.dropout_op,
                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                              first_stride=[2, 2, 2], basic_block=basic_block))

        self.encoder_trans1 = DHTransformer(in_chans=self.downblock_channal[2], out_chans=self.downblock_channal[3], is_pool=True, sr_ratios=4)
        self.encoder_trans2 = DHTransformer(in_chans=self.downblock_channal[3], out_chans=self.downblock_channal[4], is_pool=True, sr_ratios=2)

        # middle attention
        self.mattn = Spartial_Attention3d(kernel_size=3)
        self.mdcat1 = nn.Sequential(
            StackedConvLayers(self.downblock_channal[0], self.downblock_channal[0], 1,
                              self.conv_op, self.conv_kwargs, self.norm_op,
                              self.norm_op_kwargs, self.dropout_op,
                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                              first_stride=[2,2,2], basic_block=basic_block))
        self.mdcat2 = nn.Sequential(
            StackedConvLayers(self.downblock_channal[0]+self.downblock_channal[1],
                              self.downblock_channal[0]+self.downblock_channal[1], 1,
                              self.conv_op, self.conv_kwargs, self.norm_op,
                              self.norm_op_kwargs, self.dropout_op,
                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                              first_stride=[2, 2, 2], basic_block=basic_block))
        self.mupcat3 = nn.Sequential(
            StackedConvLayers(self.downblock_channal[0] + self.downblock_channal[1] + self.downblock_channal[2],
                            self.downblock_channal[2], 1,
                              self.conv_op, self.conv_kwargs, self.norm_op,
                              self.norm_op_kwargs, self.dropout_op,
                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                              first_stride=None, basic_block=basic_block))
        self.gate3 = Gate(self.downblock_channal[2], self.downblock_channal[2])
        self.mupcat2 = transpconv(self.downblock_channal[0] + self.downblock_channal[1] + self.downblock_channal[2],
                                  self.downblock_channal[1], kernel_size=2, stride=2, bias=False)
        self.gate2 = Gate(in_channels=self.downblock_channal[1],out_channels=self.downblock_channal[1])
        self.mupcat1 = transpconv(self.downblock_channal[0] + self.downblock_channal[1] + self.downblock_channal[2],
                                  self.downblock_channal[0], kernel_size=4, stride=4, bias=False)
        self.gate1 = Gate(in_channels=self.downblock_channal[0],out_channels=self.downblock_channal[0])

        ############################### 解码部分 ########################################
        # 第1步
        self.up2 = transpconv(self.downblock_channal[4], self.downblock_channal[3], kernel_size=2, stride=2, bias=False)
        self.decoder_trans2 = nn.Sequential(
            StackedConvLayers(self.downblock_channal[3] * 2, self.downblock_channal[3], num_conv_per_stage - 1,
                              self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
            StackedConvLayers(self.downblock_channal[3], self.downblock_channal[3], 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                              self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
        )
        self.seg_out2 = nn.Conv3d(self.downblock_channal[3], num_classes, 1, 1, 0, 1, 1, seg_output_use_bias)
        # 第2步
        self.up1 = transpconv(self.downblock_channal[3], self.downblock_channal[2], kernel_size=2, stride=2, bias=False)
        self.decoder_trans1 = nn.Sequential(
            StackedConvLayers(self.downblock_channal[2] * 2, self.downblock_channal[2], num_conv_per_stage - 1,
                              self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
            StackedConvLayers(self.downblock_channal[2], self.downblock_channal[2], 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                              self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
        )
        self.seg_out1 = nn.Conv3d(self.downblock_channal[2], num_classes, 1, 1, 0, 1, 1, seg_output_use_bias)
        # 第3步
        self.end_up2 = transpconv(self.downblock_channal[2], self.downblock_channal[1], kernel_size=2, stride=2, bias=False)
        self.end2 = nn.Sequential(
            StackedConvLayers(self.downblock_channal[1] * 2, self.downblock_channal[1], num_conv_per_stage - 1,
                              self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
            StackedConvLayers(self.downblock_channal[1], self.downblock_channal[1], 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                              self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
        )
        self.end_seg_out2 = nn.Conv3d(self.downblock_channal[1], num_classes, 1, 1, 0, 1, 1, seg_output_use_bias)
        # 第4步
        self.end_up1 = transpconv(self.downblock_channal[1], self.downblock_channal[0], kernel_size=2, stride=2, bias=False)
        self.end1 = nn.Sequential(
            StackedConvLayers(self.downblock_channal[0] * 2, self.downblock_channal[0], num_conv_per_stage - 1,
                              self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
            StackedConvLayers(self.downblock_channal[0], self.downblock_channal[0], 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                              self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
        )
        self.end_seg_out1 = nn.Conv3d(self.downblock_channal[0], num_classes, 1, 1, 0, 1, 1, seg_output_use_bias)

        # 初始化权重
        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def forward(self, x):
        # print(x.shape)
        ###########trans-encoder###############
        encoder_outputs = []
        x = self.start0(x)  # Shape : [B, C, D, H, W] , C:1 -> 32
        encoder_outputs.append(x)
        x = self.start1(x)  # Shape : [B, C, D/2, H/2, W/2] , C:32 -> 64
        encoder_outputs.append(x)
        x = self.start2(x)  # Shape : [B, C, D/4, H/4, W/4] , C:64 -> 128
        encoder_outputs.append(x)
        x = self.encoder_trans1(x)  # Shape : [B, C, D/8, H/8, W/8] , C:128 -> 256
        encoder_outputs.append(x)
        x = self.encoder_trans2(x)  # Shape : [B, C, D/16, H/16, W/16] , C:256 -> 320
        encoder_outputs.append(x)
        #

        # # middle attention
        m1 = self.mattn(encoder_outputs[0])
        m2 = self.mattn(encoder_outputs[1])
        m3 = self.mattn(encoder_outputs[2])

        m1m2 = torch.cat([self.mdcat1(m1),m2],dim=1)  # Shape : [B, C=32+64, D/2, H/2, W/2]
        m_feature = torch.cat([self.mdcat2(m1m2),m3],dim=1)  # Shape : [B, C=32+64+128, D/4, H/4, W/4]

        encoder_outputs[0] = self.gate1(self.mupcat1(m_feature), encoder_outputs[0])
        encoder_outputs[1] = self.gate2(self.mupcat2(m_feature), encoder_outputs[1])
        encoder_outputs[2] = self.gate3(self.mupcat3(m_feature), encoder_outputs[2])

        # decoder
        seg_outputs = []

        xx = self.up2(x)  # Shape : [B, C, D/8, H/8, W/8]
        xx = torch.cat([encoder_outputs[-2], xx], dim=1)  # C:256 -> 256*2
        xx = self.decoder_trans2(xx)  # C:256*2 -> 256 -> 256
        seg_outputs.append(self.final_nonlin(self.seg_out2(xx)))

        xx = self.up1(xx)  # Shape : [B, C, D/4, H/4, W/4]
        # print(decoder_outputs[-3].shape, xx.shape)
        xx = torch.cat([encoder_outputs[-3], xx], dim=1)  # C:128 -> 128*2
        xx = self.decoder_trans1(xx)  # C:128*2 -> 128 -> 128
        seg_outputs.append(self.final_nonlin(self.seg_out1(xx)))

        xx = self.end_up2(xx)  # Shape : [B, C, D/2, H/2, W/2]
        # print(decoder_outputs[-4].shape, xx.shape)
        xx = torch.cat([encoder_outputs[-4], xx], dim=1)  # C:64 -> 64*2
        xx = self.end2(xx)  # C:64*2 -> 64 -> 64
        seg_outputs.append(self.final_nonlin(self.end_seg_out2(xx)))

        xx = self.end_up1(xx)  # Shape : [B, C, D, H, W]
        xx = torch.cat([encoder_outputs[-5], xx], dim=1)  # C:32 -> 32*2
        xx = self.end1(xx)  # C:32*2 -> 32 -> 32
        seg_outputs.append(self.final_nonlin(self.end_seg_out1(xx)))
        #print('finish',seg_outputs[0].shape)
        if self._deep_supervision and self.do_ds:
            output = tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
            # for out in output:
            #     print('final',out.shape)
            return output
        else:
            return seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp


######################################################################

def hidden_featuremap():
    feature = load_pickle(os.path.join(network_training_output_dir_base,'feature_map.pkl'))
    return feature
if __name__ == "__main__":
    a = hidden_featuremap()

    print(tuple(a))
