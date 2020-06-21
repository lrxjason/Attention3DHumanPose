import torch.nn as nn
import torch
import torch.nn.functional as F


class DWConv(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=3):
        super(DWConv, self).__init__()
        self.DW_conv = nn.Conv1d(in_features, in_features, kernel_size=kernel_size, stride=stride,
                                 groups=in_features, bias=False)
        self.DW_bn = nn.BatchNorm1d(in_features, momentum=0.1)
        self.PW_conv = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.PW_bn = nn.BatchNorm1d(out_features, momentum=0.1)

    def forward(self, x):
        x = self.DW_conv(x)
        x = self.DW_bn(x)
        x = self.PW_conv(x)
        x = self.PW_bn(x)
        return x


class Kernel_Attention(nn.Module):
    def __init__(self, in_features, out_features=1024, M=3, G=8, r=128, stride=3):
        super(Kernel_Attention, self).__init__()
        self.convs = nn.ModuleList([])

        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_features, in_features, kernel_size=3, dilation=i + 1, stride=stride, padding=0,
                          groups=in_features, bias=False),
                nn.BatchNorm1d(in_features),
                nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, padding=0, groups=G, bias=False),
                nn.BatchNorm1d(out_features),
                Mish()
            ))
        self.fc = nn.Linear(out_features, r)

        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(r, out_features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            if i == 0:
                fea = conv(x).unsqueeze_(dim=1)
                feas = fea
            else:
                fea = F.pad(x, (i, i), 'replicate')
                fea = conv(fea).unsqueeze_(dim=1)
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = Mish()
        self.sigmoid = nn.Sigmoid()

        self.pad = [filter_widths[0] // 2]
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

    def set_bn_momentum(self, momentum):
        for bn in self.layers_bn:
            bn.momentum = momentum
        for bn in self.layers_tem_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        sz = x.shape[:3]
        mean = x[:, :, 0:1, :].expand_as(x)
        input_pose_centered = x - mean

        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        input_pose_centered = input_pose_centered.view(input_pose_centered.shape[0], input_pose_centered.shape[1], -1)
        input_pose_centered = input_pose_centered.permute(0, 2, 1)

        x = self._forward_blocks(x, input_pose_centered)

        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)

        return x


class TemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.

    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.2, channels=1024, dense=False):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        expand_conv = []
        for i in range(len(filter_widths) - 1):
            expand_conv.append(DWConv(num_joints_in * in_features, channels,
                                      kernel_size=filter_widths[0], stride=filter_widths[0]))
        self.expand_conv = nn.ModuleList(expand_conv)

        self.cos_dis = nn.CosineSimilarity(dim=1, eps=1e-6)
        layers_tem_att = []
        layers_tem_bn = []
        self.frames = self.total_frame()

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]

        dilation_conv = []
        dilation_bn = []

        for i in range(3):
            dilation_conv.append(DWConv(channels, channels, kernel_size=filter_widths[i], stride=filter_widths[i]))
            dilation_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            dilation_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

        self.dilation_conv = nn.ModuleList(dilation_conv)
        self.dilation_bn = nn.ModuleList(dilation_bn)

        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2) if causal else 0)

            layers_tem_att.append(nn.Linear(self.frames, self.frames // next_dilation))
            layers_tem_bn.append(nn.BatchNorm1d(self.frames // next_dilation))

            layers_conv.append(Kernel_Attention(channels, out_features=channels))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self.layers_tem_att = nn.ModuleList(layers_tem_att)
        self.layers_tem_bn = nn.ModuleList(layers_tem_bn)

    def set_KA_bn(self, momentum):
        for i in range(len(self.layers_conv) // 2):
            for j in range(3):
                self.layers_conv[2 * i].convs[j][1].momentum = momentum
                self.layers_conv[2 * i].convs[j][3].momentum = momentum

    def set_expand_bn(self, momentum):
        for i in range(len(self.expand_conv)):
            self.expand_conv[i].DW_bn.momentum = momentum
            self.expand_conv[i].PW_bn.momentum = momentum

    def set_dilation_bn(self, momentum):
        for bn in self.dilation_bn:
            bn.momentum = momentum
        for i in range(len(self.dilation_conv)//2):
            self.dilation_conv[2*i].DW_bn.momentum = momentum
            self.dilation_conv[2*i].PW_bn.momentum = momentum

    def total_frame(self):
        frames = 1
        for i in range(len(self.filter_widths)):
            frames *= self.filter_widths[i]
        return frames

    def _forward_blocks(self, x, input_2D_centered):
        b, c, t = input_2D_centered.size()
        x_target = input_2D_centered[:, :, input_2D_centered.shape[2] // 2]
        x_target_extend = x_target.view(b, c, 1)
        x_traget_matrix = x_target_extend.expand_as(input_2D_centered)
        cos_score = self.cos_dis(x_traget_matrix, input_2D_centered)

        '''
        Top layers
        '''
        x_0_1 = x[:, :, 1::3]
        x_0_2 = x[:, :, 4::9]
        x_0_3 = x[:, :, 13::27]

        x = self.drop(self.relu(self.expand_conv[0](x)))
        x_0_1 = self.drop(self.relu(self.expand_conv[1](x_0_1)))
        x_0_2 = self.drop(self.relu(self.expand_conv[2](x_0_2)))
        x_0_3 = self.drop(self.relu(self.expand_conv[3](x_0_3)))

        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i + 1] + self.filter_widths[i + 1] // 2:: self.filter_widths[i + 1]]
            t_attention = self.sigmoid(self.layers_tem_bn[i](self.layers_tem_att[i](cos_score)))  # [batches frames]
            t_attention_expand = t_attention.unsqueeze(1)  # [batches channels frames]
            if i == 0:
                res_1_1 = res[:, :, 1::3]
                res_1_2 = res[:, :, 4::9]
                x = x * t_attention_expand  # broadcasting dot mul
                x_1_1 = x[:, :, 1::3]
                x_1_2 = x[:, :, 4::9]

                x = self.drop(self.layers_conv[2 * i](x))
                x = res + self.drop(self.relu(self.layers_bn[i](self.layers_conv[2 * i + 1](x))))

                x_1_1 = self.drop(self.relu(self.dilation_conv[0](x_1_1)))
                x_1_1 = res_1_1 + self.drop(self.relu(self.dilation_bn[0](self.dilation_conv[1](x_1_1))))

                x_1_2 = self.drop(self.relu(self.dilation_conv[2](x_1_2)))
                x_1_2 = res_1_2 + self.drop(self.relu(self.dilation_bn[1](self.dilation_conv[3](x_1_2))))

            elif i == 1:
                res_2_1 = res[:, :, 1::3]
                x = x * t_attention_expand  # broadcasting dot mul
                x_2_1 = x[:, :, 1::3]
                x_0_1 = x_0_1 * t_attention_expand  # broadcasting dot mul
                x = x + x_0_1

                x = self.drop(self.layers_conv[2 * i](x))
                x = res + self.drop(self.relu(self.layers_bn[i](self.layers_conv[2 * i + 1](x))))

                x_2_1 = self.drop(self.relu(self.dilation_conv[4](x_2_1)))
                x_2_1 = res_2_1 + self.drop(self.relu(self.dilation_bn[2](self.dilation_conv[5](x_2_1))))

            elif i == 2:
                x = x + x_0_2 + x_1_1
                x = x * t_attention_expand  # broadcasting dot mul
                x = self.drop(self.layers_conv[2 * i](x))
                x = res + self.drop(self.relu(self.layers_bn[i](self.layers_conv[2 * i + 1](x))))
            elif i == 3:
                x = x + x_0_3 + x_1_2 + x_2_1
                x = x * t_attention_expand  # broadcasting dot mul
                x = self.drop(self.layers_conv[2 * i](x))
                x = res + self.drop(self.relu(self.layers_bn[i](self.layers_conv[2 * i + 1](x))))

        x = self.shrink(x)
        return x

