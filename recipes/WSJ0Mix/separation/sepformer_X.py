import torch
from torch import nn



#there should exist a block for information fusion
class GAU(nn.Module):
    def __init__(self):
        super(GAU,self).__init__()
        self.Sig = nn.Sigmoid()
        self.Tan = nn.Tanh()

    def forward(self,x):
        x1 = x[:,0,:,:]
        x2 = x[:,1,:,:]
        x1 = self.Sig(x1)
        x2 = self.Tan(x2)
        return x1*x2
class GroupBlock(nn.Module):
    def __init__(self):
        super(GroupBlock,self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=(1,1),
            stride=(1,1)
            ),
            GAU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=(1,1),
            stride = (1,1)
            ),
            GAU()
        )
        self.Gate = nn.PReLU()
        pass

    def forward(self,masks,mixture):
        #mixture = mixture/torch.max(mixture)  #?? this need to be changed this is bad
        init_version = mixture*masks
        print(masks.shape,init_version.shape)
        mask1 = torch.stack([masks[0],init_version[0]],dim = 1)
        mask2 = torch.stack([masks[1],init_version[1]],dim = 1)

        mask_res1 = self.Gate(mixture[0] - init_version[1])
        mask_res2 = self.Gate(mixture[0] - init_version[0])

        mask1 = self.Conv1(mask1) + mask_res1
        mask2 = self.Conv2(mask2) + mask_res2
        return torch.cat([mask1,mask2],dim=0)
        pass
class InteractiveModule(nn.Module):
    def __init__(self, input_channels, bias=False):
        super(InteractiveModule, self).__init__()
        self.input_channels = input_channels
        self.conv2D = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(input_channels),
            nn.Dropout(0.3),
            nn.Sigmoid()
        )

        # self.S2_conv2D = nn.Sequential(
        #     nn.Conv2d(input_channels * 2, input_channels, kernel_size=1, stride=1, bias=bias),
        #     nn.BatchNorm2d(input_channels),
        #     nn.Dropout(0.3),
        #     nn.Sigmoid()
        # )

    def forward(self, speech):
        mask = self.conv2D(speech)
        speech = speech + mask * speech
        return speech


class SpectrumEncoder(nn.Module):
    """
       Encoder
       input: a tensor[M,2,T,F],which was processed by STFT
               T is the number of frames ,F is the frequency
       output: a tensor[M,C,T,F/4],C is output channels

       """

    def __init__(self, encoder_size, encoder_stride, encoder_channel, encoder_bias=False):
        super(SpectrumEncoder, self).__init__()
        self.encoder_size, self.encoder_stride, self.encoder_channel = encoder_size, encoder_stride, encoder_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, encoder_channel[0], kernel_size=encoder_size, stride=encoder_stride[0], padding=[1, 2],
                      bias=encoder_bias),
            nn.BatchNorm2d(encoder_channel[0]),
            nn.Dropout(0.3),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(encoder_channel[0], encoder_channel[1], kernel_size=encoder_size, stride=encoder_stride[1],
                      padding=[1, 2], bias=encoder_bias),
            nn.BatchNorm2d(encoder_channel[1]),
            nn.Dropout(0.3),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(encoder_channel[1], encoder_channel[2], kernel_size=encoder_size, stride=encoder_stride[2],
                      padding=[1, 2], bias=encoder_bias),
            nn.BatchNorm2d(encoder_channel[2]),
            nn.Dropout(0.3),
            nn.PReLU()
        )

    def forward(self, mixture):
        """
        Args:
            mixture: [M,2,T,F],M is batch size
        Returns:
            output: [M, C,T,F/4]
        """
        padding = torch.zeros((mixture.size(0),mixture.size(1),mixture.size(2),1)).cuda()
        mixture = torch.cat([mixture,padding],dim=3)
        result1 = mixture
        result2 = self.conv1(result1)
        result3 = self.conv2(result2)
        output  = self.conv3(result3)
        return output, result1, result2, result3
    pass


class GatedBlock(nn.Module):
    """
        Dchannel:反卷积输出的channel  [32,16,2]
        Coutput：RA那边传过来的特征channel [64,64,64]
        C_feature:encoder那边传过来的通道数 [32,16,2]

    """

    def __init__(self, Dchannel, C_Output, C_Feature, Dkernel_size=(3, 4), Dstride=(1, 2) \
                 , kernel_size=(1, 1), stride=(1, 1), padding=[1, 1],bias=False):
        super(GatedBlock, self).__init__()
        # Deconv config
        self.DC = Dchannel
        self.CO = C_Output  # output channel
        self.CF = C_Feature  # feature channel FORM encoder

        self.Deconv = nn.ConvTranspose2d(in_channels=self.CO,
                                         out_channels=self.DC,
                                         kernel_size=Dkernel_size,
                                         stride=Dstride,
                                         padding=padding,bias=bias
                                         )
        self.Conv1 = nn.Conv2d(in_channels=self.DC + self.CF,
                               out_channels=self.DC,
                               kernel_size=kernel_size,
                               stride=stride,bias=bias)

        self.BN1 = nn.BatchNorm2d(num_features=self.DC)

        self.activation1 = nn.PReLU()

        self.Conv2 = nn.Conv2d(in_channels=self.DC + self.DC,
                               out_channels=self.DC,
                               kernel_size=kernel_size,
                               stride=stride,bias=bias)

        self.BN2 = nn.BatchNorm2d(num_features=self.DC)

        self.activation2 = nn.PReLU()

    def forward(self, In_module, encode_fea):
        Highpass_long = self.Deconv(In_module)
        # print(Highpass_long.shape)
        # print(encode_fea.shape)
        Straight = torch.cat((Highpass_long, encode_fea), dim=1)


        Straight = self.Conv1(Straight)
        Straight = self.BN1(Straight)
        Straight = self.activation1(Straight)

        Straight = Straight * encode_fea

        Straight = torch.cat((Straight, Highpass_long), 1)

        Straight = self.Conv2(Straight)
        Straight = self.BN2(Straight)
        Straight = self.activation2(Straight)

        Straight = Straight + Highpass_long
        return Straight


class SpectrumDecoder(nn.Module):
    def __init__(self, channel_numbers=[32, 16, 2], Dstrides=[(1, 2), (1, 2), (1, 2)], \
                 Dkernel_sizes=[(3, 4), (3, 4), (3, 4)], Dpaddings=[[1, 1], [1, 1], [1, 1]],
                 kernel_sizes=[(1, 1), (1, 1), (1, 1)], \
                 strides=[(1, 1), (1, 1), (1, 1)], feature_channels=[32, 16, 2], Inter_Module_Out=64, \
                 Okernel_size=(1, 1), Ostride=(1, 1), bias=False):
        super(SpectrumDecoder, self).__init__()
        self.C = channel_numbers
        self.Dstrides = Dstrides
        self.Dkernel_sizes = Dkernel_sizes
        self.Dpaddings = Dpaddings
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.Gateblocks = nn.ModuleDict()
        self.feature_channels = feature_channels
        for i in range(len(self.C)):
            if i == 0:
                self.Gateblocks['Gateblock_' + str(i)] = GatedBlock(self.C[i], C_Output=Inter_Module_Out, \
                                                                    C_Feature=self.feature_channels[i], \
                                                                    Dkernel_size=self.Dkernel_sizes[i], \
                                                                    Dstride=self.Dstrides[i],
                                                                    kernel_size=self.kernel_sizes[i],
                                                                    stride=self.strides[i],
                                                                    padding=self.Dpaddings[i])
            else:
                self.Gateblocks['Gateblock_' + str(i)] = GatedBlock(self.C[i], C_Output=self.C[i - 1], \
                                                                    C_Feature=self.feature_channels[i], \
                                                                    Dkernel_size=self.Dkernel_sizes[i], \
                                                                    Dstride=self.Dstrides[i],
                                                                    kernel_size=self.kernel_sizes[i],
                                                                    stride=self.strides[i],
                                                                    padding=self.Dpaddings[i])

        self.Conv2D = nn.Conv2d(in_channels=self.C[len(self.C) - 1], \
                                out_channels=self.C[len(self.C) - 1], \
                                kernel_size=Okernel_size, \
                                stride=Ostride, bias=bias)


    def forward(self, feature1, feature2, feature3, Inter_Module_out):
        output = self.Gateblocks['Gateblock_0'](Inter_Module_out, feature3)

        output = self.Gateblocks['Gateblock_1'](output, feature2)

        output = self.Gateblocks['Gateblock_2'](output, feature1)

        output = self.Conv2D(output)

        return output

class ResBlock(nn.Module):
    def __init__(self,encoder_size=[3, 5],encoder_stride=[[1, 2], [1, 2], [1, 2]],\
                 encoder_channel=[16, 32, 64],in_channels=64,number_interactive = 1):
        super(ResBlock,self).__init__()
        self.Encoder = SpectrumEncoder(encoder_size,encoder_stride,encoder_channel)
        self.Decoder = SpectrumDecoder()

        self.Residual_paths = nn.ModuleDict()
        self.number_interactive = number_interactive
        for i in range(number_interactive):
            self.Residual_paths['inter_Module_'+str(i)] = InteractiveModule(in_channels)
        pass
    def forward(self,x):

        input,result1,result2,result3 = self.Encoder(x)
        for i in range(self.number_interactive):
            input = self.Residual_paths['inter_Module_'+str(i)](input)

        output=self.Decoder(result1,result2,result3,input)

        return output
        pass