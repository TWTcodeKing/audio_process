import cmath
import speechbrain as sb
import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Conv1d) or isinstance(m, nn.ConvTranspose1d)or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.xavier_normal_(m.weight)


class Encoder(nn.Module):
    """
    Encoder
    input: a tensor[M,2,T,F],which was processed by STFT
            T is the number of frames ,F is the frequency
    output: a tensor[M,C,T,F/4],C is output channels

    """

    def __init__(self, encoder_size, encoder_stride, encoder_channel,encoder_bias=False):
        super(Encoder, self).__init__()
        self.encoder_size, self.encoder_stride, self.encoder_channel = encoder_size, encoder_stride, encoder_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, encoder_channel[0], kernel_size=encoder_size, stride=encoder_stride[0], padding=[1, 2],bias=encoder_bias),
            nn.BatchNorm2d(encoder_channel[0]),
            nn.Dropout(0.3),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(encoder_channel[0], encoder_channel[1], kernel_size=encoder_size, stride=encoder_stride[1],
                      padding=[1, 2],bias=encoder_bias),
            nn.BatchNorm2d(encoder_channel[1]),
            nn.Dropout(0.3),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(encoder_channel[1], encoder_channel[2], kernel_size=encoder_size, stride=encoder_stride[2],
                      padding=[1, 2],bias=encoder_bias),
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
        result1 = mixture
        #print(result1.shape)
        result2 = self.conv1(result1)
        #print(result2.shape)
        result3 = self.conv2(result2)
        #print(result3.shape)
        output = self.conv3(result3)
        #print(output.shape)
        return output, result1, result2, result3

class Residual(nn.Module):
    """
       Residual:a component in RA box
       input: a tensor[M,C,T,F'],which come from encoder
               T is the number of frames ,F is the frequency,C is encoder's outrput channel
       output: a tensor[M,C,T,F'], same as input

       """

    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1,bias=False):
        super(Residual,self).__init__() #change here
        # self.conv1 = nn.Conv2d(input_channels, num_channels,
        #                        kernel_size=[5, 7], padding=[2, 3], stride=strides)
        # self.conv2 = nn.Conv2d(num_channels, num_channels,
        #                        kernel_size=[5, 7], padding=[2, 3])
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides,bias=bias)
        else:
            self.conv3 = None
        # self.bn1 = nn.BatchNorm2d(num_channels)
        # self.bn2 = nn.BatchNorm2d(num_channels)

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, num_channels,kernel_size=[5, 7], padding=[2, 3], stride=strides,bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.Dropout(0.3),
            nn.PReLU(),
            nn.Conv2d(num_channels, num_channels,kernel_size=[5, 7], padding=[2, 3],bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.Dropout(0.3)

        )
        self.prelu = nn.PReLU()
    def forward(self, X):
        #print(self.net[0].weight.device)
        Y = self.net(X)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        Y=self.prelu(Y)
        # return F.relu(Y)
        return Y

class time_self_attention(nn.Module):
    """

    """

    def __init__(self, input_channels, strides=1,multi_heads=1):
        super(time_self_attention,self).__init__()
        self.input_channels = input_channels
        self.strides = strides
        self.multi_heads=multi_heads  #number of heads
        self.Q_filters=nn.ModuleDict()
        self.K_filters=nn.ModuleDict()
        self.V_filters=nn.ModuleDict()
        for i in range(multi_heads):
            self.Q_filters['Query_'+str(i)] = nn.Sequential(
                nn.Conv2d(input_channels, input_channels // 2, kernel_size=1, stride=strides),
                nn.BatchNorm2d(input_channels // 2),
                nn.Dropout(0.3),
                nn.PReLU()
            )
            self.K_filters['Key_'+str(i)] = nn.Sequential(
                nn.Conv2d(input_channels, input_channels // 2, kernel_size=1, stride=strides),
                nn.BatchNorm2d(input_channels // 2),
                nn.Dropout(0.3),
                nn.PReLU()
            )
            self.V_filters['Value_'+str(i)] = nn.Sequential(
                nn.Conv2d(input_channels, input_channels // 2, kernel_size=1, stride=strides),
                nn.BatchNorm2d(input_channels // 2),
                nn.Dropout(0.3),
                nn.PReLU()
            )


        self.conv = nn.Sequential(
            nn.Conv2d(input_channels // 2 * self.multi_heads, input_channels, kernel_size=1, stride=strides),
            nn.BatchNorm2d(input_channels),
            nn.Dropout(0.3),
            nn.PReLU()
        )

    def forward(self, Fres):
        """
        Fres:[M,C,T,F']
        CONV(Fres):[M,C/2,T,F']
        reshape Fres:[M,T,C/2,F']
        """
        C = Fres.shape[1] // 2
        Fr = Fres.shape[3]

        Q_results = {}
        K_results = {}
        V_results = {}
        for i in range(self.multi_heads):
            Q_results['Query_'+str(i)]= self.Q_filters['Query_'+str(i)](Fres)
            Q_results['Query_'+str(i)]= torch.transpose(Q_results['Query_'+str(i)], 1, 2)

            K_results['Key_'+str(i)] = self.K_filters['Key_'+str(i)](Fres)
            K_results['Key_'+str(i)]= torch.transpose(K_results['Key_'+str(i)], 1, 2)

            V_results['Value_'+str(i)] = self.V_filters['Value_'+str(i)](Fres)
            V_results['Value_'+str(i)] = torch.transpose(V_results['Value_'+str(i)], 1, 2)

        mul_results= {}
        trans= {}
        for i in range(self.multi_heads):
            mul_results['mul_'+str(i)] = torch.matmul(Q_results['Query_'+str(i)], \
                                        K_results['Key_'+str(i)].permute(0, 1, 3, 2))  #Query multiply with Key

            score = F.softmax(torch.divide(mul_results['mul_'+str(i)], (C * Fr) ** 0.5), dim=-1)
            SA = torch.matmul(score, V_results['Value_'+str(i)])
            trans['trans_'+str(i)]= torch.transpose(SA, 1, 2)


        output_of_selfattention=None
        for i in range(self.multi_heads):
            if i == 0:
                output_of_selfattention = trans['trans_'+str(i)]
            else:
                output_of_selfattention = torch.cat((output_of_selfattention,trans['trans_'+str(i)]),dim=1)

        result = Fres + self.conv(output_of_selfattention)
        return result


class freq_self_attention(nn.Module):
    """

    """

    def __init__(self, input_channels,strides=1,multi_heads=1):
        super(freq_self_attention,self).__init__()
        self.input_channels = input_channels
        self.strides = strides
        self.multi_heads = multi_heads  # number of heads
        self.Q_filters = nn.ModuleDict()
        self.K_filters = nn.ModuleDict()
        self.V_filters =nn.ModuleDict()
        for i in range(self.multi_heads):
            self.Q_filters['Query_' + str(i)] = nn.Sequential(
                nn.Conv2d(input_channels, input_channels // 2, kernel_size=1, stride=strides),
                nn.BatchNorm2d(input_channels // 2),
                nn.Dropout(0.3),
                nn.PReLU()
            )
            self.K_filters['Key_' + str(i)] = nn.Sequential(
                nn.Conv2d(input_channels, input_channels // 2, kernel_size=1, stride=strides),
                nn.BatchNorm2d(input_channels // 2),
                nn.Dropout(0.3),
                nn.PReLU()
            )
            self.V_filters['Value_' + str(i)] = nn.Sequential(
                nn.Conv2d(input_channels, input_channels // 2, kernel_size=1, stride=strides),
                nn.BatchNorm2d(input_channels // 2),
                nn.Dropout(0.3),
                nn.PReLU()
            )

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels // 2 *  self.multi_heads, input_channels, kernel_size=1, stride=strides),
            nn.BatchNorm2d(input_channels),
            nn.Dropout(0.3),
            nn.PReLU()
        )

    def forward(self, Fres):
        """
        Fres:[M,C,T,F']
        CONV(Fres):[M,C/2,T,F']
        reshape Fres:[M,F,C/2,T]
        """
        C = Fres.shape[1] // 2
        Fr = Fres.shape[3]

        Q_results = {}
        K_results = {}
        V_results = {}
        for i in range(self.multi_heads):
            Q_results['Query_' + str(i)] = self.Q_filters['Query_' + str(i)](Fres)
            Q_results['Query_' + str(i)] = Q_results['Query_' + str(i)].permute(0,3,1,2)

            K_results['Key_' + str(i)] = self.K_filters['Key_' + str(i)](Fres)
            K_results['Key_' + str(i)] = K_results['Key_' + str(i)].permute(0,3,1,2)

            V_results['Value_' + str(i)] = self.V_filters['Value_' + str(i)](Fres)
            V_results['Value_' + str(i)] = V_results['Value_' + str(i)].permute(0,3,1,2)

        mul_results = {}
        trans = {}
        for i in range(self.multi_heads):
            mul_results['mul_' + str(i)] = torch.matmul(Q_results['Query_' + str(i)], \
                                                        K_results['Key_' + str(i)].permute(0, 1, 3,
                                                                                           2))  # Query multiply with Key

            score = F.softmax(torch.divide(mul_results['mul_' + str(i)], (C * Fr) ** 0.5), dim=-1)
            SA = torch.matmul(score, V_results['Value_' + str(i)])
            trans['trans_' + str(i)] = SA.permute(0,2,3,1)

        output_of_selfattention = None
        for i in range(self.multi_heads):
            if i == 0:
                output_of_selfattention = trans['trans_' + str(i)]
            else:
                output_of_selfattention = torch.cat((output_of_selfattention, trans['trans_' + str(i)]), dim=1)

        result = Fres + self.conv(output_of_selfattention)
        return result



class RA_block(nn.Module):
    """
    input:encoder的输出[M,C,T,F']
    经过两个residual的输出[M,C,T,F']
    经过self-attention的两个输出[M,C,T,F']

    再把上面三个输出拼接起来： [M,3C,T,F']
    再经过一个conv：[M,C,T,F']
    """

    def __init__(self, input_channels,bias=False,multi_heads=1):
        super(RA_block,self).__init__()
        self.input_channels = input_channels

        self.Res1 = nn.Sequential(
            Residual(self.input_channels, self.input_channels),
            Residual(self.input_channels, self.input_channels)
        )

        self.time_att1 = time_self_attention(self.input_channels,multi_heads=multi_heads)
        self.freq_att1 = freq_self_attention(self.input_channels,multi_heads=multi_heads)
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels * 3, self.input_channels, kernel_size=1, stride=1,bias=bias),
            nn.BatchNorm2d(self.input_channels),
            nn.Dropout(0.3),
            nn.PReLU()
        )
        # 上下两个RAblock一起写掉了
        self.Res2 = nn.Sequential(
            Residual(self.input_channels, self.input_channels),
            Residual(self.input_channels, self.input_channels)
        )

        self.time_att2 = time_self_attention(self.input_channels,multi_heads=multi_heads)
        self.freq_att2 = freq_self_attention(self.input_channels,multi_heads=multi_heads)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels * 3, self.input_channels, kernel_size=1, stride=1,bias=bias),
            nn.BatchNorm2d(self.input_channels),
            nn.Dropout(0.3),
            nn.PReLU()
        )

    def forward(self, speech_input, noise_input):
        X1 = self.Res1(speech_input)
        X2 = self.freq_att1(X1)
        X3 = self.time_att1(X1)
        concat_feature = torch.cat([X1, X2, X3], dim=1)
        speech_result = self.conv1(concat_feature)

        Y1 = self.Res2(noise_input)
        Y2 = self.freq_att2(Y1)
        Y3 = self.time_att2(Y1)
        concat_feature = torch.cat([Y1, Y2, Y3], dim=1)
        nosie_result = self.conv2(concat_feature)

        return speech_result, nosie_result


class Interaction(nn.Module):
    def __init__(self, input_channels,bias=False):
        super(Interaction,self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=1, stride=1,bias=bias),
            nn.BatchNorm2d(input_channels),
            nn.Dropout(0.3),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=1, stride=1,bias=bias),
            nn.BatchNorm2d(input_channels),
            nn.Dropout(0.3),
            nn.Sigmoid()
        )

    def forward(self, speech_input, nosie_input):
        fusion = torch.cat([speech_input, nosie_input], dim=1)
        mask1 = self.conv1(fusion)
        mask2 = self.conv2(fusion)
        result_speech = speech_input + mask1 * nosie_input
        result_nosie = nosie_input + mask2 * speech_input

        return result_speech, result_nosie


class Separator(nn.Module):
    """
       encoder和decoder的中间部分，由四个RA_block和四个Interaction组成
       输入：来自两个encoder的输出[M,C,T,F']
       输出：也是[M,C,T,F']
    """

    def __init__(self, input_channels, box_number=4):  # box number是RAblock的个数
        super(Separator,self).__init__()
        self.input_channels = input_channels
        self.box_number = box_number
        # self.box = {}
        self.box=nn.ModuleDict()
        for i in range(box_number):
            self.box["RA"+str(i)]=RA_block(self.input_channels)
            self.box["Interaction"+str(i)]=Interaction(self.input_channels)


    def forward(self, speech_input, noise_input):

        speech_result,nosie_result=self.box["RA0"](speech_input,noise_input)
        speech_result,nosie_result=self.box["Interaction0"](speech_result,nosie_result)

        for i in range(self.box_number):
            if i==0:
                continue
            else:
                speech_result, nosie_result = self.box["RA"+str(i)](speech_result, nosie_result)
                speech_result, nosie_result = self.box["Interaction"+str(i)](speech_result, nosie_result)


        return speech_result, nosie_result


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


class Decoder(nn.Module):
    def __init__(self, channel_numbers=[32, 16, 2], Dstrides=[(1, 2), (1, 2), (1, 2)], \
                 Dkernel_sizes=[(3, 4), (3, 4), (3, 4)], Dpaddings=[[1, 1], [1, 1], [1, 1]],
                 kernel_sizes=[(1, 1), (1, 1), (1, 1)], \
                 strides=[(1, 1), (1, 1), (1, 1)], feature_channels=[32, 16, 2], Inter_Module_Out=64, \
                 Okernel_size=(1, 1), Ostride=(1, 1),bias=False):
        super(Decoder, self).__init__()
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
                                stride=Ostride,bias=bias)

    pass

    def forward(self, feature1, feature2, feature3, Inter_Module_out):
        output = self.Gateblocks['Gateblock_0'](Inter_Module_out, feature3)

        #print(output.shape)
        output = self.Gateblocks['Gateblock_1'](output, feature2)

        #print(output.shape)
        output = self.Gateblocks['Gateblock_2'](output, feature1)

        #print(output.shape)
        output = self.Conv2D(output)

        return output
        pass


class SN_net(nn.Module):
    """
    encoder + separator + decoder
    input:[M,T]     ，一般T是32K
    output:[M,C,T]  ，C是说话人的个数
    """

    def __init__(self, process_channel=320,encoder_size=[3, 5],encoder_stride=[[1, 2], [1, 2], [1, 2]],encoder_channel=[16, 32, 64],input_channels=64, box_number=4,bias=False):  # box number是RAblock的个数
        super(SN_net,self).__init__() # change here
        self.endcoder_size=encoder_size
        self.endcoder_stride=encoder_stride
        self.endcoder_channel=encoder_channel
        self.input_channels=input_channels
        self.process_channel=process_channel

        self.preprocess1=nn.Sequential(
            nn.Conv1d(1, self.process_channel//2, 16, bias=False, stride=8, padding=4),
            nn.BatchNorm1d(self.process_channel//2),
            nn.PReLU(),
            nn.Conv1d(self.process_channel//2, self.process_channel, 8, bias=False, stride=4, padding=2),
            nn.BatchNorm1d(self.process_channel),
            nn.PReLU()
        )
        self.preprocess2 = nn.Sequential(
            nn.Conv1d(1, self.process_channel // 2, 16, bias=False, stride=8, padding=4),
            nn.BatchNorm1d(self.process_channel // 2),
            nn.PReLU(),
            nn.Conv1d(self.process_channel//2, self.process_channel, 8, bias=False, stride=4, padding=2),
            nn.BatchNorm1d(self.process_channel),
            nn.PReLU()
        )


        self.speech_encoder=Encoder(encoder_size,encoder_stride,encoder_channel)  #从stft的[M,2,T,F]变成[M,C,T,F'],F'是F/4
        self.noise_encoder=Encoder(encoder_size,encoder_stride,encoder_channel)

        self.separator=Separator(input_channels)  #输入输出形状不变

        self.speech_decoder=Decoder()
        self.noise_decoder=Decoder()
        self.afterprocess1=nn.Sequential(
            nn.Conv2d(2,1,kernel_size=1,bias=False),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            # nn.ConvTranspose1d(self.process_channel,1,kernel_size=320,stride=160,bias=False,padding=80)
        )
        self.deconv1=nn.Sequential(
            nn.ConvTranspose1d(self.process_channel, self.process_channel//2, kernel_size=16, stride=8, bias=False, padding=4),
            nn.BatchNorm1d(self.process_channel//2),
            nn.PReLU(),
            nn.ConvTranspose1d(self.process_channel//2, 1, kernel_size=8, stride=4, bias=False, padding=2)
        )
        self.afterprocess2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            # nn.ConvTranspose1d(self.process_channel, 1, kernel_size=320, stride=160, bias=False, padding=80)
        )
        self.deconv2=nn.Sequential(
            nn.ConvTranspose1d(self.process_channel, self.process_channel // 2, kernel_size=16, stride=8, bias=False,
                               padding=4),
            nn.BatchNorm1d(self.process_channel // 2),
            nn.PReLU(),
            nn.ConvTranspose1d(self.process_channel // 2, 1, kernel_size=8, stride=4, bias=False, padding=2)
        )
        initialize_weights(self)

    def forward(self,input):

        pad_length = 32000 - input.shape[1]
        input = self.pad_signal(input,hop_length = pad_length)
        input=torch.unsqueeze(input,1)
        input_channel1=self.preprocess1(input)
        input_channel2=self.preprocess2(input)
        encoder_input=torch.cat([torch.unsqueeze(input_channel1,1),torch.unsqueeze(input_channel2,1)],1)

        # input = self.pad_signal(input)
        # Window = torch.hann_window(window_length=100)
        # Window = Window.cuda()
        # input = torch.stft(input,n_fft=200,hop_length=100,win_length=100,window=Window,
        #                    center=False,onesided=False)
        # encoder_input = torch.transpose(input,dim0=3,dim1=1)
        # encoder_input=encoder_input.cuda()

        speech_input,speech_result1,speech_result2,speech_result3=self.speech_encoder(encoder_input)
        noise_input, nosie_result1, nosie_result2, nosie_result3 = self.noise_encoder(encoder_input)

        sep_outs,sep_outn=self.separator(speech_input,noise_input)

        s1=self.speech_decoder(speech_result1,speech_result2,speech_result3,sep_outs)
        s2=self.noise_decoder(nosie_result1, nosie_result2, nosie_result3,sep_outn)

        s1=self.afterprocess1(s1)
        s2=self.afterprocess2(s2)



        s1=torch.squeeze(s1,1)
        s2=torch.squeeze(s2,1)

        s1=self.deconv1(s1)
        s2=self.deconv2(s2)

        result=torch.cat([s1,s2],1)
        result = torch.transpose(result,1,2)
        result = result[:,:32000-pad_length,:]
        return result

    def pad_signal(self,input, hop_length=100):
        # 输入波形: (B, T) or (B, 1, T)
        # 调整和填充

        if input.dim() not in [2]:
            raise RuntimeError("Input can only be 2.")

        batch_size = input.size(0)  # 每一个批次的大小
        pad = torch.zeros(batch_size, hop_length)
        pad = pad.cuda()
        input = torch.cat([input, pad], dim=1)

        return input

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls()
        model.load_state_dict(package['state_dict'])
        return model
    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


if __name__ == '__main__':
    signal = torch.rand(1,32000)
    pad = torch.rand(1,0)
    signal = torch.cat([signal, pad], dim=1)
    print(signal.shape)

