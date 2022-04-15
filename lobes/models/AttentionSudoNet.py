"""!
@brief SuDO-RM-RF model

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import torch
import torch.nn as nn
import math


def _padding(input, K):
    """Padding the audio times.

    Arguments
    ---------
    K : int
        Chunks of length.
    P : int
        Hop size.
    input : torch.Tensor
        Tensor of size [B, N, L].
        where, B = Batchsize,
               N = number of filters
               L = time points
    """
    N, L = input.shape
    P = K // 2
    gap = K - (P + L % K) % K
    if gap > 0:
        pad = torch.Tensor(torch.zeros(N, gap)).type(input.type())
        input = torch.cat([input, pad], dim=1)

    _pad = torch.Tensor(torch.zeros(N, P)).type(input.type())
    input = torch.cat([_pad, input, _pad], dim=1)

    return input, gap

def _Segmentation(input, K):
    """The segmentation stage splits

    Arguments
    ---------
    K : int
        Length of the chunks.
    input : torch.Tensor
        Tensor with dim [B, N, L].

    Return
    -------
    output : torch.tensor
        Tensor with dim [B, N, K, S].
        where, B = Batchsize,
           N = number of filters
           K = time points in each chunk
           S = the number of chunks
           L = the number of time points
    """
    N, L = input.shape
    P = K // 2
    input, gap = _padding(input, K)
    # [B, N, K, S]
    input1 = input[:, :-P].contiguous().view(N, -1, K)
    input2 = input[:, P:].contiguous().view(N, -1, K)

    input = (
        torch.cat([input1, input2], dim=2).view(N, -1, K).transpose(1, 2)
    )

    return input.contiguous(), gap

def _over_add(input, gap):
    """Merge the sequence with the overlap-and-add method.

    Arguments
    ---------
    input : torch.tensor
        Tensor with dim [B, N, K, S].
    gap : int
        Padding length.

    Return
    -------
    output : torch.tensor
        Tensor with dim [B, N, L].
        where, B = Batchsize,
           N = number of filters
           K = time points in each chunk
           S = the number of chunks
           L = the number of time points

    """
    N, K, S = input.shape
    P = K // 2
    # [B, N, S, K]
    input = input.transpose(1, 2).contiguous().view(N, -1, K * 2)
    input1 = input[:, :, :K].contiguous().view(N, -1)[:, P:]
    input2 = input[:, :, K:].contiguous().view(N, -1)[:, :-P]
    input = input1 + input2
    # [B, N, L]
    if gap > 0:
        input = input[:, :-gap]
    return input



def build_filter(pos, freq, POS):
    result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)



def get_dct_filter(length, mapper_x, channel):
    dct_filter = torch.zeros(channel, length)
    #dct_filter,pad= _Segmentation(dct_filter,K=20)
    #_,H,W = dct_filter.shape
    c_part = channel // len(mapper_x)
    for i, u in enumerate(mapper_x):
            for t_h in range(length):
                #for t_w in range(W):
                dct_filter[i * c_part: (i + 1) * c_part, t_h] = build_filter(t_h, u,length)

    dct_filter,pad= _Segmentation(dct_filter,K=20)
    dct_filter = _over_add(dct_filter,pad)
    return dct_filter


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
        return mapper_x,mapper_y
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        # mapper_x = all_bot_indices_x[:num_freq]
        # mapper_y = all_bot_indices_y[:num_freq]
        mapper = all_low_indices_x[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        # mapper_x = all_bot_indices_x[:num_freq]
        # mapper_y = all_bot_indices_y[:num_freq]
        mapper = all_bot_indices_x[:num_freq]
    else:
        raise NotImplementedError
    return mapper




class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_length, reduction=16, freq_sel_method='top16',K=20):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.limit_length = dct_length


        # self.Chunk_size = K
        # self.Chunk_num = 21
        mapper_x,_ = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp * (dct_length // 7) for temp in mapper_x]
        #mapper_y = [temp * (self.Chunk_num // 7) for temp in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        #self.dct_layer = MultiSpectralDCTLayer(dct_length, mapper_x, mapper_y,channel)
        self.dct_layer = MultiSpectralDCTLayer(dct_length, mapper_x,channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B,N,L = x.shape
        x_pooled = x
        if L != self.limit_length:
            x_pooled = torch.nn.functional.adaptive_avg_pool1d(x,self.limit_length)
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(B,N,1)
        return x * y.expand_as(x)




class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, length, mapper_x, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)
        self.mapper_x = mapper_x
        #self.mapper_y = mapper_y
        self.channel = channel
        # fixed DCT init
        self.register_buffer('weight', get_dct_filter(length, mapper_x,channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 3, 'x must been 3 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape
        # B,N,L = x.shape
        x = x * self.weight
        result = torch.sum(x, dim=2)
        return result




class SudoEncoder(nn.Module):

    def __init__(self,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 num_sources=2
                 ):
        super(SudoEncoder,self).__init__()
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources
        self.encoder = nn.Conv1d(in_channels=1, out_channels=enc_num_basis,
                      kernel_size=enc_kernel_size,
                      stride=enc_kernel_size // 2,
                      padding=enc_kernel_size // 2)
        torch.nn.init.xavier_uniform(self.encoder.weight)

    def forward(self,x):
        return self.encoder(x)

class SudoDecoder(nn.Module):
    def __init__(self,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 num_sources=2
                 ):
        super(SudoDecoder,self).__init__()
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_num_basis * num_sources,
            out_channels=num_sources,
            output_padding=(enc_kernel_size // 2) - 1,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            groups=num_sources)
        torch.nn.init.xavier_uniform(self.decoder.weight)
    def forward(self,x):
        return self.decoder(x.view(x.shape[0], -1, x.shape[-1]))


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())


class ConvNormAct(nn.Module):
    '''
    This class defines the convolution layer with normalization and a PReLU
    activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    '''
    This class defines the convolution layer with normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    '''
    This class defines a normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: number of output channels
        '''
        super().__init__()
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    '''
    This class defines the dilated convolution with normalized output.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class ChannelAttention(nn.Module):

    def __init__(self,channel,reduction=16):
        super(ChannelAttention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        B,N,_ = x.shape
        att = x.clone()
        att = self.avg_pool(att).permute(0,2,1)
        att = self.fc(att).permute(0,2,1)
        return att*x

class UConvBlock(nn.Module):
    '''
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    '''

    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1,
                                    stride=1, groups=1) #point wise conv
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=5,
                                           stride=1, groups=in_channels, d=1)) #Depth wise conv

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels,
                                               kSize=2*stride + 1,
                                               stride=stride,
                                               groups=in_channels, d=1))
        if upsampling_depth > 1:
            self.upsampler = torch.nn.Upsample(scale_factor=2,
                                               # align_corners=True,
                                               # mode='bicubic'
                                               )
        self.final_norm = NormAct(in_channels)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.atts = nn.ModuleList()
        self.att_norm = GlobLN(in_channels)
        #self.att = MultiSpectralAttentionLayer(channel=in_channels,dct_length=200)
        self.att = ChannelAttention(channel=in_channels)

    def forward(self, x):
        '''
        :param x: input feature map
        :return: transformed feature map
        '''
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)
        # Gather them now in reverse order
        output[-1] = self.att_norm(self.att(output[-1])+output[-1])
        for i in range(1,self.depth):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k

        expanded = self.final_norm(output[-1])


        return self.res_conv(expanded) + residual


class SuDORMRF(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 num_blocks=16,
                 upsampling_depth=4,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 num_sources=2):
        super(SuDORMRF, self).__init__()

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.enc_kernel_size // 2 * 2 **
                       self.upsampling_depth) // math.gcd(
                       self.enc_kernel_size // 2,
                       2 ** self.upsampling_depth)

        # Front end

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=enc_num_basis,
            out_channels=out_channels,
            kernel_size=1)

        # Separation module
        self.sm = nn.Sequential(*[
            UConvBlock(out_channels=out_channels,
                       in_channels=in_channels,
                       upsampling_depth=upsampling_depth)
            for _ in range(num_blocks)])

        mask_conv = nn.Conv1d(out_channels, num_sources * enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
        self.mask_nl_class = nn.ReLU()
    # Forward pass
    def forward(self, x):
        # Front end

        # Split paths
        s = x.clone()
        # Separation module
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        x = x * s.unsqueeze(1)
        # Back end
        return x

    def pad_to_appropriate_length(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(
                list(appropriate_shape[:-1]) +
                [appropriate_shape[-1] + self.lcm - values_to_pad],
                dtype=torch.float32)
            padded_x[..., :x.shape[-1]] = x
            return padded_x.to(x.device)
        return x

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]


if __name__ == "__main__":
    model = SuDORMRF(out_channels=256,
                     in_channels=512,
                     num_blocks=8,
                     upsampling_depth=5,
                     enc_kernel_size=21,
                     enc_num_basis=512,
                     num_sources=2)


    encoder = SudoEncoder()
    decoder = SudoDecoder()
    dummy_input = torch.rand(1, 1, 32000).detach().cpu()
    import time
    st = time.time()
    input = model.pad_to_appropriate_length(dummy_input)
    input = encoder(input)
    estimated_sources = model(input)
    estimated_sources = decoder(estimated_sources)
    print("time taken(total):", time.time() - st)
    estimated_sources = model.remove_trailing_zeros(estimated_sources,dummy_input)
    # from ptflops import get_model_complexity_info
    # import sys
    # ost = sys.stdout
    #
    #
    #
    # flops, params = get_model_complexity_info(model, (512,3200),
    #                                           as_strings=True,
    #                                           print_per_layer_stat=True,
    #                                           ost=ost)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))



