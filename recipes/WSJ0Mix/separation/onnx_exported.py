import os.path as osp
import torch
import numpy as np
import torchvision
import onnx
import onnxruntime as ort
#give it a try
import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
from torch.cuda.amp import autocast
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import logging
import sys
from torch import nn



def Sepformer_exported():
    # # Load hyperparameters file with command-line overrides
    # hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # with open(hparams_file) as fin:
    #     hparams = load_hyperpyyaml(fin, overrides)
    #
    # # Initialize ddp (useful only for multi-GPU DDP training)
    # sb.utils.distributed.ddp_init_group(run_opts)
    #
    # # Logger info
    # logger = logging.getLogger(__name__)
    #
    # # Load pretrained model if pretrained_separator is present in the yaml
    # if "pretrained_separator" in hparams:
    #     run_on_main(hparams["pretrained_separator"].collect_files)
    #     hparams["pretrained_separator"].load_collected()
    #
    # # Brain class initialization
    # separator = Separation(
    #     modules=hparams["modules"],
    #     opt_class=hparams["optimizer"],
    #     hparams=hparams,
    #     run_opts=run_opts,
    #     checkpointer=hparams["checkpointer"],
    # )
    #
    # # re-initialize the parameters if we don't use a pretrained model
    # if "pretrained_separator" not in hparams:
    #     for module in separator.modules.values():
    #         separator.reset_layer_recursively(module)

    #model loader
    # root_path = './results/sepformer/1234/save/CKPT+2021-03-08+20-49-50+00/'
    # hparams = {}
    # hparams['encoder'] = root_path + 'encoder.ckpt'
    # hparams['maskner'] = root_path + 'masknet.ckpt'
    # hparams['decoder'] = root_path + 'decoder.ckpt'
    # Real_Model = Sepformer(separator.hparams)

    Real_Model = torch.load("Improved_Sudormrf_U16_Bases512_WSJ02mix.pt",map_location=torch.device('cpu'))
    dummy_input = torch.randn(1,64000,device='cpu')
    model = Real_Model.eval()
    input_names = ["raw_audio"]
    output_names = ["est_audio"]
    if not osp.exists('sepformer.onnx'):
        # translate your pytorch model to onnx
        torch.onnx.export(model, dummy_input,"SudoNet.onnx", verbose=True, input_names=input_names,
                          output_names=output_names,opset_version=11,dynamic_axes = {'raw_audio':[0,1],'est_audio':[0,1]})

    pass



def quantize(model_name):
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU
    Uses unsigned ints for activation values, signed ints for weights, per
    https://onnxruntime.ai/docs/performance/quantization.html#data-type-selection
    it is faster on most CPU architectures
    Args:
        onnx_model_path: Path to location the exported ONNX model is stored
    Returns: The Path generated for the quantized
    """
    from onnxruntime.quantization import QuantizationMode, quantize

    onnx_model = onnx.load(model_name)
    quantized_model = quantize(
        model=onnx_model,
        quantization_mode=QuantizationMode.IntegerOps,
        force_fusions=True,
        symmetric_weight=True,
    )
    onnx.save(quantized_model,"sepformer-quantized.onnx")



if __name__ == '__main__':
    Sepformer_exported()
    #quantize("sepformer.onnx")
    # ort_session = ort.InferenceSession('Sudormrf-sim.onnx')
    #
    # import time
    #
    # test_arr = np.random.randn(1, 1, 427918).astype(np.float32)
    # st = time.time()
    # outputs = ort_session.run(None, {'raw_audio': test_arr})
    # et = time.time()
    # print("time Spending:",et-st)
    # print('onnx result:', outputs[0].shape)
    # dummy_input = torch.randn(10, 3, 224, 224, device='cpu')
    # print('pytorch result:', model(torch.from_numpy(test_arr)))
    #
    # input_names = ["input"]
    # output_names = ["output"]
    #
    # if not osp.exists('resnet50.onnx'):
    #     # translate your pytorch model to onnx
    #     torch.onnx.export(model, dummy_input, "resnet50.onnx", verbose=True, input_names=input_names,
    #                       output_names=output_names)
    #