import io
import numpy as np

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
# import torch.onnx
import onnx
# import onnxruntime

#Function to Convert to ONNX 
def Convert_ONNX(model_path): 

    # set the model to inference mode 
    model = torch.load(model_path)
    model = model.to(device)
    print(model)
    model.eval() 

    # Let's create a dummy input   
    input1 = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)  
    input2 = torch.randn(1, 1, 8, 8, requires_grad=True).to(device)  
    dummy_input = (input1, input2)
    # dummy_input = dummy_input.to(device)

    # Export the model   
    torch.onnx.export(model,         # model being run 
                      dummy_input,       # model input (or a tuple for multiple inputs) 
                      "nyu.onnx",       # where to save the model  
                      export_params=True,  # store the trained parameter weights inside the model file 
                      opset_version=12,    # the ONNX version to export the model to 
                      do_constant_folding=True,  # whether to execute constant folding for optimization 
                      input_names = ["rgb", "depth"],   # the model's input names 
                      output_names = ["output depth"]) # the model's output names 
                      # dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                            # 'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')

if __name__ == '__main__':
    model_path = './result/test/ep22_71.34_rmse0.744.pth'
    device = 'cuda:0'

    Convert_ONNX(model_path)
