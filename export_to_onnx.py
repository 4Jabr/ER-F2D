import torch
import torch.onnx
from model import build_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_model', type=str, required=True)
parser.add_argument('--num_enc_dec_layers', default=12, type=int)
parser.add_argument('--dim_feedforward', default=2048, type=int)
parser.add_argument('--hidden_dim', default=768, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--nheads', default=12, type=int)
parser.add_argument('--pre_norm', action='store_true')
parser.add_argument('--num_res_blocks', default=1, type=int)
args = parser.parse_args()

print("Loading PyTorch model...")
model = build_model(args)
checkpoint = torch.load(args.path_to_model)
model = torch.nn.DataParallel(model)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

print("Exporting to ONNX...")
dummy_image = torch.randn(1, 3, 224, 224).cuda()  # RGB: 3 channels
dummy_events = torch.randn(1, 5, 224, 224).cuda()  # Events: 5 channels (voxel grid)

torch.onnx.export(
    model.module,
    (dummy_image, dummy_events),
    "model.onnx",
    input_names=['image', 'events'],
    output_names=['depth'],
    dynamic_axes={'image': {0: 'batch'}, 'events': {0: 'batch'}, 'depth': {0: 'batch'}},
    opset_version=17
)

print("ONNX model saved to model.onnx")
