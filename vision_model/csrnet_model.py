# csrnet_model.py

import torch
import torch.nn as nn
from collections import OrderedDict
import os
import gdown
import io
'''def ensure_weights_downloaded(filename):
    base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir=os.path.join(base_dir,"vision_model","weights")
    weights_path=os.path.join(weights_dir,filename)
    if os.environ.get('RENDER','false').lower()=='true' and not os.path.exists(weights_path):
        print(f"Downloading weights file {filename} on Render..")
        os.makedirs(weights_dir, exist_ok=True)

        try:
            file_id_map={
                "csrnet_pretrained.pth":"1djQ-QfUVD47rqv5gplPJHYtOKWICeA2k"
                ,"csrnet_pretrained_2.pth":"1rJBGXi7bAAiLpG8WAgq3bEyyO_VpUWY2"
            }

            if filename in file_id_map:
                gdrive_url=f"https://drive.google.com/uc?id={file_id_map[filename]}"
                gdown.download(gdrive_url, weights_path, quiet=False)
                print(f"Downloaded {filename} to {weights_path}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    return weights_path'''

def ensure_weights_downloaded(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(base_dir, "vision_model", "weights")
    weights_path = os.path.join(weights_dir, filename)

    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir, exist_ok=True)
    
    
    if os.environ.get('RENDER', 'false').lower() == 'true' and not os.path.exists(weights_path):
        print(f"Downloading weights file {filename} on Render..")
        
        try:
            file_id_map = {
                "csrnet_pretrained.pth": "1djQ-QfUVD47rqv5gplPJHYtOKWICeA2k",
                "csrnet_pretrained_2.pth": "1rJBGXi7bAAiLpG8WAgq3bEyyO_VpUWY2"
            }

            if filename in file_id_map:
                gdrive_url = f"https://drive.google.com/uc?id={file_id_map[filename]}"
                gdown.download (gdrive_url, weights_path, quiet=False,fuzzy=True)
                print(f"Downloaded {filename} to {weights_path}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    
    return weights_path

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M',
                              256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = self._make_layers(self.frontend_feat)
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _make_layers(self, cfg, in_channels=3, dilation=False):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv = nn.Conv2d(in_channels, v, kernel_size=3,
                                 padding=2 if dilation else 1,
                                 dilation=2 if dilation else 1)
                layers += [conv, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

def load_csrnet_model_1(weight_path=None, device='cpu'):
    if weight_path is None or not os.path.exists(weight_path):
        weight_path = ensure_weights_downloaded("csrnet_pretrained.pth")
    
    model = CSRNet().to(device)

    print(f"[DEBUG] Loading model from: {weight_path}")

    with open(weight_path, 'rb') as f:
        buffer=io.BytesIO(f.read())
    
    checkpoint = torch.load(buffer, map_location=device, weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def load_csrnet_model_2(weight_path=None, device='cpu'):
    if weight_path is None or not os.path.exists(weight_path):
        weight_path = ensure_weights_downloaded("csrnet_pretrained_2.pth")

    model = CSRNet().to(device)

    print(f"[DEBUG] Loading model from: {weight_path}")

    with open(weight_path, 'rb') as f:
        buffer=io.BytesIO(f.read())
    checkpoint = torch.load(buffer, map_location=device, weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

