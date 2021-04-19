import torch
import struct
import sys 
sys.path.insert(1,'/home/altex/yolov5')
from utils.torch_utils import select_device

# Initialize
device = select_device('cpu')
# Load model
model_path = '/home/altex/Models-YoloV5_r4/yolov5s_480/exp/weights/best.pt'
model = torch.load(model_path, map_location=device)['model'].float()  # load to FP32
model.to(device).eval()

f = open(model_path.replace('.pt','.wts'), 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')
