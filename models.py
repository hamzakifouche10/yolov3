from typing import List, Dict, Tuple
from torch import Tensor
from torch.nn import ModuleList, Module
import torch.nn.functional as F

from utils.google_utils import *
from utils.parse_config import *
from utils.utils import *

ONNX_EXPORT = True

class CallsExt(object):
    def call1(self, x):
        raise NotImplementedError
        return torch.tensor(0)
    def call2(self, x1, x2: List[Tensor]):
        raise NotImplementedError
        return torch.tensor(0)
    def call3(self, x1, x2: List[int], x3: List[Tensor]):
        raise NotImplementedError
        return torch.tensor(0), torch.tensor(0)

class ExtSequential(CallsExt, nn.Sequential):
    def __init__(self):
        super(ExtSequential, self).__init__()
    @torch.jit.export
    def call1(self, x):
         return self(x)

def create_modules(module_defs, img_size):
    # Constructs module list of layer blocks from module configuration in module_defs

    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = ExtSequential()
        # if i == 0:
        #     modules.add_module('BatchNorm2d_0', nn.BatchNorm2d(output_filters[-1], momentum=0.1))

        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            size = int(mdef['size'])
            stride = int(mdef['stride']) if 'stride' in mdef else (int(mdef['stride_y']), int(mdef['stride_x']))
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=size,
                                                   stride=stride,
                                                   padding=(size - 1) // 2 if int(mdef['pad']) else 0,
                                                   groups=int(mdef['groups']) if 'groups' in mdef else 1,
                                                   bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.003, eps=1E-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())

        elif mdef['type'] == 'maxpool':
            size = int(mdef['size'])
            stride = int(mdef['stride'])
            maxpool = nn.MaxPool2d(kernel_size=size, stride=stride, padding=(size - 1) // 2)
            if size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules.add_module('MaxPool2d', maxpool)

        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules.add_module('Upsample', nn.Upsample(size=tuple(int(x * g) for x in img_size)))  # img_size = (320, 192)
            else:
                modules.add_module('Upsample', nn.Upsample(scale_factor=int(mdef['stride'])))

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = [int(x) for x in mdef['from'].split(',')]
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = weightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        elif mdef['type'] == 'reorg':
            assert len(mdef) == 2 and 'stride' in mdef and mdef['stride'] != 0
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            l = [int(x) for x in mdef['from'].split(',')] if 'from' in mdef else []
            mask = [int(x) for x in mdef['mask'].split(',')] if 'mask' in mdef else []
            anchors = torch.tensor([float(x) for x in mdef['anchors'].split(',')]).reshape((-1, 2))
            if len(mask) > 0:
                anchors = anchors[mask]
            modules = YOLOLayer(anchors=anchors,  # anchor list
                                nc=int(mdef['classes']),  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=l)  # output layers

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                bo = -4.5  #  obj bias
                bc = math.log(1 / (modules.nc - 0.99))  # cls bias: class probability is sigmoid(p) = 1/nc

                j = l[yolo_index] if 'from' in mdef else -1
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                bias[:, 4] += bo - bias[:, 4].mean()  # obj
                bias[:, 5:] += bc - bias[:, 5:].mean()  # cls, view with utils.print_model_biases(model)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class weightedFeatureFusion(CallsExt, nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        super(weightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = torch.nn.Parameter(torch.zeros(self.n))  # layer weights
        else:
            self.w = torch.tensor(0)

    def forward(self, x, outputs: List[Tensor]):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]
        else:
            w = torch.tensor(0)

        # Fusion
        nc = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            ac = a.shape[1]  # feature channels
            dc = nc - ac  # delta channels

            # Adjust channels
            if dc > 0:  # slice input
                x[:, :ac] = x[:, :ac] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            elif dc < 0:  # slice feature
                x = x + a[:, :nc]
            else:  # same shape
                x = x + a
        return x
    def call2(self, x1, x2: List[Tensor]):
        return self(x1, x2)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x.mul_(torch.sigmoid(x))


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x.mul_(F.softplus(x).tanh())


class YOLOLayer(CallsExt, nn.Module):
    layers: List[int]
    def __init__(self, anchors, nc, img_size, yolo_index, layers: List[int]):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
        self.img_size = 0
        self.stride = 0.
        self.grid_xy = Tensor()
        self.anchor_vec = Tensor()
        self.anchor_wh = Tensor()
        self.ng = Tensor()
        self.ONNX_EXPORT = ONNX_EXPORT

        if ONNX_EXPORT:
            stride = [32, 16, 8][yolo_index]  # stride of this layer
            nx = img_size[1] // stride  # number x grid points
            ny = img_size[0] // stride  # number y grid points
            self.create_grids(img_size, (nx, ny))

    def forward(self, p, img_size:List[int], out: List[Tensor])->Tuple[Tensor, Tensor]:
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        # if not self.ONNX_EXPORT:
        #     if (self.nx, self.ny) != (nx, ny):
        #         self.create_grids(img_size, (nx, ny), p)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        io = p.clone()  # inference output
        io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # xy
        io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
        io[..., :4] *= self.stride
        torch.sigmoid_(io[..., 4:])
        return io.view(bs, -1, self.no), p

    def call3(self, x1, x2: List[int], x3: List[Tensor]):
        return self(x1,x2,x3)

    def create_grids(self, img_size:List[int]=(416,416), ng:Tuple[int,int] =(13, 13), as_tensor=torch.tensor(0, device='cpu', dtype=torch.float32)):
        nx, ny = ng  # x and y grid size
        self.img_size = max(img_size)
        self.stride = self.img_size / max(ng)

        # build xy offsets
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        self.grid_xy = torch.stack((xv, yv), 2).to(as_tensor).view((1, 1, ny, nx, 2))

        # build wh gains
        self.anchor_vec = self.anchors.to(as_tensor) / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(as_tensor)
        self.ng = torch.tensor(ng).to(as_tensor)
        self.nx = nx
        self.ny = ny


class Darknet(nn.Module):
    # YOLOv3 object detection model
    module_defs: List[Dict[str, str]]
    routs: List[bool]
    route_layers: List[List[int]]
    strides: List[int]
    def __init__(self, cfg, img_size=(416, 416)):
        super(Darknet, self).__init__()

        self.img_size = 0

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        self.route_layers = [([int(layer) for layer in mdef['layers'].split(',')] if 'layers' in mdef else []) for mdef in self.module_defs]
        self.strides = [(int(mdef['stride']) if 'stride' in mdef else 0) for mdef in self.module_defs]
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info()  # print model description

    def core_forward(self, x):
        img_size = x.shape[-2:]
        yolo_out_io: List[Tensor] = []
        yolo_out_p: List[Tensor] = []
        out: List[Tensor] = []
        i = -1
        for module in self.module_list:
            i += 1
            mdef = self.module_defs[i]
            # module = ExtModule(module)
            mtype = mdef['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                assert(len(x.shape)>2)
                x = module.call1(x)
            elif mtype == 'shortcut':  # sum
                x = module.call2(x, out)  # weightedFeatureFusion()
            elif mtype == 'route':  # concat
                layers = self.route_layers[i]
                if len(layers) == 1:
                    x = out[layers[0]]
                else:
                    if out[layers[1]].shape[2:] != out[layers[0]].shape[2:]:
                        out[layers[1]] = F.interpolate(out[layers[1]], scale_factor=[0.5, 0.5])
                    for layer_no in layers[1:]:
                        assert out[layer_no].shape[2:] == out[layers[0]].shape[2:],  str(out[layers[0]].shape) +" " + str(out[layer_no].shape) + " " + str(layer_no)
                    x = torch.cat([out[layer] for layer in layers], 1)
                    # print(''), [print(out[layer].shape) for layer in layers], print(x.shape)
            elif mtype == 'reorg':  # concat
                x = F.interpolate(x, scale_factor=[1/self.strides[i], 1/self.strides[i]])
            elif mtype == 'yolo':
                io, p = module.call3(x, img_size, out)
                yolo_out_io.append(io)
                yolo_out_p.append(p)
            else:
                raise Exception("Unsupported layer type [" + mtype + "]")
            out.append(x if self.routs[i] else torch.tensor(0))

        return torch.cat(yolo_out_io, 1), yolo_out_p

    def forward(self, x):
        return self.core_forward(x)

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info()  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


class DarknetInferenceWithNMS(Darknet):
    '''
    returns list of (rects, best_class_id, best_class_score, class_scores)
    '''
    module_defs: List[Dict[str, str]]
    routs: List[bool]
    route_layers: List[List[int]]
    strides: List[int]
    classes: List[int]
    def __init__(self, cfg, img_size=(416, 416), conf_thres=0.1, iou_thres=0.6, classes=[], agnostic_nms=False, half=False):
        super(DarknetInferenceWithNMS, self).__init__(cfg, img_size)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes if classes is not None else []
        self.agnostic_nms = agnostic_nms
        self.half = half
    def forward(self, img):
        if len(img.shape) == 3:
            if img.shape[-1] < 4:
                img = img.permute(2,0,1)
            img = img.unsqueeze(0)
        if img.dtype == torch.uint8:
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
        pred = self.core_forward(img)[0]
        if self.half:
            pred = pred.float()
        res = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        return [(r[:, :4], r[:, 5].long(), r[:, 4], r[:, 6:]) for r in res]


def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if int(mdef['batch_normalize']):
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if int(mdef['batch_normalize']):
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        torch.save(chkpt, 'converted.pt')
        print("Success: converted '%s' to 'converted.pt'" % weights)

    else:
        print('Error: extension not supported.')


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    msg = weights + ' missing, try downloading from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0'

    if weights and not os.path.isfile(weights):
        d = {'yolov3-spp.weights': '16lYS4bcIdM2HdmyJBVDOvt3Trx6N3W2R',
             'yolov3.weights': '1uTlyDWlnaqXcsKOktP5aH_zRDbfcDp-y',
             'yolov3-tiny.weights': '1CCF-iNIIkYesIDzaPvdwlcf7H9zSsKZQ',
             'yolov3-spp.pt': '1f6Ovy3BSq2wYq4UfvFUpxJFNDFfrIDcR',
             'yolov3.pt': '1SHNFyoe5Ni8DajDNEqgB2oVKBb_NoEad',
             'yolov3-tiny.pt': '10m_3MlpQwRtZetQxtksm9jqHrPTHZ6vo',
             'darknet53.conv.74': '1WUVBid-XuoUBmvzBVUCBl_ELrzqwA8dJ',
             'yolov3-tiny.conv.15': '1Bw0kCpplxUqyRYAJr9RY9SGnOJbo9nEj',
             'ultralytics49.pt': '158g62Vs14E3aj7oPVPuEnNZMKFNgGyNq',
             'ultralytics68.pt': '1Jm8kqnMdMGUUxGo8zMFZMJ0eaPwLkxSG',
             'yolov3-spp-ultralytics.pt': '1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4'}

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        else:  # download from pjreddie.com
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            r = os.system('curl -f ' + url + ' -o ' + weights)

        # Error check
        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.system('rm ' + weights)  # remove partial downloads
            raise Exception(msg)
