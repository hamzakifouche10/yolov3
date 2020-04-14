import torch
from models import parse_model_cfg

def convert(cfg, img_size, ONNX_EXPORT=False, out_filename=""):
    module_defs = parse_model_cfg(cfg)

    hyperparams = module_defs.pop(0)
    yolo_index = -1

    init_str =   ("import torch\n"
                  "import torch.nn as nn\n"
                  "from darknet2pytorch_classes import YOLOLayer\n\n"
                  "class Yolo(nn.Module):\n"
                  "    def __init__(self):\n"
                  "        super(Yolo, self).__init__()\n"
                   )
    forward_str =("    def forward_impl(self, x):\n"
                  "        img_size = x.shape[-2:]\n")

    routs = set()  # set of layers which rout to deeper layers
    for i, mdef in enumerate(module_defs):
        # if mdef['type'] == 'convolutional':
        #     if not int(mdef['batch_normalize']):
        #         routs.add(i)  # detection output (goes into yolo layer)
        if mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]
            routs.update({i + l if l < 0 else l for l in layers})
        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = [int(x) for x in mdef['from'].split(',')]
            routs.update({i + l if l < 0 else l for l in layers})

    yolo_layers = []

    out_shape = [hyperparams['channels'], img_size[0], img_size[1]]
    output_filters = []
    module_names = []
    for i, mdef in enumerate(module_defs):
        if mdef['type'] == 'convolutional':
            init_str +=    "        self.conv{0} = nn.Sequential()\n".format(i)
            module_name = 'conv'
            bn = int(mdef['batch_normalize'])
            filters = mdef['filters']
            stride = (int(mdef['stride']),int(mdef['stride'])) if 'stride' in mdef else (int(mdef['stride_y']), int(mdef['stride_x']))
            padding = (int(mdef['size']) - 1) // 2 if int(mdef['pad']) else 0
            groups = int(mdef['groups']) if 'groups' in mdef else 1
            bias = not bn
            init_str +=   ("        self.conv{i}.add_module('Conv2d', nn.Conv2d(in_channels={in_channels},\n"
                           "                       out_channels={out_channels},\n"
                           "                       kernel_size={kernel_size},\n"
                           "                       stride={stride},\n"
                           "                       padding={padding},\n"
                           "                       groups={groups},\n"
                           "                       bias={bias}))\n"
                           ).format(i=i, out_channels=filters, kernel_size=mdef['size'], in_channels=out_shape[0],
                                    stride=stride, padding=padding, groups=groups, bias=bias)
            if bn:
                init_str +=    "        self.conv{i}.add_module('BatchNorm2d', nn.BatchNorm2d({filters}, momentum=0.003, eps=1E-4))\n".format(i=i, filters=filters)

            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                init_str +=    "        self.conv{i}.add_module('activation', nn.LeakyReLU(0.1, inplace=True))\n".format(i=i)
            elif mdef['activation'] == 'swish':
                assert False
            out_shape = [int(filters), out_shape[1]//stride[0], out_shape[2]//stride[1]]
            forward_str += "        x = self.conv{i}(x)  # {out_shape}\n".format(i=i, out_shape=out_shape)

        elif mdef['type'] == 'maxpool':
            init_str +=    "        self.maxpool{0} = nn.Sequential()\n".format(i)
            module_name = "maxpool"
            size = int(mdef['size'])
            stride = int(mdef['stride'])
            padding = (size - 1) // 2
            if size == 2 and stride == 1:  # yolov3-tiny
                init_str += "        self.maxpool{i}.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))\n".format(i=i)
            init_str += "        self.maxpool{i}.add_module('MaxPool2d', nn.MaxPool2d(kernel_size={size}, stride={stride}, padding={padding}))\n".format(i=i, size=size, stride=stride, padding=padding)
            out_shape = [out_shape[0], out_shape[1]//stride, out_shape[2]//stride]
            forward_str += "        x = self.maxpool{i}(x)  # {out_shape}\n".format(i=i, out_shape=out_shape)

        elif mdef['type'] == 'upsample':
            init_str +=    "        self.upsample{0} = nn.Sequential()\n".format(i)
            module_name = "upsample"
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                size = tuple(int(x * g) for x in img_size)
                init_str += "        self.upsample{i}.add_module('Upsample', nn.Upsample(size=({size[0]}, {size[1]})))\n".format(i=i, size=size)
                out_shape = [out_shape[0], size[0], size[1]]
            else:
                stride = int(mdef['stride'])
                init_str += "        self.upsample{i}.add_module('Upsample', nn.Upsample(scale_factor={stride}))\n".format(i=i, stride=stride, padding=padding)
                out_shape = [out_shape[0], out_shape[1] * stride, out_shape[2] * stride]
            forward_str += "        x = self.upsample{i}(x)  # {out_shape}\n".format(i=i, out_shape=out_shape)

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) if int(x) >= 0 else i + int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[l][0] for l in layers])
            out_shape = [filters, output_filters[layers[0]][1], output_filters[layers[0]][2]]
            if len(layers) == 1:
                forward_str += "        x = x{l}  # route {i} {out_shape}\n".format(i=i, l=layers[0], out_shape=out_shape)
            else:
                tensor_list_str = "x{0}".format(layers[0])
                if output_filters[layers[1]][1:] != output_filters[layers[0]][1:]:
                    assert len(layers) == 2
                    assert output_filters[layers[1]][1] / 2 == output_filters[layers[0]][1]
                    assert output_filters[layers[1]][2] / 2 == output_filters[layers[0]][2]
                    tensor_list_str += ", F.interpolate(x{0}, scale_factor=[0.5, 0.5])".format(layers[1])
                else:
                    tensor_list_str += ", x{0}".format(layers[1])
                forward_str += "        x = torch.cat([{tensor_list_str}], 1)  # route {i} {out_shape}\n".format(i=i, tensor_list_str=tensor_list_str, out_shape=out_shape)

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            assert False
            # layers = [int(x) for x in mdef['from'].split(',')]
            # filters = output_filters[-1]
            # modules = weightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            assert False
            pass

        elif mdef['type'] == 'reorg':
            assert False
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            mask = [int(x) for x in mdef['mask'].split(',')] if 'mask' in mdef else []
            anchors = torch.tensor([float(x) for x in mdef['anchors'].split(',')]).reshape((-1, 2))
            if len(mask) > 0:
                anchors = anchors[mask]
            init_str +=   ("        self.yolo{i} = YOLOLayer(anchors={anchors},\n"
                           "                            nc={classes},  # number of classes\n"
                           "                            img_size={img_size},  # (416, 416)\n"
                           "                            yolo_index={yolo_index},  # 0, 1, 2...\n"
                           "                            ONNX_EXPORT={ONNX_EXPORT})\n"
                           ).format(i=i, anchors=anchors.tolist(), classes=mdef['classes'], img_size=img_size,
                                    yolo_index=yolo_index, ONNX_EXPORT=ONNX_EXPORT)
            module_name = "yolo{0}"
            forward_str += "        x{i} = self.yolo{i}(x, img_size)\n".format(i=i)
            yolo_layers.append(i)
        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])
        output_filters.append(out_shape)
        module_names.append("{module_name}{i}".format(module_name=module_name, i=i))
        if i in routs:
            forward_str += "        x{0} = x\n".format(i)
    init_str += "        self.module_names = {module_names}\n".format(module_names=module_names)
    yolo_list_str = ','.join(["x{i}".format(i=i) for i in  yolo_layers])
    if yolo_list_str:
        forward_str += "        x = torch.cat([{yolo_list_str}], 1)\n".format(i=i, yolo_list_str=yolo_list_str)
    forward_str += ("        return x\n"
                    "    def forward(self, x):\n"
                    "        return self.forward_impl(x)\n")

    str = init_str + forward_str
    if out_filename:
        with open(out_filename, 'w') as f:
            f.write(str)
    return str


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    opt = parser.parse_args()
    print(opt)

    str = convert(opt.cfg, (416, 416), ONNX_EXPORT, 'yolo_plain.py')
    print(str)

    import yolo_plain
    f = yolo_plain.Yolo()
