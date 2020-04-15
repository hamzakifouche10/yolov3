from typing import List, Dict, Tuple, Callable
import torch
import torch.nn as nn
import darknet2pytorch_convert
import models
from utils import torch_utils

def convert(cfg, img_size, ONNX_EXPORT=False, out_filename=""):
    return darknet2pytorch_convert.convert(cfg, img_size, ONNX_EXPORT, out_filename)


def convert_state_dict_keys(state_dict, module_names):
    '''
    converts state_dict from https://github.com/ultralytics/yolov3 with module_list.*.* keys to new keys like conv0.*
    :param state_dict:
    :param module_names: list on new modules from model.module_names
    :return: new dict to load to model
    '''
    new_dict = dict()
    for k,v in state_dict.items():
        names = k.split(".", 2)
        assert len(names) == 3
        assert names[0] == 'module_list', k
        idx = int(names[1])
        assert idx<len(module_names), (k, len(module_names))
        new_dict[module_names[idx]+"."+names[2]] = v
    return new_dict


def fuse(module, name = 'model'):
    # Fuse Conv2d + BatchNorm2d layers throughout model
    if isinstance(module, nn.Sequential):
        for i in range(len(module)-1, 0, -1):
            bn_layer = module[i]
            if isinstance(bn_layer, nn.modules.batchnorm.BatchNorm2d):
                # fuse this bn layer with the previous conv2d layer
                conv = module[i - 1]
                if isinstance(conv, nn.modules.conv.Conv2d):
                    print('fusing ', name)
                    fused = torch_utils.fuse_conv_and_bn(conv, bn_layer)
                    module[i - 1] = fused
                    module.__delitem__(i)
    for name, child in module.named_children():
        fuse(child, name)

# non_max_suppression = torch.jit.script(models.non_max_suppression)

def create_yolo_with_nms(YoloClass, conf_thres=0.1, iou_thres=0.6, classes=[], agnostic_nms=False, half=False):
    class YoloWithNMS(YoloClass):
        '''
        returns list of (rects, best_class_id, best_class_score, class_scores)
        '''
        module_defs: List[Dict[str, str]]
        routs: List[bool]
        route_layers: List[List[int]]
        strides: List[int]
        classes: List[int]
        non_max_suppression: Callable
        def __init__(self, conf_thres=0.1, iou_thres=0.6, classes=[], agnostic_nms=False, half=False):
            super(YoloWithNMS, self).__init__()
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.classes = classes if classes is not None else []
            self.agnostic_nms = agnostic_nms
            self.half = half
        def load_state_dict(self, state_dict):
            import darknet2pytorch_convert
            super(YoloWithNMS, self).load_state_dict(convert_state_dict_keys(state_dict, self.module_names))
        def forward(self, img):
            if len(img.shape) == 3:
                if img.shape[-1] < 4:
                    img = img.permute(2,0,1)
                img = img.unsqueeze(0)
            if img.dtype == torch.uint8:
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
            pred = self.forward_impl(img)
            if self.half:
                pred = pred.float()
            res = models.non_max_suppression(pred.cpu(), self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            return [(r[:, :4], r[:, 5].long(), r[:, 4], r[:, 6:]) for r in res]
    return YoloWithNMS(conf_thres, iou_thres, classes, agnostic_nms, half)
