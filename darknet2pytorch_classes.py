from typing import List, Dict, Tuple, Callable
from torch import Tensor
import torch
import torch.nn as nn


class YOLOLayer(nn.Module):
    __constants__ = ['ONNX_EXPORT']
    def __init__(self, anchors, nc, img_size, yolo_index, ONNX_EXPORT):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
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

    def forward(self, p, img_size:List[int]):
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if not self.ONNX_EXPORT:
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids(img_size, (nx, ny), p)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        io = p.clone()  # inference output
        io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # xy
        io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
        io[..., :4] *= self.stride
        torch.sigmoid_(io[..., 4:])
        return io.view(bs, -1, self.no)

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


