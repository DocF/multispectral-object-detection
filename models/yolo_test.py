# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


#
# """
#     FQY    Build Two Stream Detection Model
# """
# class TwostreamNet(nn.Module):
#     def __init__(self, rgb_stream, ir_stream, my_layer):    # rgb stream, ir stream, rest detect network
#         super(TwostreamNet, self).__init__()
#
#         self.rgb_stream = rgb_stream
#         self.ir_stream = ir_stream
#         self.my_layer = my_layer
#         self.detect = my_layer[-1][0]
#
#     def forward(self, rgb_input, ir_input):
#
#         # ---------------- Inputs ----------------------
#         x = self.rgb_stream(rgb_input)
#         y = self.ir_stream(ir_input)
#
#         # Concatenate the two stream inputs
#         x = torch.cat((x, y), dim=1).to(device)
#         x = Conv(c1=x.shape[1], c2=128, k=3).to(device)(x)
#
#         # print("Two Stream Output")
#         # print(x.shape)
#         # print()
#
#         # # YOLOv5 backbone
#         # backbone:
#         # # [from, number, module, args]
#         # 0 [[-1, 1, Focus, [64, 3]],  # 0-P1/2
#         # 1 [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
#         # 2 [-1, 3, C3, [128]],
#         # 3 [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
#         # 4 [-1, 9, C3, [256]],
#         # 5 [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
#         # 6 [-1, 9, C3, [512]],
#         # 7 [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
#         # 8 [-1, 1, SPP, [1024, [5, 9, 13]]],
#         # 9 [-1, 3, C3, [1024, False]],  # 9
#         #  ]
#         #
#         # # YOLOv5 head
#         # head:
#         # 10 [[-1, 1, Conv, [512, 1, 1]],
#         # 11 [-1, 1, nn.Upsample, [None, 2, 'nearest']],
#         # 12 [[-1, 6], 1, Concat, [1]],  # cat backbone P4
#         # 13 [-1, 3, C3, [512, False]],  # 13
#         #
#         # 14 [-1, 1, Conv, [256, 1, 1]],
#         # 15 [-1, 1, nn.Upsample, [None, 2, 'nearest']],
#         # 16 [[-1, 4], 1, Concat, [1]],  # cat backbone P3
#         # 17 [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
#         #
#         # 18 [-1, 1, Conv, [256, 3, 2]],
#         # 19 [[-1, 14], 1, Concat, [1]],  # cat head P4
#         # 20 [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
#         #
#         # 21 [-1, 1, Conv, [512, 3, 2]],
#         # 22 [[-1, 10], 1, Concat, [1]],  # cat head P5
#         # 23 [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
#         #
#         # 24 [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
#         #  ]
#
#
#         # Rest Layer
#         x = self.my_layer[0][0].to(device)(x)
#         feature_4 = x
#         x = self.my_layer[1][0].to(device)(x)
#         x = self.my_layer[2][0].to(device)(x)
#         feature_6 = x
#         x = self.my_layer[3][0].to(device)(x)
#         x = self.my_layer[4][0].to(device)(x)
#         x = self.my_layer[5][0].to(device)(x)
#         x = self.my_layer[6][0].to(device)(x)
#         feature_10 = x
#         x = self.my_layer[7][0].to(device)(x)
#         x = self.my_layer[8][0].to(device)([x, feature_6])          # Concatenate
#         x = self.my_layer[9][0].to(device)(x)
#         x = self.my_layer[10][0].to(device)(x)
#         feature_14 = x
#         x = self.my_layer[11][0].to(device)(x)
#         x = self.my_layer[12][0].to(device)([x, feature_4])          # Concatenate
#         x = self.my_layer[13][0].to(device)(x)
#         out_1 = x
#         x = self.my_layer[14][0].to(device)(x)
#         x = self.my_layer[15][0].to(device)([x, feature_14])
#         x = self.my_layer[16][0].to(device)(x)
#         out_2 = x
#         x = self.my_layer[17][0].to(device)(x)
#         x = self.my_layer[18][0].to(device)([x, feature_10])      # Concatenate
#         x = self.my_layer[19][0].to(device)(x)
#         x = self.detect.to(device)([out_1, out_2, x])
#
#         return x
#
#
class Model(nn.Module):

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict

        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict
            # print("YAML")
            # print(self.yaml)

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # print(self.model)
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        # print(m)

        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # print("1, ch, s, s", 1, ch, s, s)
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.Tensor([8.0, 16.0, 32.0])
            # print("m.stride", m.stride)
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, x2, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, x2, profile)  # single-scale inference, train


    def forward_once(self, x, x2, profile=False):
        """

        :param x:          RGB Inputs
        :param x2:         IR  Inputs
        :param profile:
        :return:
        """
        y, dt = [], []  # outputs
        i = 0
        for m in self.model:
            # print("Moudle", i)
            if m.f != -1:  # if not from previous layer
                if m.f != -4:
                    # print(m)
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            if m.f == -4:
                x = m(x2)
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            # print(len(y))
            i+=1

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

#
# class TwoStreamModel(nn.Module):
#
#     def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
#         super(TwoStreamModel, self).__init__()
#         if isinstance(cfg, dict):
#             self.yaml = cfg  # model dict
#
#         else:  # is *.yaml
#             import yaml  # for torch hub
#             self.yaml_file = Path(cfg).name
#             with open(cfg) as f:
#                 self.yaml = yaml.safe_load(f)  # model dict
#             # print("YAML")
#             # print(self.yaml)
#
#         # Define model
#         ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
#         if nc and nc != self.yaml['nc']:
#             logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
#             self.yaml['nc'] = nc  # override yaml value
#         if anchors:
#             logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
#             self.yaml['anchors'] = round(anchors)  # override yaml value
#         self.model, self.save = parse_model_rgb_ir(deepcopy(self.yaml), ch=[ch])  # model, savelist
#         # print(self.model)
#         self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
#         # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])
#
#         # Build strides, anchors
#         # m = self.model[-1]  # Detect()
#         # print(type(self.model))
#         # m = self.model["rest_net"]  # Detect()
#         m = list(self.model.children())[-1]
#         # self.stride = None
#         if isinstance(m, Detect):
#             s = 256  # 2x min stride
#             m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
#             m.anchors /= m.stride.view(-1, 1, 1)
#             check_anchor_order(m)
#             self.stride = m.stride
#             self._initialize_biases()  # only run once
#             # logger.info('Strides: %s' % m.stride.tolist())
#
#         # Init weights, biases
#         initialize_weights(self)
#         self.info()
#         logger.info('')
#
#     def forward(self, x, augment=False, profile=False):
#         if augment:
#             img_size = x.shape[-2:]  # height, width
#             s = [1, 0.83, 0.67]  # scales
#             f = [None, 3, None]  # flips (2-ud, 3-lr)
#             y = []  # outputs
#             for si, fi in zip(s, f):
#                 xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
#                 yi = self.forward_once(xi)[0]  # forward
#                 # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
#                 yi[..., :4] /= si  # de-scale
#                 if fi == 2:
#                     yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
#                 elif fi == 3:
#                     yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
#                 y.append(yi)
#             return torch.cat(y, 1), None  # augmented inference, train
#         else:
#             return self.forward_once(x, profile)  # single-scale inference, train
#
#     def forward_once(self, x, profile=False):
#         y, dt = [], []  # outputs
#         for m in self.model:
#             if m.f != -1:  # if not from previous layer
#                 # print(m)
#                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
#                 # print(x)
#
#             if profile:
#                 o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
#                 t = time_synchronized()
#                 for _ in range(10):
#                     _ = m(x)
#                 dt.append((time_synchronized() - t) * 100)
#                 if m == self.model[0]:
#                     logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
#                 logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
#
#             x = m(x)  # run
#             y.append(x if m.i in self.save else None)  # save output
#
#         if profile:
#             logger.info('%.1fms total' % sum(dt))
#         return x
#
#     def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
#         # https://arxiv.org/abs/1708.02002 section 3.3
#         # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
#         m = self.model[-1]  # Detect() module
#         for mi, s in zip(m.m, m.stride):  # from
#             b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
#             b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
#             b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
#             mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
#
#     def _print_biases(self):
#         m = self.model[-1]  # Detect() module
#         for mi in m.m:  # from
#             b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
#             logger.info(
#                 ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
#
#     # def _print_weights(self):
#     #     for m in self.model.modules():
#     #         if type(m) is Bottleneck:
#     #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights
#
#     def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
#         logger.info('Fusing layers... ')
#         for m in self.model.modules():
#             if type(m) is Conv and hasattr(m, 'bn'):
#                 m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
#                 delattr(m, 'bn')  # remove batchnorm
#                 m.forward = m.fuseforward  # update forward
#         self.info()
#         return self
#
#     def nms(self, mode=True):  # add or remove NMS module
#         present = type(self.model[-1]) is NMS  # last layer is NMS
#         if mode and not present:
#             logger.info('Adding NMS... ')
#             m = NMS()  # module
#             m.f = -1  # from
#             m.i = self.model[-1].i + 1  # index
#             self.model.add_module(name='%s' % m.i, module=m)  # add
#             self.eval()
#         elif not mode and present:
#             logger.info('Removing NMS... ')
#             self.model = self.model[:-1]  # remove
#         return self
#
#     def autoshape(self):  # add autoShape module
#         logger.info('Adding autoShape... ')
#         m = autoShape(self)  # wrap model
#         copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
#         return m
#
#     def info(self, verbose=False, img_size=640):  # print model information
#         model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # print("ch", ch)
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:

            if m is Focus:
                c1, c2 = 3, args[0]
                # print("focus c2", c2)
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            else:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR]:
                    args.insert(2, n)  # number of repeats
                    n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Add:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add2:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            # print("Add2 arg", args[0])
            args = [c2, args[1]]
        elif m is GPT:
            c2 = ch[f[0]]
            args = [c2]
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        # if i == 4:
        #     ch = []
        ch.append(c2)
    # print(layers)
    return nn.Sequential(*layers), sorted(save)


def parse_model_rgb_ir(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 =  [], [], ch[-1]  # layers, savelist, ch out

    for i, (f, n, m, args) in enumerate(d['backbone']+ d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)


    layers_rgb = layers[:4].copy()
    layer_ir = layers[:4].copy()
    rgb_stream = nn.Sequential(*layers_rgb)
    ir_stream = nn.Sequential(*layer_ir)

    # 以concat为界，分割模型
    my_layer = []
    for i in range(4, len(layers)):
        my_layer.append([layers[i]].copy())

    # print("My Layer")
    # print(len(my_layer))
    # for i in range(len(my_layer)):
    #     print(my_layer[i])
    # layer_4 = layers[4].copy()
    # layer_5 = layers[5].copy()
    # layers_rest = layers[4:].copy()
    # rest_net = nn.Sequential(*layers_rest)
    # print(rest_net)
    # print(" REST Net")
    # print(rest_net)

    model = TwostreamNet(rgb_stream, ir_stream, my_layer)
    print("Two Stream Model")
    print(model)

    # return nn.Sequential(*layers), sorted(save)
    return model, sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/home/fqy/proj/paper/YOLOFusion/models/transformer/yolov5s_fusion_transformer(x3)_vedai.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)
    print(device)


    model = Model(opt.cfg).to(device)
    input_rgb = torch.Tensor(8, 3, 640, 640).to(device)
    input_ir = torch.Tensor(8, 3, 640, 640).to(device)

    output = model(input_rgb, input_ir)
    print("YOLO")
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    # print(output)

    # # Create model
    # model =TwoStreamModel(opt.cfg).to(device)
    # print(model)
    # input_rgb = torch.Tensor(8, 3, 640, 640).to(device)
    # input_ir = torch.Tensor(8, 3, 640, 640).to(device)
    # output = model.model(input_rgb, input_ir)
    # print("YOLO Fusion")
    # print(output[0].shape)
    # print(output[1].shape)
    # print(output[2].shape)
    # print(output.shape)

    # print(model)
    # model.train()
    # torch.save(model, "yolov5s.pth")

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
