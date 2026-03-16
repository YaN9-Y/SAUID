import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.autograd as autograd
from functools import reduce
from math import exp
from .utils import rgb2hsv
from torch.nn.modules.utils import _triple, _pair, _single



class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]






class SegmentEntropyLoss(nn.Module):
    def __init__(self):
        super(SegmentEntropyLoss, self).__init__()

    def __call__(self, segment_result):
        semantic_map_prop = torch.softmax(segment_result, dim=1)
        semantic_entropy = torch.mean(-torch.log(semantic_map_prop) * semantic_map_prop)
        return semantic_entropy





class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss



class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            #'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            #'relu3_2': relu3_2,
            #'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            #'relu4_2': relu4_2,
            #'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            #'relu5_3': relu5_3,
            #'relu5_4': relu5_4,
        }
        return out


class SimilarityContrastLoss(nn.Module):
    def __init__(self):
        super(SimilarityContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.weights = [0, 0.4, 0.6, 0, 1.0]

    def compute_similarity(self, x, y):
        b, ch, h, w = x.size()
        x = x.view(b, ch, w * h)
        y = y.view(b, ch, w * h)
        y_T = y.transpose(1, 2)
        G = x.bmm(y_T) / (h * w * ch)
        self.weights = [0, 0.4, 0.6, 0, 1.0]

        return G

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)

        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = torch.mean(torch.abs(self.compute_similarity(a_vgg[i], p_vgg[i])))
            # if not self.ab:
            #     d_an = self.l1(a_vgg[i], n_vgg[i])
            #     contrastive = d_ap / (d_an + 1e-7)
            # else:
            #     contrastive = d_ap
            d_an = torch.mean(torch.abs(self.compute_similarity(a_vgg[i], n_vgg[i])))
            contrastive = d_an / ( d_ap + 1e-7)


            loss += self.weights[i] * contrastive
        return loss



class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()
        self.weight = torch.tensor([[[[0,1,0],[1, -4 ,1], [0 ,1 ,0]]]]).float().cuda().expand(1,256,3,3)
        self.criterion = nn.L1Loss()
    def __call__(self, x, y):
        return self.criterion(F.conv2d(x,self.weight,padding=1), F.conv2d(y,self.weight,padding=1)) / 256


class DecoderFeatureLoss(nn.Module):
    def __init__(self):
        super(DecoderFeatureLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def __call__(self, x, y, weight=[1.0,1.0,1.0,1.0,1.0]): #convconv64_256, deconv64_128_128,conv128_128,deconv128_256_64,conv256_3
        assert len(weight) == 5 and len(x) == len(y)
        decoder_feature_loss = 0.
        for i in range(len(x)):
            decoder_feature_loss += weight[i] * self.criterion(x[i],y[i])

        return decoder_feature_loss



class DarkChannelLoss(nn.Module):
    def __init__(self):
        super(DarkChannelLoss, self).__init__()

    def forward(self, pred_img):
        return torch.mean(torch.abs(self.get_dark_channel(pred_img)))


    def get_dark_channel(input, pred_img):
        channel_wise_min = torch.min(pred_img, keepdim=True, dim=1)[0]
        neighbour_min = -soft_pool2d((-channel_wise_min),kernel_size=15)
        # return torch.mean(torch.exp(neighbour_min,2))
        return neighbour_min



def soft_pool2d(x, kernel_size=2, stride=None, force_inplace=False):
    kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)
    # Get input sizes
    _, c, h, w = x.size()
    # Create per-element exponential value sum : Tensor [b x 1 x h x w]
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x h x w] -> [b x c x h' x w']
    return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))


class Masked_DarkChannelLoss(nn.Module):
    def __init__(self):
        super(Masked_DarkChannelLoss, self).__init__()

    def forward(self, pred_img, mask):
        return torch.mean(torch.abs(self.get_dark_channel(pred_img*(1-mask))))


    def get_dark_channel(input, pred_img):
        channel_wise_min = torch.min(pred_img, keepdim=True, dim=1)[0]
        neighbour_min = -soft_pool2d((-channel_wise_min),kernel_size=15)
        # return torch.mean(torch.exp(neighbour_min,2))
        return neighbour_min

