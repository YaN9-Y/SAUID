import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from functools import reduce
from torch.autograd import Variable
from .blocks import FeatureFusionBlock_custom, Interpolate, _make_encoder, _make_scratch, _make_pretrained_efficientnet_lite3
import os
from .mobilenetv2 import mobilenet_v2
from .utils import set_bn_momentum, IntermediateLayerGetter, _SimpleSegmentationModel


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            # print(m.name)
            # if classname.find('pretrained') != -1:
            #     print(classname)
            #     return
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)



        self.apply(init_func)


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeeplabV3Plus():
    def __init__(self):
        self.num_classes = 19
        self.output_stride = 16
        self.backbone = mobilenet_v2(pretrained=True, output_stride=self.output_stride)

        if self.output_stride == 8:
            aspp_dilate = [12, 24, 36]
        else:
            aspp_dilate = [6, 12, 18]

        self.backbone.low_level_features = self.backbone.features[0:4]
        self.backbone.high_level_features = self.backbone.features[4:-1]
        self.backbone.features = None
        self.backbone.classifier = None

        inplanes = 320
        low_level_planes = 24

        self.return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        self.classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, self.num_classes, aspp_dilate)

        set_bn_momentum(self.backbone, momentum=0.01)

        self.backbone = IntermediateLayerGetter(self.backbone, return_layers=self.return_layers)


    def forward(self, x):
        return self.classifier(self.backbone(x))


def _segm_mobilenet(num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24


    return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()


    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        # if not require_immediate_features:
        #     return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
        # else:
        #     result = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
        #     feature = torch.cat([low_level_feature, output_feature], dim=1)  # C: 304
        #     return result, [feature]
        result = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
        feature = torch.cat([low_level_feature, output_feature], dim=1).detach()  # C: 304
        return result, [feature]

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class UNet(BaseNetwork):
    def __init__(self, base_channel_nums, in_channels=3, out_channels=3,norm_type='batch', init_weights=True):
        super(UNet, self).__init__()

        act_type = 'leakyrelu'
        norm_type = norm_type
        mode = 'CNA'
        use_spectral_norm = False

        self.enc_conv0 = conv_block(in_nc=in_channels, out_nc=base_channel_nums // 2, kernel_size=3, stride=1,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=None,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv1 = conv_block(in_nc=base_channel_nums // 2, out_nc=base_channel_nums, kernel_size=3, stride=1,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv2 = conv_block(in_nc=base_channel_nums, out_nc=2 * base_channel_nums, kernel_size=3, stride=2,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv3 = conv_block(in_nc=2 * base_channel_nums, out_nc=4 * base_channel_nums, kernel_size=3, stride=2,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)
        #
        self.bottleneck1 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3, pad_type='reflect',
                                       act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck2 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck3 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck4 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)

        # self.deconv1 = deconv_block(in_nc=base_channel_nums*4, out_nc=base_channel_nums*2, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')
        self.dec_up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv1 = conv_block(base_channel_nums * 8, out_nc=base_channel_nums * 2, kernel_size=3,
                                    act_type=act_type, norm_type=norm_type, pad_type='reflect', mode=mode,
                                    use_spectral_norm=use_spectral_norm)

        self.dec_up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv2 = conv_block(base_channel_nums * 4, out_nc=base_channel_nums * 1, kernel_size=3,
                                    act_type=act_type, norm_type=None, pad_type='reflect', mode=mode,
                                    use_spectral_norm=use_spectral_norm)

        # self.deconv2 = deconv_block(in_nc=base_channel_nums*2, out_nc=base_channel_nums*1, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')

        self.dec_conv_last = conv_block(base_channel_nums * 2, out_nc=base_channel_nums, kernel_size=3,
                                        pad_type='reflect',
                                        act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)

        self.dec_conv_last_2 = conv_block(base_channel_nums * 1, out_nc=out_channels, kernel_size=3,
                                          pad_type='reflect',
                                          act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)

        # self.add_fog_block = FogFusionBlock(base_channel_nums)

        # self.haze_color_array = (torch.tensor([[255,255,255],[245,245,245],[255,218,185],[218,165,32],[238,213,210],[193,205,205],[238,238,208],[174,238,238]]).float()/255.0).cuda()

        if init_weights:
            self.init_weights('xaiver')

    def forward(self, x, requires_t=False):
  
        hazy = x
        x0 = (x - 0.5) * 2
        x = self.enc_conv0(x0)
        # print(x)
        x1 = self.enc_conv1(x)
        # print(x1)
        x2 = self.enc_conv2(x1)
        # print(x2)
        x3 = self.enc_conv3(x2)
        # print(x3)
        # x = self.bottleneck1(x3, depth)
        # x = self.bottleneck2(x, depth)
        # x = self.bottleneck3(x, depth)
        # x = self.bottleneck4(x, depth)
        # x = x3
        x = self.bottleneck1(x3)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)

        x = self.dec_up1(torch.cat([x, x3], dim=1))
        x = self.dec_conv1(x)
        x = self.dec_up2(torch.cat([x, x2], dim=1))
        x = self.dec_conv2(x)
        x = self.dec_conv_last(torch.cat([x, x1], dim=1))

        # x = self.dec_up1(x)
        # x = self.dec_conv1(x)
        # x = self.dec_up2(x)
        # x = self.dec_conv2(x)
        # x = self.dec_conv_last(x)

        # x = self.add_fog_block(x,t,A)

        x = self.dec_conv_last_2(x)
        #x = torch.tanh(x)
        t = ((F.tanh(x)+1)/2).clamp(0.01,1)

        A = 0.9* torch.ones(hazy.shape[0],3,1,1,device='cuda').detach()
        
        x = ((hazy - A)/t + A).clamp(0,1)

        if requires_t:
            return x, t
        return x


class UNet_SemanticRemoval(BaseNetwork):
    def __init__(self, base_channel_nums, dataset, in_channels=3, out_channels=3, semantic_map_channels=40, norm_type='batch', init_weights=True):
        super(UNet_SemanticRemoval, self).__init__()

        act_type = 'leakyrelu'
        norm_type = norm_type
        mode = 'CNA'
        use_spectral_norm = False
        if str.lower(dataset) in ['nyu']:
            self.use_dc_A = False
        elif str.lower(dataset) in ['cityscapes']:
            self.use_dc_A = True
        self.transmission_estimator = TransmissionEstimator()

        self.enc_conv0 = conv_block(in_nc=in_channels, out_nc=base_channel_nums // 2, kernel_size=3, stride=1,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=None,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv1 = conv_block(in_nc=base_channel_nums // 2, out_nc=base_channel_nums, kernel_size=3, stride=1,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv2 = conv_block(in_nc=base_channel_nums, out_nc=2 * base_channel_nums, kernel_size=3, stride=2,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv3 = conv_block(in_nc=2 * base_channel_nums, out_nc=4 * base_channel_nums, kernel_size=3, stride=2,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)
        #
        self.bottleneck1 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3, pad_type='reflect',
                                       act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck2 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck3 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck4 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)

        # self.deconv1 = deconv_block(in_nc=base_channel_nums*4, out_nc=base_channel_nums*2, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')
        self.dec_up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv1 = conv_block(int(base_channel_nums * (8)), out_nc=base_channel_nums * 2, kernel_size=3,
                                    act_type=act_type, norm_type=norm_type, pad_type='reflect', mode=mode,
                                    use_spectral_norm=use_spectral_norm)

        self.dec_up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv2 = conv_block(base_channel_nums * (4), out_nc=base_channel_nums * 1, kernel_size=3,
                                    act_type=act_type, norm_type=None, pad_type='reflect', mode=mode,
                                    use_spectral_norm=use_spectral_norm)

        # self.deconv2 = deconv_block(in_nc=base_channel_nums*2, out_nc=base_channel_nums*1, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')

        self.dec_conv_last = conv_block(int(base_channel_nums * (2)), out_nc=base_channel_nums, kernel_size=3,
                                        pad_type='reflect',
                                        act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)

        self.dec_conv_last_2 = conv_block(base_channel_nums * 1, out_nc=out_channels, kernel_size=3,
                                          pad_type='reflect',
                                          act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)

        #self.fusion_seblock_1 = SEBottleneck()

        if str.lower(dataset) in ['nyu']:
            self.semantic_fusion_block_1 = SemanticBlock(origin_feature_channel=base_channel_nums*4,
                                                         semantic_map_channel=semantic_map_channels,
                                                         semantic_feature_channel=256)
            self.semantic_fusion_block_2 = SemanticBlock(origin_feature_channel=base_channel_nums * 2,
                                                         semantic_map_channel=semantic_map_channels,
                                                         semantic_feature_channel=256)
            self.semantic_fusion_block_3 = SemanticBlock(origin_feature_channel=base_channel_nums * 1,
                                                         semantic_map_channel=semantic_map_channels,
                                                         semantic_feature_channel=256)

        if str.lower(dataset) in ['cityscapes']:
            self.semantic_fusion_block_1 = SemanticBlock(origin_feature_channel=base_channel_nums*4,
                                                         semantic_map_channel=semantic_map_channels,
                                                         semantic_feature_channel=304)
            self.semantic_fusion_block_2 = SemanticBlock(origin_feature_channel=base_channel_nums * 2,
                                                         semantic_map_channel=semantic_map_channels,
                                                         semantic_feature_channel=304)
            self.semantic_fusion_block_3 = SemanticBlock(origin_feature_channel=base_channel_nums * 1,
                                                         semantic_map_channel=semantic_map_channels,
                                                         semantic_feature_channel=304)




        # self.semantic_block_0 = conv_block(semantic_feature_channels, out_nc=base_channel_nums//2, kernel_size=3,
        #                                   pad_type='reflect',
        #                                   act_type=act_type, norm_type='layer', use_spectral_norm=use_spectral_norm)
        #
        # self.semantic_block_1 = conv_block(base_channel_nums//2, out_nc=base_channel_nums, kernel_size=3, stride=2,
        #                                    pad_type='reflect',
        #                                    act_type=act_type, norm_type='layer', use_spectral_norm=use_spectral_norm)
        # self.semantic_block_2 = conv_block(base_channel_nums, out_nc=int(base_channel_nums*2), kernel_size=3,stride=2,
        #                                    pad_type='reflect',
        #                                    act_type=act_type, norm_type='layer', use_spectral_norm=use_spectral_norm)
        # self.se_1 = SELayer(channel=int(base_channel_nums * (4)+semantic_feature_channels), out_channel=base_channel_nums * (2))
        # self.se_2 = SELayer(channel=base_channel_nums * (1)+semantic_feature_channels, out_channel=base_channel_nums * (1))
        # self.se_3 = SELayer(channel=int(base_channel_nums * (1)+semantic_feature_channels), reduction=8, out_channel=int(base_channel_nums *(1)))
        #
        # self.pa_1 = PALayer(channel=int(base_channel_nums * (2))+semantic_feature_channels)
        # self.pa_2 = PALayer(channel=int(base_channel_nums * (1))+semantic_feature_channels)
        # self.pa_3 = PALayer(channel=int(base_channel_nums * (1)+semantic_feature_channels), reduction=8)




        # self.add_fog_block = FogFusionBlock(base_channel_nums)

        # self.haze_color_array = (torch.tensor([[255,255,255],[245,245,245],[255,218,185],[218,165,32],[238,213,210],[193,205,205],[238,238,208],[174,238,238]]).float()/255.0).cuda()

        if init_weights:
            self.init_weights('xaiver')

    def forward(self, x, semantic_maps, semantic_features, requires_t=False, use_GF=False):
        if len(semantic_features) > 1:
            semantic_features_1, semantic_features_2, semantic_features_3 = semantic_features[0], semantic_features[1],semantic_features[2]
        else:
            semantic_features_1, semantic_features_2, semantic_features_3 = semantic_features[0],semantic_features[0],semantic_features[0]

        hazy = x
        x0 = (x - 0.5) * 2
        x = self.enc_conv0(x0)
        # print(x)
        x1 = self.enc_conv1(x)
        # print(x1)
        x2 = self.enc_conv2(x1)
        # print(x2)
        x3 = self.enc_conv3(x2)
        # print(x3)
        # x = self.bottleneck1(x3, depth)
        # x = self.bottleneck2(x, depth)
        # x = self.bottleneck3(x, depth)
        # x = self.bottleneck4(x, depth)
        # x = x3
        x = self.bottleneck1(x3)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)


        # print(x.shape)
        # print(semantic_maps.shape)
        # print(semantic_features_1.shape)
        x = self.semantic_fusion_block_1(origin_feature=x, semantic_map=semantic_maps, semantic_feature=semantic_features_1)
        x = self.dec_up1(torch.cat([x, x3], dim=1))
        x = self.dec_conv1(x)
        # x_t = self.se_1(x, semantic_features_1)
        # x_t = self.pa_1(x_t, semantic_features_1)
        # x = x_t + x

        x = self.semantic_fusion_block_2(origin_feature=x, semantic_map=semantic_maps,
                                         semantic_feature=semantic_features_2)
        x = self.dec_up2(torch.cat([x, x2], dim=1))
        x = self.dec_conv2(x)
        # x_t = self.se_2(x,semantic_features_2)
        # x_t = self.pa_2(x_t,semantic_features_2)
        # x = x_t + x

        x = self.semantic_fusion_block_3(origin_feature=x, semantic_map=semantic_maps,
                                         semantic_feature=semantic_features_3)
        x  = self.dec_conv_last(torch.cat([x, x1], dim=1))
        # x_t = self.se_3(x,semantic_features_2)
        # x_t = self.pa_3(x_t,semantic_features_2)
        # x = x_t + x
        # x = self.dec_up1(x)
        # x = self.dec_conv1(x)
        # x = self.dec_up2(x)
        # x = self.dec_conv2(x)
        # x = self.dec_conv_last(x)

        # x = self.add_fog_block(x,t,A)

        x = self.dec_conv_last_2(x)
        # x = torch.tanh(x)
        t = ((F.tanh(x) + 1) / 2).clamp(0.05, 1)
        if use_GF:
            t = self.transmission_estimator.get_refined_transmission(I=hazy,rawt=t)

        #A = 0.9 * torch.ones(hazy.shape[0], 3, 1, 1, device='cuda').detach()
        if self.use_dc_A:
            A = self.transmission_estimator.get_atmosphere_light_new(hazy).detach()
        elif not self.use_dc_A:
            #A = hazy.max(dim=3)[0].max(dim=2,keepdim=True)[0].unsqueeze(3).detach()
            #A = A.clamp(0,220/255.0)
            A = 0.9*torch.ones(x.shape[0],3,1,1,device='cuda')

        x = ((hazy - A) / t + A).clamp(0, 1)

        if requires_t:
            return x, t
        return x


class SemanticBlock(nn.Module):
    def __init__(self, origin_feature_channel, semantic_map_channel, semantic_feature_channel,reduction=16):
        super(SemanticBlock, self).__init__()

        self.se_map = SELayer(channel=int(origin_feature_channel + semantic_map_channel), reduction=reduction,
                            out_channel=origin_feature_channel)

        self.pa_map = PALayer(channel=int(origin_feature_channel + semantic_map_channel), reduction=reduction)



        self.se_feature = SELayer(channel=int(origin_feature_channel + semantic_feature_channel), reduction=reduction,
                                  out_channel=origin_feature_channel)

        self.pa_feature = PALayer(channel=int(origin_feature_channel + semantic_feature_channel), reduction=reduction)


    def forward(self, origin_feature, semantic_map, semantic_feature):
        n, c, h, w = semantic_map.shape
        n_orig, c_orig, h_orig, w_orig = origin_feature.shape

        semantic_map = F.interpolate(semantic_map,size=[h_orig,w_orig],mode='nearest')
        semantic_feature = F.interpolate(semantic_feature,size=[h_orig,w_orig],mode='bilinear')

        f_map_begin = origin_feature
        f_map_path = self.se_map(x=f_map_begin,semantic_feature=(semantic_map))
        f_map_path = self.pa_map(x=f_map_path, semantic_feature=(semantic_map))
        f_map_path = f_map_begin + f_map_path

        f_feature_begin = origin_feature
        f_feature_path = self.se_feature(x=f_feature_begin,semantic_feature=(semantic_feature))
        f_feature_path = self.pa_feature(x=f_feature_path, semantic_feature=(semantic_feature))
        f_feature_path = f_feature_begin + f_feature_path


        semantic_entropy = self.get_semantic_entropy(semantic_map)
        # print(semantic_entropy.shape)
        semantic_entropy_min = torch.min(semantic_entropy.view(n,1,h_orig*w_orig),dim=2,keepdim=True)[0].unsqueeze(3)
        semantic_entropy_max = torch.max(semantic_entropy.view(n, 1, h_orig*w_orig), dim=2, keepdim=True)[0].unsqueeze(3)


        # larger entropy -> more uncertainty -> more weight for semantic feature
        semantic_weight_map_for_feature = (semantic_entropy - semantic_entropy_min) / (semantic_entropy_max-semantic_entropy_min+0.001)
        semantic_weight_map_for_map = 1 - semantic_weight_map_for_feature

        f_map_path = f_map_path * semantic_weight_map_for_map
        f_feature_path = f_feature_path * semantic_weight_map_for_feature

        return origin_feature + f_map_path + f_feature_path



    def get_semantic_entropy(self, semantic_maps):
        semantic_map_prop = torch.softmax(semantic_maps,dim=1)
        semantic_entropy = torch.sum(-torch.log(semantic_map_prop) * semantic_map_prop,dim=1,keepdim=True)
        return semantic_entropy




class UNet_HazeRefine(BaseNetwork):
    def __init__(self, base_channel_nums, in_channels=3, out_channels=3,norm_type='batch', init_weights=True):
        super(UNet_HazeRefine, self).__init__()

        act_type = 'leakyrelu'
        norm_type = norm_type
        mode = 'CNA'
        use_spectral_norm = False

        self.enc_conv0 = conv_block(in_nc=in_channels, out_nc=base_channel_nums // 2, kernel_size=3, stride=1,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=None,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv1 = conv_block(in_nc=base_channel_nums // 2, out_nc=base_channel_nums, kernel_size=3, stride=1,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv2 = conv_block(in_nc=base_channel_nums, out_nc=2 * base_channel_nums, kernel_size=3, stride=2,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv3 = conv_block(in_nc=2 * base_channel_nums, out_nc=4 * base_channel_nums, kernel_size=3, stride=2,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)
        #
        self.bottleneck1 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3, pad_type='reflect',
                                       act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck2 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck3 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck4 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)

        # self.deconv1 = deconv_block(in_nc=base_channel_nums*4, out_nc=base_channel_nums*2, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')
        self.dec_up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv1 = conv_block(base_channel_nums * 8, out_nc=base_channel_nums * 2, kernel_size=3,
                                    act_type=act_type, norm_type=norm_type, pad_type='reflect', mode=mode,
                                    use_spectral_norm=use_spectral_norm)

        self.dec_up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv2 = conv_block(base_channel_nums * 4, out_nc=base_channel_nums * 1, kernel_size=3,
                                    act_type=act_type, norm_type=None, pad_type='reflect', mode=mode,
                                    use_spectral_norm=use_spectral_norm)

        # self.deconv2 = deconv_block(in_nc=base_channel_nums*2, out_nc=base_channel_nums*1, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')

        self.dec_conv_last = conv_block(base_channel_nums * 2, out_nc=base_channel_nums, kernel_size=3,
                                        pad_type='reflect',
                                        act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)

        self.dec_conv_last_2 = conv_block(base_channel_nums * 1, out_nc=out_channels, kernel_size=3,
                                          pad_type='reflect',
                                          act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)

        # self.add_fog_block = FogFusionBlock(base_channel_nums)

        # self.haze_color_array = (torch.tensor([[255,255,255],[245,245,245],[255,218,185],[218,165,32],[238,213,210],[193,205,205],[238,238,208],[174,238,238]]).float()/255.0).cuda()

        if init_weights:
            self.init_weights('xaiver')

    def forward(self, x ):

        x0 = (x - 0.5) * 2
        x = self.enc_conv0(x0)
        # print(x)
        x1 = self.enc_conv1(x)
        # print(x1)
        x2 = self.enc_conv2(x1)
        # print(x2)
        x3 = self.enc_conv3(x2)
        # print(x3)
        # x = self.bottleneck1(x3, depth)
        # x = self.bottleneck2(x, depth)
        # x = self.bottleneck3(x, depth)
        # x = self.bottleneck4(x, depth)
        # x = x3
        x = self.bottleneck1(x3)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)

        x = self.dec_up1(torch.cat([x, x3], dim=1))
        x = self.dec_conv1(x)

        x = self.dec_up2(torch.cat([x, x2], dim=1))
        x = self.dec_conv2(x)
        x = self.dec_conv_last(torch.cat([x, x1], dim=1))

        # x = self.dec_up1(x)
        # x = self.dec_conv1(x)
        # x = self.dec_up2(x)
        # x = self.dec_conv2(x)
        # x = self.dec_conv_last(x)

        # x = self.add_fog_block(x,t,A)

        x = self.dec_conv_last_2(x)
        #x = torch.tanh(x)
        x = torch.tanh(x)
        x = ((x0+x).clamp(-1,1)+1)/2




        return x


class HazeRemovalNet(BaseNetwork):
    def __init__(self, base_channel_nums, init_weights=True, path=None, min_beta=0.04, max_beta=0.2):
        super(HazeRemovalNet, self).__init__()
        self.transmission_estimator = TransmissionEstimator()

        # norm_type = 'batch'
        # act_type = 'leakyrelu'
        # mode = 'CNA'
        use_spectral_norm = False

        self.MIN_BETA=min_beta
        self.MAX_BETA=max_beta


        #self.use_dc_A = True if use_dc_A == 1 else False

        backbone = "efficientnet_lite3"
        non_negative = True
        exportable = True
        align_corners = True
        blocks = {'expand': True}

        features = base_channel_nums

        use_pretrained = False #if os.path.exists(path) else True

        # self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        features1 = features
        features2 = features
        features3 = features
        features4 = features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1 = features
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8

        # def _make_encoder(backbone, features, use_pretrained, groups=1, expand=False, exportable=True):
        #     if backbone == "resnext101_wsl":
        #         pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
        #         scratch = _make_scratch([256, 512, 1024, 2048], features, groups=groups,
        #                                 expand=expand)  # efficientnet_lite3
        #     elif backbone == "efficientnet_lite3":
        #         pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable)
        #         scratch = _make_scratch([32, 48, 136, 384], features, groups=groups,
        #                                 expand=expand)  # efficientnet_lite3
        #     else:
        #         print(f"Backbone '{backbone}' not implemented")
        #         assert False
        #
        #     return pretrained, scratch

        # self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups,
        #                                               expand=self.expand, exportable=exportable)

        self.scratch = _make_scratch([32, 48, 136, 384], features, groups=self.groups,expand=self.expand)


        self.scratch.activation = nn.ReLU(False)

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False,
                                                            align_corners=align_corners)

        self.scratch.output_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=0, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=0),
            self.scratch.activation,
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True) if non_negative else nn.Identity(),
            # nn.Identity(),
        )

        # if path:
        #     self.load(path)

        # self.haze_color_array = (torch.tensor([[255,255,255],[245,245,245],[255,218,185],[218,165,32],[238,213,210],[193,205,205],[238,238,208],[174,238,238]]).float()/255.0).cuda()

        # if init_weights:
        #     self.init_weights('kaiming')

        # self.beta_pred_conv4 = conv_block(in_nc=7*base_channel_nums, out_nc=2*base_channel_nums, kernel_size=3, act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)


        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=[1,1])
        self.final_conv_beta_1 = conv_block(in_nc=32 +48 +136 +384, out_nc=2*base_channel_nums, kernel_size=1, act_type=None, norm_type=None, use_spectral_norm=use_spectral_norm)
        self.final_conv_beta_2 = conv_block(in_nc=2*base_channel_nums, out_nc=1, kernel_size=1, act_type=None, norm_type=None, use_spectral_norm=use_spectral_norm)


        if init_weights:
            self.init_weights('xaiver')

        self.pretrained =_make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable)



    def forward(self, x_0, require_paras=False):

        layer_1 = self.pretrained.layer1(x_0)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_beta = F.adaptive_avg_pool2d(layer_1, [1,1]).detach()
        layer_2_beta = F.adaptive_avg_pool2d(layer_2, [1, 1]).detach()
        layer_3_beta = F.adaptive_avg_pool2d(layer_3, [1, 1]).detach()
        layer_4_beta = F.adaptive_avg_pool2d(layer_4, [1, 1]).detach()

        beta = self.final_conv_beta_1(torch.cat([layer_1_beta, layer_2_beta, layer_3_beta, layer_4_beta], dim=1))
        beta = self.final_conv_beta_2(beta)
        # beta = (torch.tanh(beta)+1)/2

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # print(x_0.shape)
        # print(layer_1.shape)
        # print(layer_2.shape)
        # print(layer_3.shape)
        # print(layer_4.shape)
        #
        # print('----------------')
        # print(layer_1_rn.shape)
        # print(layer_2_rn.shape)
        # print(layer_3_rn.shape)
        # print(layer_4_rn.shape)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        # print(path_4.shape)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)



        res = self.scratch.output_conv(path_1)

        beta = self.MIN_BETA + (self.MAX_BETA-self.MIN_BETA)*(torch.tanh(beta) + 1) / 2
        # d = self.MIN_D + ((torch.tanh(d) + 1) / 2) * (self.MAX_D - self.MIN_D)

        res = ((torch.tanh(res) + 1) / 2)

        # res = torch.tanh(res)
        # res = (x_0+res).clamp(0,1)

        #print(d)

        if require_paras:
            return res, beta
        else:
            return res


        # print(out)
        #return d




        # x = self.enc_conv0(x_0)
        # x1 = self.enc_conv1(x)
        # x2 = self.enc_conv2(x1)
        # x3 = self.enc_conv3(x2)
        #
        # x1_d = F.interpolate(x1,scale_factor=0.25, mode='bilinear')
        # x2_d = F.interpolate(x2,scale_factor=0.5, mode='bilinear')
        # x_b = self.beta_pred_conv4(torch.cat([x1_d,x2_d,x3],dim=1))
        # x_b = self.avg_pool(x_b)
        # x_b = self.final_conv(x_b)
        # x_b = (torch.tanh(x_b) + 1) / 2
        # beta = x_b * (self.MAX_BETA - self.MIN_BETA) + self.MIN_BETA
        #
        #
        #
        # x = self.bottleneck1(x3)
        # x = self.bottleneck2(x)
        # x = self.bottleneck3(x)
        # x = self.bottleneck4(x)
        # # x = x3
        # x = self.dec_up1(torch.cat([x, x3], dim=1))
        # x = self.dec_conv1(x)
        # x = self.dec_up2(torch.cat([x, x2], dim=1))
        # x = self.dec_conv2(x)
        # x = self.dec_conv_last(torch.cat([x, x1], dim=1))
        # x = self.dec_conv_last_2(x)
        #
        # # x = self.dec_up1(x)
        # # x = self.dec_conv1(x)
        # # x = self.dec_up2(x)
        # # x = self.dec_conv2(x)
        # # x = self.dec_conv_last(x)
        # # x = self.dec_conv_last_2(x)
        # # x = self.dec_conv_last_3(x)
        # x = (torch.tanh(x) + 1) / 2
        # # return x
        #
        # # x = self.dec_up1(x)
        # # x = self.dec_conv1(x)
        # # x = self.dec_up2(x)
        # # x = self.dec_conv2(x)
        # # x = self.dec_conv_last(x)
        # d = self.MIN_D + x*(self.MAX_D-self.MIN_D)
        # #d = x.clamp(min=0.3, max=1000)
        #
        # A = self.forward_get_A(x_0)
        # # beta = self.forward_get_beta(x_0)
        # # print(beta)
        # t = torch.exp(-d*beta)
        # t = t.clamp(0.1,0.95)
        # # print(t)
        #
        # if require_paras:
        #     return ((x_0-A)/t + A).clamp(0,1), d, beta
        # else:
        #     return ((x_0-A)/t + A).clamp(0,1)


    # def forward_get_beta(self, x):  # output: N,1,1,1
    #     x = self.enc_conv0(x)
    #     x1 = self.enc_conv1(x)
    #     x2 = self.enc_conv2(x1)
    #     x3 = self.enc_conv3(x2)
    #
    #
    #     x = self.beta_pred_conv4(x)
    #     x = self.avg_pool(x)
    #     x = self.final_conv(x)
    #     x = (torch.tanh(x)+1)/2
    #     x = x*(self.MAX_BETA-self.MIN_BETA)+self.MIN_BETA
    #
    #     return x



    #def forward_get_A(self, x): # output A: N,3,1,1
    #    return x.max(dim=3)[0].max(dim=2,keepdim=True)[0].unsqueeze(3)

    # def forward_check(self, x):



class HazeRemovalNet2(BaseNetwork):
    def __init__(self, base_channel_nums, in_channels=3, out_channels=3, init_weights=True, norm_type='batch', min_beta=0.04, max_beta=0.2):
        super(HazeRemovalNet2, self).__init__()

        act_type = 'leakyrelu'
        norm_type = norm_type
        mode = 'CNA'
        use_spectral_norm = False

        self.MIN_BETA=min_beta
        self.MAX_BETA=max_beta


        self.enc_conv0 = conv_block(in_nc=in_channels, out_nc=base_channel_nums // 2, kernel_size=3, stride=1,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=None,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv1 = conv_block(in_nc=base_channel_nums // 2, out_nc=base_channel_nums, kernel_size=3, stride=1,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv2 = conv_block(in_nc=base_channel_nums, out_nc=2 * base_channel_nums, kernel_size=3, stride=2,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv3 = conv_block(in_nc=2 * base_channel_nums, out_nc=4 * base_channel_nums, kernel_size=3, stride=2,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)
        #
        self.bottleneck1 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3, pad_type='reflect',
                                       act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck2 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck3 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck4 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)

        # self.deconv1 = deconv_block(in_nc=base_channel_nums*4, out_nc=base_channel_nums*2, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')
        self.dec_up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv1 = conv_block(base_channel_nums * 8, out_nc=base_channel_nums * 2, kernel_size=3,
                                    act_type=act_type, norm_type=norm_type, pad_type='reflect', mode=mode,
                                    use_spectral_norm=use_spectral_norm)

        self.dec_up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv2 = conv_block(base_channel_nums * 4, out_nc=base_channel_nums * 1, kernel_size=3,
                                    act_type=act_type, norm_type=None, pad_type='reflect', mode=mode,
                                    use_spectral_norm=use_spectral_norm)

        # self.deconv2 = deconv_block(in_nc=base_channel_nums*2, out_nc=base_channel_nums*1, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')

        self.dec_conv_last = conv_block(base_channel_nums * 2, out_nc=base_channel_nums, kernel_size=3,
                                        pad_type='reflect',
                                        act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)

        self.dec_conv_last_2 = conv_block(base_channel_nums * 1, out_nc=out_channels, kernel_size=3,
                                          pad_type='reflect',
                                          act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)

        self.final_conv_beta_1 = conv_block(in_nc= int(base_channel_nums * (1+2+4)), out_nc=2 * base_channel_nums, kernel_size=1,
                                            act_type=None, norm_type=None, use_spectral_norm=use_spectral_norm)
        self.final_conv_beta_2 = conv_block(in_nc=2 * base_channel_nums, out_nc=1, kernel_size=1, act_type=None,
                                            norm_type=None, use_spectral_norm=use_spectral_norm)

        # self.add_fog_block = FogFusionBlock(base_channel_nums)

        # self.haze_color_array = (torch.tensor([[255,255,255],[245,245,245],[255,218,185],[218,165,32],[238,213,210],[193,205,205],[238,238,208],[174,238,238]]).float()/255.0).cuda()

        if init_weights:
            self.init_weights('xaiver')

    def forward(self, x , require_paras=False):

        #x0 = (x - 0.5) * 2
        x0 =x
        x = self.enc_conv0(x)
        # print(x)
        x1 = self.enc_conv1(x)
        # print(x1)
        x2 = self.enc_conv2(x1)
        # print(x2)
        x3 = self.enc_conv3(x2)
        # print(x3)
        # x = self.bottleneck1(x3, depth)
        # x = self.bottleneck2(x, depth)
        # x = self.bottleneck3(x, depth)
        # x = self.bottleneck4(x, depth)
        # x = x3
        x = self.bottleneck1(x3)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)

        x = self.dec_up1(torch.cat([x, x3], dim=1))
        x = self.dec_conv1(x)
        x = self.dec_up2(torch.cat([x, x2], dim=1))
        x = self.dec_conv2(x)
        x = self.dec_conv_last(torch.cat([x, x1], dim=1))

        # x = self.dec_up1(x)
        # x = self.dec_conv1(x)
        # x = self.dec_up2(x)
        # x = self.dec_conv2(x)
        # x = self.dec_conv_last(x)

        # x = self.add_fog_block(x,t,A)

        x = self.dec_conv_last_2(x)
        #x = torch.tanh(x)
        #x = (F.tanh(x)+1)/2
        #x = ((x0+x).clamp(-1,1)+1)/2
        x = (x0+x).clamp(0,1)

        layer_1_beta = F.adaptive_avg_pool2d(x1, [1, 1]).detach()
        layer_2_beta = F.adaptive_avg_pool2d(x2, [1, 1]).detach()
        layer_3_beta = F.adaptive_avg_pool2d(x3, [1, 1]).detach()
        beta = self.final_conv_beta_1(torch.cat([layer_1_beta, layer_2_beta, layer_3_beta], dim=1))
        beta = self.final_conv_beta_2(beta)
        beta = self.MIN_BETA + (self.MAX_BETA-self.MIN_BETA)*(torch.tanh(beta) + 1) / 2

        if require_paras:
            return x, beta
        else:
            return x






class HazeProduceNet(BaseNetwork):
    def __init__(self, base_channel_nums, in_channels=6, out_channels=3, init_weights=True, norm_type='batch', min_beta=0.04, max_beta=0.2):
        super(HazeProduceNet, self).__init__()

        act_type = 'leakyrelu'
        norm_type = norm_type
        mode = 'CNA'
        use_spectral_norm = False
        self.MAX_BETA = max_beta
        self.MIN_BETA = min_beta
        self.transmission_estimator = TransmissionEstimator()

        self.enc_conv0 = conv_block(in_nc=in_channels, out_nc=base_channel_nums // 2, kernel_size=3, stride=1,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)

        self.enc_conv1 = conv_block(in_nc=base_channel_nums//2, out_nc=base_channel_nums, kernel_size=3, stride=1,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type, use_spectral_norm=use_spectral_norm)

        self.enc_conv2 = conv_block(in_nc=base_channel_nums, out_nc=2 * base_channel_nums, kernel_size=3, stride=2,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type, use_spectral_norm=use_spectral_norm)

        self.enc_conv3 = conv_block(in_nc=2 * base_channel_nums, out_nc=4 * base_channel_nums, kernel_size=3, stride=2,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type, use_spectral_norm=use_spectral_norm)
        #
        self.bottleneck1 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3, pad_type='reflect',
                                       act_type=act_type, norm_type=norm_type, mode=mode, use_spectral_norm=use_spectral_norm)
        self.bottleneck2 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode, use_spectral_norm=use_spectral_norm)
        self.bottleneck3 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode, use_spectral_norm=use_spectral_norm)
        self.bottleneck4 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode, use_spectral_norm=use_spectral_norm)


        # self.deconv1 = deconv_block(in_nc=base_channel_nums*4, out_nc=base_channel_nums*2, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')
        self.dec_up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv1 = conv_block(base_channel_nums * 8, out_nc=base_channel_nums * 2, kernel_size=3, act_type=act_type, norm_type=norm_type, pad_type='reflect', mode=mode, use_spectral_norm=use_spectral_norm)

        self.dec_up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv2 = conv_block(base_channel_nums * 4, out_nc=base_channel_nums * 1, kernel_size=3, act_type=act_type, norm_type=None, pad_type='reflect', mode=mode, use_spectral_norm=use_spectral_norm)

        # self.deconv2 = deconv_block(in_nc=base_channel_nums*2, out_nc=base_channel_nums*1, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')

        self.dec_conv_last = conv_block(base_channel_nums * 2, out_nc=base_channel_nums, kernel_size=3, pad_type='reflect',
                                        act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)

        self.dec_conv_last_2 = conv_block(base_channel_nums * 1, out_nc=out_channels, kernel_size=3,
                                          pad_type='reflect',
                                          act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)


        # self.add_fog_block = FogFusionBlock(base_channel_nums)

        # self.haze_color_array = (torch.tensor([[255,255,255],[245,245,245],[255,218,185],[218,165,32],[238,213,210],[193,205,205],[238,238,208],[174,238,238]]).float()/255.0).cuda()


        if init_weights:
            self.init_weights('xaiver')

    def forward(self, x, d, beta, A=None, requires_direct_fog=False):
        if A is None:
            #A = x.max(dim=3)[0].max(dim=2,keepdim=True)[0].unsqueeze(3)
            A = self.transmission_estimator.get_atmosphere_light_new(x)
        t = torch.exp(-d*beta)
        t = t.clamp(0.1,0.95)

        
        x_clean = t*x+A*(1-t).clamp(0,1)
        #x0 = (x_clean-0.5)*2
        x0 = x
        x = self.enc_conv0(torch.cat([x_clean,x_clean],dim=1))
        # print(x)
        x1 = self.enc_conv1(x)
        # print(x1)
        x2 = self.enc_conv2(x1)
        # print(x2)
        x3 = self.enc_conv3(x2)
        # print(x3)
        # x = self.bottleneck1(x3, depth)
        # x = self.bottleneck2(x, depth)
        # x = self.bottleneck3(x, depth)
        # x = self.bottleneck4(x, depth)
        # x = x3
        x = self.bottleneck1(x3)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)

        x = self.dec_up1(torch.cat([x, x3], dim=1))
        x = self.dec_conv1(x)
        x = self.dec_up2(torch.cat([x, x2], dim=1))
        x = self.dec_conv2(x)
        x = self.dec_conv_last(torch.cat([x, x1], dim=1))

        # x = self.dec_up1(x)
        # x = self.dec_conv1(x)
        # x = self.dec_up2(x)
        # x = self.dec_conv2(x)
        # x = self.dec_conv_last(x)

        # x = self.add_fog_block(x,t,A)

        x = self.dec_conv_last_2(x)
        #x = torch.tanh(x)
        # x = (F.tanh(x)+1)/2
        #x = ((x0+x).clamp(-1,1)+1)/2
        x = (x_clean + x).clamp(0, 1)
        if requires_direct_fog:
            return x, x_clean
        else:
            return x

    def forward_random_parameters(self, x, d, requires_direct_fog=False): #x:NCHW, ex:e^(-d(x)), N,1,H,W, beta:N,1,1,1
        n,c,h,w = x.shape
        beta = self.MIN_BETA + torch.rand(n, 1, 1, 1).cuda() * (self.MAX_BETA - self.MIN_BETA)

        if requires_direct_fog:
            res, res_redirect = self(x, d, beta, requires_direct_fog=requires_direct_fog)
            return res, res_redirect, beta
        else:
            res = self(x, d, beta, requires_direct_fog=requires_direct_fog)
            return res, beta


    def direct_forward(self, x):
        x_clean = x
        x = self.enc_conv0(torch.cat([x_clean, x_clean], dim=1))
        # print(x)
        x1 = self.enc_conv1(x)
        # print(x1)
        x2 = self.enc_conv2(x1)
        # print(x2)
        x3 = self.enc_conv3(x2)
        # print(x3)
        # x = self.bottleneck1(x3, depth)
        # x = self.bottleneck2(x, depth)
        # x = self.bottleneck3(x, depth)
        # x = self.bottleneck4(x, depth)
        # x = x3
        x = self.bottleneck1(x3)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)

        x = self.dec_up1(torch.cat([x, x3], dim=1))
        x = self.dec_conv1(x)
        x = self.dec_up2(torch.cat([x, x2], dim=1))
        x = self.dec_conv2(x)
        x = self.dec_conv_last(torch.cat([x, x1], dim=1))

        # x = self.dec_up1(x)
        # x = self.dec_conv1(x)
        # x = self.dec_up2(x)
        # x = self.dec_conv2(x)
        # x = self.dec_conv_last(x)

        # x = self.add_fog_block(x,t,A)

        x = self.dec_conv_last_2(x)
        # x = torch.tanh(x)
        # x = (F.tanh(x)+1)/2
        # x = ((x0+x).clamp(-1,1)+1)/2
        x = (x_clean + x).clamp(0, 1)

        return x

# class FogFusionBlock(nn.Module):
#     def __init__(self, base_channel_num, use_spectral_norm=False):
#         super(FogFusionBlock, self).__init__()
#
#         self.fog_block1 = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(kernel_size=3, in_channels=3, out_channels=base_channel_num//2), mode=use_spectral_norm),
#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(kernel_size=3, in_channels=base_channel_num//2, out_channels=base_channel_num),mode=use_spectral_norm)
#         )
#
#         self.fog_block2 = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(kernel_size=3, in_channels=base_channel_num, out_channels=base_channel_num),mode=use_spectral_norm),
#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(kernel_size=3, in_channels=base_channel_num, out_channels=base_channel_num),
#                           mode=use_spectral_norm),
#         )
#
#         self.fog_block3 = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(kernel_size=3, in_channels=base_channel_num, out_channels=base_channel_num),
#                           mode=use_spectral_norm),
#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(kernel_size=3, in_channels=base_channel_num, out_channels=base_channel_num),
#                           mode=use_spectral_norm),
#         )
#
#         self.scene_block1 = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(kernel_size=3, in_channels=1, out_channels=base_channel_num//2), mode=use_spectral_norm),
#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(kernel_size=3, in_channels=base_channel_num//2, out_channels=base_channel_num),mode=use_spectral_norm)
#         )
#
#         self.scene_block2 = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(kernel_size=3, in_channels=base_channel_num, out_channels=base_channel_num),
#                           mode=use_spectral_norm),
#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(kernel_size=3, in_channels=base_channel_num, out_channels=base_channel_num),
#                           mode=use_spectral_norm),
#         )
#
#         self.scene_block3 = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(kernel_size=3, in_channels=base_channel_num, out_channels=base_channel_num),
#                           mode=use_spectral_norm),
#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(kernel_size=3, in_channels=base_channel_num, out_channels=base_channel_num),
#                           mode=use_spectral_norm),
#         )
#
#     def forward(self, J,t,A): # J:scene_features, N,C,H,W,   t:N,1,H,W, A:N,3,1,1
#         scene_input = t
#         scene_input = self.scene_block1(scene_input)
#         scene_mul = self.scene_block2(scene_input)
#         scene_bias = self.scene_block3(scene_input)
#         scene_output = J*scene_mul + scene_bias
#
#         fog_input = A*(1-t)
#         fog_input = self.fog_block1(fog_input)
#         fog_bias = self.fog_block3(fog_input)
#         fog_mul = self.fog_block2(fog_input)
#         fogged_scene = (scene_output)*fog_mul + fog_bias
#
#         return fogged_scene

#
# class DepthEstimationNet(BaseNetwork):
#     def __init__(self, base_channel_nums,  min_d=0.3, max_d=5, path=None):
#         super(DepthEstimationNet, self).__init__()
#
#         act_type = 'leakyrelu'
#         norm_type = 'batch'
#         mode = 'CNA'
#         use_spectral_norm = False
#         init_weights = True
#
#         self.MIN_D = min_d
#         self.MAX_D = max_d
#
#         self.enc_conv0 = conv_block(in_nc=3, out_nc=base_channel_nums // 2, kernel_size=3, stride=1,
#                                     pad_type='reflect', mode=mode, act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)
#
#         self.enc_conv1 = conv_block(in_nc=base_channel_nums // 2, out_nc=base_channel_nums, kernel_size=3, stride=1,
#                                     pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type, use_spectral_norm=use_spectral_norm)
#
#         self.enc_conv2 = conv_block(in_nc=base_channel_nums, out_nc=2 * base_channel_nums, kernel_size=3, stride=2,
#                                     pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type, use_spectral_norm=use_spectral_norm)
#
#         self.enc_conv3 = conv_block(in_nc=2 * base_channel_nums, out_nc=4 * base_channel_nums, kernel_size=3, stride=2,
#                                     pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type, use_spectral_norm=use_spectral_norm)
#         #
#         self.bottleneck1 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
#                                        out_nc=4 * base_channel_nums, kernel_size=3, pad_type='reflect',
#                                        act_type=act_type, norm_type=norm_type, mode=mode)
#         self.bottleneck2 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
#                                        out_nc=4 * base_channel_nums, kernel_size=3,
#                                        pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode)
#         self.bottleneck3 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
#                                        out_nc=4 * base_channel_nums, kernel_size=3,
#                                        pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode)
#         self.bottleneck4 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
#                                        out_nc=4 * base_channel_nums, kernel_size=3,
#                                        pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode)
#
#         # self.deconv1 = deconv_block(in_nc=base_channel_nums*4, out_nc=base_channel_nums*2, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')
#         self.dec_up1 = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.dec_conv1 = conv_block(base_channel_nums * 8, out_nc=base_channel_nums * 2, kernel_size=3,
#                                     act_type=act_type, norm_type=norm_type, pad_type='reflect', mode=mode, use_spectral_norm=use_spectral_norm)
#
#         self.dec_up2 = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.dec_conv2 = conv_block(base_channel_nums * 4, out_nc=base_channel_nums * 1, kernel_size=3,
#                                     act_type=act_type, norm_type=None, pad_type='reflect', mode=mode, use_spectral_norm=use_spectral_norm)
#
#         # self.deconv2 = deconv_block(in_nc=base_channel_nums*2, out_nc=base_channel_nums*1, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')
#
#         self.dec_conv_last = conv_block(base_channel_nums * 2, out_nc=base_channel_nums, kernel_size=3,
#                                         pad_type='reflect',
#                                         act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)
#
#         self.dec_conv_last_2 = conv_block(base_channel_nums * 1, out_nc=1, kernel_size=3,
#                                           pad_type='reflect',
#                                           act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)
#
#
#         # self.haze_color_array = (torch.tensor([[255,255,255],[245,245,245],[255,218,185],[218,165,32],[238,213,210],[193,205,205],[238,238,208],[174,238,238]]).float()/255.0).cuda()
#
#         if init_weights:
#             self.init_weights('kaiming')
#
#
#     def forward(self, x):
#         x = self.enc_conv0(x)
#         # print(x)
#         x1 = self.enc_conv1(x)
#         # print(x1)
#         x2 = self.enc_conv2(x1)
#         # print(x2)
#         x3 = self.enc_conv3(x2)
#
#         x = self.bottleneck1(x3)
#         x = self.bottleneck2(x)
#         x = self.bottleneck3(x)
#         x = self.bottleneck4(x)
#
#         x = self.dec_up1(torch.cat([x, x3], dim=1))
#         x = self.dec_conv1(x)
#         x = self.dec_up2(torch.cat([x, x2], dim=1))
#         x = self.dec_conv2(x)
#         x = self.dec_conv_last(torch.cat([x, x1], dim=1))
#
#         # x = self.dec_up1(x)
#         # x = self.dec_conv1(x)
#         # x = self.dec_up2(x)
#         # x = self.dec_conv2(x)
#         # x = self.dec_conv_last(x)
#         x = self.dec_conv_last_2(x)
#
#         x = (F.tanh(x) + 1) / 2
#         # x = x.clamp(0.05,0.95)
#         d = self.MIN_D + x * (self.MAX_D-self.MIN_D)
#         return d

class DepthEstimationNet(BaseNetwork):
    def __init__(self, base_channel_nums=48, min_d=0.3, max_d=10, path=None, init_weights=True):
        super(DepthEstimationNet, self).__init__()
        self.transmission_estimator = TransmissionEstimator()

        self.MIN_D = min_d
        self.MAX_D = max_d

        backbone = "efficientnet_lite3"
        non_negative = True
        exportable = True
        align_corners = True
        blocks = {'expand': True}

        features = base_channel_nums

        use_pretrained = False if os.path.exists(path) else True

        # self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        features1 = features
        features2 = features
        features3 = features
        features4 = features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1 = features
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8

        # self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups,
        #                                               expand=self.expand, exportable=exportable)

        self.scratch = _make_scratch([32, 48, 136, 384], features, groups=self.groups, expand=self.expand)

        self.scratch.activation = nn.ReLU(False)

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False,
                                                            align_corners=align_corners)

        self.scratch.output_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=0, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=0),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True) if non_negative else nn.Identity(),
            # nn.Identity(),
        )

        # if path:
        #     self.load(path)

        # self.haze_color_array = (torch.tensor([[255,255,255],[245,245,245],[255,218,185],[218,165,32],[238,213,210],[193,205,205],[238,238,208],[174,238,238]]).float()/255.0).cuda()

        if init_weights:
            self.init_weights('xaiver')

        self.pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable)


    def forward(self, x):
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # print(x.shape)
        # print(layer_1.shape)
        # print(layer_2.shape)
        # print(layer_3.shape)
        # print(layer_4.shape)
        #
        # print('----------------')
        # print(layer_1_rn.shape)
        # print(layer_2_rn.shape)
        # print(layer_3_rn.shape)
        # print(layer_4_rn.shape)


        path_4 = self.scratch.refinenet4(layer_4_rn)
        # print(path_4.shape)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)



        out = self.scratch.output_conv(path_1)

        out = ((torch.tanh(out)+1)/2)

        # out = self.transmission_estimator.get_refined_transmission(x, out)
        out = self.MIN_D + out * (self.MAX_D-self.MIN_D)

        # out = ((torch.tanh(out) + 1) / 2)
        # print(out)
        return out


class MidasNet_small(BaseNetwork):
    """Network for monocular depth estimation.
    """
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters, strict=False)


    def __init__(self, path=None, features=64, backbone="efficientnet_lite3", non_negative=True, exportable=True, align_corners=True,
                 blocks={'expand': True}):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet_small, self).__init__()

        use_pretrained = True if path else False

        # self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        features1 = features
        features2 = features
        features3 = features
        features4 = features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1 = features
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8

        self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups,
                                                      expand=self.expand, exportable=exportable)

        self.scratch.activation = nn.ReLU(False)

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False,
                                                            align_corners=align_corners)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),

        )

        self.load(path)

        self._mean = torch.tensor([0.485, 0.456, 0.406],device='cuda').reshape(3,1,1).unsqueeze(0).expand(1,3,1,1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device='cuda').reshape(3, 1, 1).unsqueeze(0).expand(1, 3, 1, 1)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        # if self.channels_last == True:
        #     print("self.channels_last = ", self.channels_last)
        #     x.contiguous(memory_format=torch.channels_last)
        x = (x-self._mean)/(self._std)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)


        return out




def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA', use_spectral_norm=False):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC', 'CAN'], 'Wrong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = spectral_norm(nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups), mode=use_spectral_norm)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

    elif mode =='CAN':
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, a, n)



def deconv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, padding=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)

    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


def pixelshuffle_block(in_nc, out_nc, bias=True, norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)

    c = nn.Conv2d(in_nc, 2*in_nc, kernel_size=3, stride=1, padding=1, bias=bias)
    ps = nn.PixelShuffle(upscale_factor=2)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(c, ps, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, ps,c)


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x

def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def norm(norm_type, nc=None):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    elif norm_type == 'layer':
        layer = LayerNorm(affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


class LayerNorm(nn.Module):
    def __init__(self, input_size=None, return_stats=False, affine=True, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.return_stats = return_stats
        self.input_size = input_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x



def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1, use_spectral_norm=False):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode,use_spectral_norm=use_spectral_norm)
        if mode == 'CNA' or mode == 'CAN':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode, use_spectral_norm=use_spectral_norm)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class ContextBlock(nn.Module):
    def __init__(self, input_channel=32, output_channel=32, square=False):
        super().__init__()
        self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        if square:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 8, 8)
        else:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 3, 3)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
        self.fusion = nn.Conv2d(4*output_channel, input_channel, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x_reduce = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x_reduce))
        conv2 = self.lrelu(self.conv2(x_reduce))
        conv3 = self.lrelu(self.conv3(x_reduce))
        conv4 = self.lrelu(self.conv4(x_reduce))
        out = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.fusion(out) + x
        return out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights('xaiver')

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class LocalDiscriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(LocalDiscriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )



        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        outputs = conv4
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv4)

        return torch.mean(outputs), [conv1, conv2, conv3, conv4]

class TransmissionEstimator(nn.Module):
    def __init__(self, width=5,):
        super(TransmissionEstimator, self).__init__()
        self.width = width
        self.t_min = 0.2
        self.alpha = 2.5
        self.A_max = 220.0/255
        self.omega=0.95
        self.p = 0.001
        self.max_pool = nn.MaxPool2d(kernel_size=width,stride=1)
        self.max_pool_with_index = nn.MaxPool2d(kernel_size=width, return_indices=True)
        self.guided_filter = GuidedFilter(r=15,eps=1e-3)


    def get_dark_channel(self, x):
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = F.pad(x, (self.width//2, self.width//2,self.width//2, self.width//2), mode='constant', value=1)
        x = -(self.max_pool(-x))
        return x

    def get_atmosphere_light(self,I,dc):
        n,c,h,w = I.shape
        flat_I = I.view(n,c,-1)
        flat_dc = dc.view(n,1,-1)
        searchidx = torch.argsort(flat_dc, dim=2, descending=True)[:,:,:int(h*w*self.p)]
        searchidx = searchidx.expand(-1,3,-1)
        searched = torch.gather(flat_I,dim=2, index=searchidx)
        return torch.max(searched, dim=2 ,keepdim=True)[0].unsqueeze(3)

    def get_transmission(self, I, A):
        return 1-self.omega* self.get_dark_channel(I/A)

    def get_refined_transmission(self, I, rawt):
        I_max = torch.max(I.contiguous().view(I.shape[0],-1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        I_min = torch.min(I.contiguous().view(I.shape[0],-1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        normI = (I - I_min)/(I_max-I_min+0.01)
        refinedT = self.guided_filter(normI, rawt)

        # n,c,h,w = refinedT.shape
        # T_max = torch.max(refinedT.view(n,c,-1), dim=2, keepdim=True)[0].unsqueeze(3)
        # T_min = torch.min(refinedT.view(n,c,-1), dim=2, keepdim=True)[0].unsqueeze(3)
        # refinedT = (refinedT-T_min)/(T_max-T_min)

        # print('-------------------')
        # print(torch.max(refinedT))
        # print(torch.min(refinedT))
        # print(refinedT.shape)
        return refinedT

    def get_radiance(self,I, A, t):
        return (I-A)/t + A



    def get_depth(self, I):
        I_dark = self.get_dark_channel(I)

        A = self.get_atmosphere_light(I, I_dark)
        A[A>self.A_max] = self.A_max
        rawT = self.get_transmission(I, A)

        # print(I)

        refinedT = self.get_refined_transmission(I, rawT)
        return refinedT
        
    def direct_get_transmission(self,I):
        I_dark = self.get_dark_channel(I)

        A = self.get_atmosphere_light(I, I_dark).clamp(0.01,1)
        A[A>self.A_max] = self.A_max
        rawT = self.get_transmission(I, A)

        # print(I)

        refinedT = self.get_refined_transmission(I, rawT)
        return refinedT

    def get_atmosphere_light_new(self, I):
        I_dark = self.get_dark_channel(I)
        A = self.get_atmosphere_light(I, I_dark)
        #A[A > self.A_max] = self.A_max
        return A



class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        #assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return torch.mean(mean_A * x + mean_b, dim=1 ,keepdim=True)

def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


#
# class HazeRender(nn.Module):
#     def __init__(self):
#
#     def blur_with_depth(self, hazy_img, depth):
#
#         hazy_img_pieces = F.unfold(hazy_img, kernel_size=3, dilation=1, padding=1, stride=1)
#         depth = F.unfold
#


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, sigma=3):
        super(GaussianFilter, self).__init__()

        self.gaussian_kernel = self.cal_kernel(kernel_size=kernel_size, sigma=sigma).expand(1,1,-1,-1).cuda()


    def apply_gaussian_filter(self, x):
        # cal gaussian filter of N C H W in cuda
        n,c,h,w = x.shape
        gaussian = torch.nn.functional.conv2d(x,self.gaussian_kernel.expand(c,1,-1,-1),padding=self.gaussian_kernel.shape[2]//2, groups=c)

        return gaussian

    def cal_gaussian_kernel_at_ij(self, i, j, sigma):
        return (1. / (2 * math.pi * pow(sigma, 2))) * math.exp(-(pow(i, 2) + pow(j, 2)) / (2 * pow(sigma, 2)))

    def cal_kernel(self, kernel_size=3, sigma=1.):
        kernel = torch.ones((kernel_size, kernel_size)).float()
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = self.cal_gaussian_kernel_at_ij(-(kernel_size // 2) + j, (kernel_size // 2) - i, sigma=sigma)

        kernel = kernel / torch.sum(kernel)
        return kernel



class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, out_channel=None):
        super(SELayer, self).__init__()
        if out_channel is None:
            out_channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            #nn.Linear(channel, channel // reduction, bias=False),
            nn.Conv2d(channel, channel//reduction, bias=False, padding=0, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, out_channel, bias=False, padding=0, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, semantic_feature):
        b, c, _, _ = x.size()
        y = self.avg_pool(torch.cat([x,semantic_feature],dim=1))
        y = self.fc(y)
        return x * y


class PALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, semantic_feature):
        y = self.pa(torch.cat([x,semantic_feature],dim=1))
        return x*y

