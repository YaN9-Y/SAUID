import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .networks import UNet_HazeRefine, Discriminator, _segm_mobilenet, TransmissionEstimator,  LocalDiscriminator, UNet_SemanticRemoval
from .loss import SegmentEntropyLoss ,WeightedL1Loss, AdversarialLoss, PerceptualLoss, HueLoss, GradientLoss, OnewayHueLoss, TVLoss, GrayWorldLoss, DarkChannelLoss, TransmissionLoss, VerticalDepthLoss, SimilarityContrastLoss, Masked_DarkChannelLoss
from .refinenet_network import rf_lw50_ours
import torchvision.transforms.functional as TFF
import torchvision.transforms as TF



class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()

        if config.MODEL == 1:
            self.name = 'reconstruct'
        elif config.MODEL == 2:
            self.name = 'feature_process'

        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, 'weights.pth')
        self.gen_optimizer_path = os.path.join(config.PATH, 'optimizer_'+self.name + '.pth')
        self.dis_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')
        self.transformer_weights_path = os.path.join(config.PATH, self.name + '.pth')
        self.transformer_discriminator_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')
        self.reconstructor_weights_path = os.path.join(config.PATH, self.name + '.pth')

    def load(self):
        pass

    def save(self, save_best, psnr, iteration):
        pass




class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss(type='lsgan')
        self.weighted_l1loss = WeightedL1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.hue_loss = HueLoss()
        self.oneway_hue_loss = OnewayHueLoss()
        self.grad_loss = GradientLoss()
        self.tv_loss = TVLoss()
        self.gray_world_loss = GrayWorldLoss()
        self.dark_channel_loss = Masked_DarkChannelLoss()

        self.vertical_loss = VerticalDepthLoss()
        self.segment_entropy_loss = SegmentEntropyLoss()
        # self.transmission_penalty_loss = TransmissionPenalty()
        self.transmission_loss = TransmissionLoss()
        self.contrastive_loss = SimilarityContrastLoss()
        self._mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).cuda()
        self._std = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).cuda()
        self.use_dc_A = True if config.USE_DC_A == 1 else False
        self.transmission_estimator = TransmissionEstimator()
        self.min_T = self.config.MIN_T
        self.max_T = self.config.MAX_T
        self.get_random_patch_function = TF.RandomCrop(size=[self.config.CROP_SIZE // 2])




        self.net_c2h = UNet_HazeRefine(base_channel_nums=config.BASE_CHANNEL_NUM)

        if str.lower(self.config.SEG_DATASET) == 'cityscapes':
            self.classes_num = 19
            # C = 304
            self.net_seg = _segm_mobilenet(num_classes=19,output_stride=16,pretrained_backbone=True)
            checkpoint = torch.load('./checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth', map_location=torch.device('cpu'))
            self.net_seg = self.net_seg.cuda().eval()
            self.net_seg.load_state_dict(checkpoint['model_state'])
            del checkpoint
            print("Resume model from %s" % './checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')

        elif str.lower(self.config.SEG_DATASET) == 'nyu':
            #c=256
            self.classes_num = 40
            self.net_seg = rf_lw50_ours(num_classes=40,pretrained=True).eval().cuda()
            print("Resume model from %s" % './checkpoints/50_nyud.ckpt')

        self.net_h2c = UNet_SemanticRemoval(base_channel_nums=config.BASE_CHANNEL_NUM, out_channels=1, dataset=self.config.SEG_DATASET,
                                            semantic_map_channels=self.classes_num, norm_type=self.config.NORM_TYPE)


        self.epoch = 0

        if config.MODE == 1:

            self.discriminator_c = Discriminator(in_channels=3, use_spectral_norm=True, use_sigmoid=True)
            self.discriminator_h = Discriminator(in_channels=3, use_spectral_norm=True, use_sigmoid=True)
            self.discriminator_h2c_local = LocalDiscriminator(in_channels=3, use_spectral_norm=True, use_sigmoid=True)


            self.optimizer = optim.Adam(
                [
                    {'params': self.net_c2h.parameters()},
                    {'params': self.net_h2c.parameters()},

                ],

                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2),
                weight_decay=config.WEIGHT_DECAY
            )


            self.optimizer_dis = optim.Adam(
                [
                    {'params': self.discriminator_c.parameters()},
                    {'params': self.discriminator_h.parameters()},
                    {'params': self.discriminator_h2c_local.parameters()},
                    # {'params': self.discriminator_seg_c.parameters()},
                    # {'params': self.discriminator_seg_h.parameters()},

                ],

                lr=float(config.LR * config.D2G_LR),
                betas=(config.BETA1, config.BETA2)
            )


            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.T_MAX, last_epoch=self.epoch-1)







    def forward_h2c(self, x, requires_t=False, use_GF=False):
        x_coarse_t = self.transmission_estimator.direct_get_transmission(x).clamp(0.1,1)
        A = self.transmission_estimator.get_atmosphere_light_new(x)
        x_coarse = self.transmission_estimator.get_radiance(x,A,x_coarse_t).detach().clamp(0.01,1)

        if str.lower(self.config.DATASET) in ['real']:
            segment_h, feature = self.forward_segment(x, require_immediate_features=True)
            return self.net_h2c(x, segment_h, feature, requires_t, use_GF=use_GF)

        else:
            segment_h, feature = self.forward_segment(x_coarse, require_immediate_features=True)
            return self.net_h2c(x, segment_h, feature, requires_t, use_GF=use_GF)


    # I = J*t + A(1-t) = Jt+A - At = t(J-A) + A, t= (I-A)/(J-A)
    def forward_c2h_cycle(self,x, x_segment, t_h2c):
        x_segment = F.interpolate(x_segment,size=x.shape[2:4],mode='nearest')
        seg_result_index = torch.argmax(x_segment, dim=1, keepdim=True)
        if str.lower(self.config.DATASET) in ['cityscapes', 'real']:
            A = self.transmission_estimator.get_atmosphere_light_new(x)
        elif str.lower(self.config.DATASET) in ['nyu']:
            A  = 0.9 * torch.ones(x.shape[0], 3, 1, 1, device='cuda')
        t_new = t_h2c.clone()

        for i in range(self.classes_num):
            t_new[torch.where(seg_result_index==i)] = torch.mean(t_h2c[torch.where(seg_result_index==i)])

        t_new = self.transmission_estimator.get_refined_transmission(x, t_new)

        x = (x*t_new + A*(1-t_new)).clamp(0,1)
        x = self.net_c2h(x)

        return x

    def forward_c2h_random(self, x, x_segment):

        x_segment = F.interpolate(x_segment,size=x.shape[2:4],mode='nearest')
        if str.lower(self.config.SEG_DATASET) in ['cityscapes']:
            A = self.transmission_estimator.get_atmosphere_light_new(x)
        elif str.lower(self.config.SEG_DATASET) in ['nyu']:
            A = 0.9 * torch.ones(x.shape[0], 3, 1, 1, device='cuda')


        t_random_index = self.min_T + torch.rand(self.classes_num, device='cuda')*(self.max_T-self.min_T)  # (1,40)
        seg_result_index = torch.argmax(x_segment,dim=1,keepdim=True)  # (1,1,64,64)
        seg_result_transmission = t_random_index[seg_result_index]

        refined_t = self.transmission_estimator.get_refined_transmission(x,seg_result_transmission).detach()

        x = (x * refined_t + A*(1-refined_t)).clamp(0,1)

        x = self.net_c2h(x)

        return x

    def forward_segment(self, x, require_immediate_features=False):
        x = (x-self._mean)/self._std

        result, feature = self.net_seg(x)

        if not require_immediate_features:
            return result
        else:
            return result, feature





    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]




    def save(self, save_best, psnr, iteration):
        if self.config.MODEL == 1:
            torch.save({
                'net_h2c':self.net_h2c.state_dict(),
                'net_c2h':self.net_c2h.state_dict(),
            },self.gen_weights_path[:-4]+'_'+self.name+'.pth' if not save_best else self.gen_weights_path[
                                                                   :-4] +'_'+self.name+ "_%.2f" % psnr + "_RGB" + "_%d" % iteration + '.pth', _use_new_zipfile_serialization=False)
            torch.save({'discriminator_c': self.discriminator_c.state_dict(),
                        'discriminator_h': self.discriminator_h.state_dict(),
                        'discriminator_local': self.discriminator_h2c_local.state_dict(),

                        }, self.gen_weights_path[
                           :-4] + '_' + self.name + '_dis.pth' if not save_best else self.gen_weights_path[
                                                                                     :-4] + '_' + self.name + "_dis_%.2f" % psnr +
                                                                                        "_RGB" + "_%d" % iteration + '.pth')

            torch.save({
                'iteration': self.iteration,
                'epoch': self.epoch,
                'scheduler': self.scheduler.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'optimizer_dis': self.optimizer_dis.state_dict(),

            }, self.gen_optimizer_path if not save_best else self.gen_optimizer_path[
                                                           :-4] + "_%.2f" % psnr + "_RGB" + "_%d" % iteration + '.pth', _use_new_zipfile_serialization=False)


    def load(self):


        if os.path.exists(self.gen_weights_path[:-4] + '_reconstruct' + '.pth'):
            print('Loading %s weights...' % 'reconstruct')

            if torch.cuda.is_available():
                weights = torch.load(self.gen_weights_path[:-4] + '_reconstruct' + '.pth')
            else:
                weights = torch.load(self.gen_weights_path[:-4] + '_reconstruct' + '.pth',
                                     lambda storage, loc: storage)

            self.net_h2c.load_state_dict(weights['net_h2c'])
            self.net_c2h.load_state_dict(weights['net_c2h'])


            print('Loading %s weights...' % 'reconstruct complete!')

        if os.path.exists(self.gen_weights_path[:-4] + '_' + self.name + '_dis.pth') and self.config.MODE == 1:
            print('Loading discriminator weights...')

            if torch.cuda.is_available():
                weights = torch.load(self.gen_weights_path[:-4] + '_' + self.name + '_dis.pth')
            else:
                weights = torch.load(self.gen_weights_path[:-4] + '_' + self.name + '_dis.pth',
                                     lambda storage, loc: storage)

            self.discriminator_c.load_state_dict(weights['discriminator_c'])
            self.discriminator_h.load_state_dict(weights['discriminator_h'])
            self.discriminator_h2c_local.load_state_dict(weights['discriminator_local'])


        if os.path.exists(self.gen_optimizer_path) and self.config.MODE == 1:
            print('Loading %s optimizer...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(self.gen_optimizer_path)
            else:
                data = torch.load(self.gen_optimizer_path, lambda storage, loc: storage)

            self.optimizer.load_state_dict(data['optimizer'])
            self.scheduler.load_state_dict(data['scheduler'])
            self.epoch = data['epoch']
            self.iteration = data['iteration']
            self.optimizer_dis.load_state_dict(data['optimizer_dis'])

    def backward(self, gen_loss):
        gen_loss.backward()
        self.optimizer.step()


    def update_scheduler(self):
        self.scheduler.step()



    def process(self, clean_images, hazy_images):
        self.iteration += 1

        self.optimizer_dis.zero_grad()
        self.discriminator_h.zero_grad()
        self.discriminator_c.zero_grad()



        clean_images_h2c_for_dehaze, t_h2c = self.forward_h2c(hazy_images, requires_t=True)
        clean_images_h2c_for_rehaze = clean_images_h2c_for_dehaze#.detach()

        segment_h2c = self.forward_segment(clean_images_h2c_for_dehaze)

        hazy_images_h2c2h = self.forward_c2h_cycle(clean_images_h2c_for_rehaze,segment_h2c.detach(), t_h2c.detach())


        segment_c = self.forward_segment(clean_images)

        if self.config.USE_SKY_MASK == 1:
            sky_segment_h2c = self.get_sky_mask(segment_h2c)
        else:
            sky_segment_h2c = torch.zeros(hazy_images.shape,device='cuda')

        hazy_images_c2h_for_rehaze = self.forward_c2h_random(clean_images, segment_c.detach())
        hazy_images_c2h_for_dehaze = hazy_images_c2h_for_rehaze#.detach()
        clean_images_c2h2c= self.forward_h2c(hazy_images_c2h_for_dehaze)


        segment_c2h = self.forward_segment(hazy_images_c2h_for_rehaze)
        #




        gen_loss = 0
        dis_loss = 0

        #### dis loss ####

        ### global ###
        dis_real_clean, _ = self.discriminator_c(clean_images)
        dis_fake_clean, _ = self.discriminator_c(clean_images_h2c_for_dehaze.detach())

        dis_clean_real_loss = self.adversarial_loss((dis_real_clean), is_real=True, is_disc=True)
        dis_clean_fake_loss = self.adversarial_loss((dis_fake_clean), is_real=False, is_disc=True)

        dis_clean_loss = (dis_clean_real_loss + dis_clean_fake_loss) / 2
        dis_clean_loss.backward()

        dis_real_haze, _ = self.discriminator_h(
            (hazy_images))
        dis_fake_haze, _ = self.discriminator_h(
            hazy_images_c2h_for_rehaze.detach())

        dis_haze_real_loss = self.adversarial_loss((dis_real_haze), is_real=True, is_disc=True)
        dis_haze_fake_loss = self.adversarial_loss((dis_fake_haze), is_real=False, is_disc=True)
        dis_haze_loss = (dis_haze_real_loss + dis_haze_fake_loss) / 2
        dis_haze_loss.backward()


        ###local###
        clean_images_patch = self.get_random_patch(clean_images)
        clean_images_h2c_patch = self.get_random_patch(clean_images_h2c_for_dehaze.detach())

        dis_real_clean_local, _ = self.discriminator_h2c_local(clean_images_patch)
        dis_fake_clean_local, _ = self.discriminator_h2c_local(
            clean_images_h2c_patch.detach())

        dis_clean_real_loss_local = self.adversarial_loss((dis_real_clean_local), is_real=True, is_disc=True)
        dis_clean_fake_loss_local = self.adversarial_loss((dis_fake_clean_local), is_real=False, is_disc=True)

        dis_clean_loss_local = (dis_clean_fake_loss_local + dis_clean_real_loss_local) / 2
        dis_clean_loss_local.backward()

        dis_loss += (dis_clean_fake_loss + dis_clean_real_loss + dis_haze_real_loss + dis_haze_fake_loss) / 4
        dis_loss_local = dis_clean_loss_local / 2






        self.optimizer_dis.step()



        ### gen loss ####
        self.optimizer.zero_grad()
        self.net_h2c.zero_grad()
        self.net_c2h.zero_grad()
        self.net_seg.zero_grad()


        ### cycle reconstruction loss###
        cycle_loss_c2h2c = self.l1_loss(clean_images,
                                        clean_images_c2h2c)
        cycle_loss_h2c2h = self.l1_loss(hazy_images, hazy_images_h2c2h)
        cycle_loss = (cycle_loss_c2h2c + cycle_loss_h2c2h) / 2



        dark_channel_loss = self.dark_channel_loss(clean_images_h2c_for_dehaze, mask=sky_segment_h2c)

        # ### TV loss ###
        tv_loss_h2c = self.tv_loss(clean_images_h2c_for_dehaze)
        tv_loss_c2h2c = self.tv_loss(hazy_images_c2h_for_rehaze)

        tv_loss = (tv_loss_h2c + tv_loss_c2h2c) / 2



        ### global ###
        gen_fake_haze, _ = self.discriminator_h(
            (hazy_images_c2h_for_rehaze))
        gen_fake_clean, _ = self.discriminator_c(
            clean_images_h2c_for_dehaze)
        gen_fake_clean_local, _ = self.discriminator_h2c_local(
            clean_images_h2c_patch)

        gen_fake_haze_ganloss = self.adversarial_loss((gen_fake_haze), is_real=True, is_disc=False)
        gen_fake_clean_ganloss = self.adversarial_loss((gen_fake_clean), is_real=True, is_disc=False)
        gen_fake_clean_local_ganloss = self.adversarial_loss((gen_fake_clean_local), is_real=True, is_disc=False)

        gen_gan_loss = (gen_fake_clean_ganloss + gen_fake_haze_ganloss + gen_fake_clean_local_ganloss) / 3




        ### segment entropy loss ###


        segment_loss_h2c = self.segment_entropy_loss(segment_h2c)
        segment_loss_c2h = -self.segment_entropy_loss(segment_c2h)

        segment_entropy_loss = segment_loss_h2c-segment_loss_c2h




        ### total loss ###

        gen_loss += self.config.GAN_LOSS_WEIGHT * gen_gan_loss
        gen_loss += self.config.CYCLE_LOSS_WEIGHT * cycle_loss
        gen_loss += self.config.SEGMENT_ENTROPY_LOSS_WEIGHT * segment_entropy_loss
        gen_loss += self.config.DARK_CHANNEL_LOSS_WEIGHT * dark_channel_loss

        gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss


        gen_loss.backward()



        self.optimizer.step()



        logs = [
            ("g_cyc", cycle_loss.item()),
            ("g_tv", tv_loss.item()),
            ("g_segment", segment_entropy_loss.item()),

            ("g_dark", dark_channel_loss.item()),
            ("g_tv", tv_loss.item()),
            ("g_gan", gen_gan_loss.item()),
            ("g_total", gen_loss.item()),
            ("d_dis", dis_loss.item()),
            ("lr", self.get_current_lr()),
        ]
        return clean_images_c2h2c, gen_loss, dis_loss, logs

    def get_random_patch(self, images):
        images = TFF.pad(images, padding=self.config.CROP_SIZE // 4, padding_mode='reflect')
        patch = self.get_random_patch_function(images)
        return patch





    def get_sky_mask(self, segment):
        segment = segment.max(1)[1]
        segment = (segment==10).unsqueeze(1)
        return segment.float()

















