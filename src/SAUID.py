import os
import numpy as np
import torch
import torch.nn.functional as F
import kornia
import cv2
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import Model
from .utils import Progbar, create_dir, stitch_images, imsave, decode_segment_result
from .metrics import PSNR_RGB
from torch.utils.tensorboard import SummaryWriter
import math


class SAUID():
    def __init__(self, config):
        self.config = config

        self.model = Model(config).to(config.DEVICE)

        self.psnr = PSNR_RGB(255.0).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, crop_size=None, hazy_flist=config.TEST_HAZY_FLIST,
                                        clean_flist=config.TEST_CLEAN_FLIST, augment=False,
                                        split=self.config.TEST_MODE)

        else:
            self.train_dataset = Dataset(config, crop_size=config.CROP_SIZE, clean_flist=config.TRAIN_CLEAN_FLIST,
                                         hazy_flist=config.TRAIN_HAZY_FLIST, augment=True, split='unpair')
            # self.val_dataset = Dataset(config, crop_size=256, clean_flist=config.VAL_FLIST,  augment=False, preload_to_memory=False, split='validate')
            self.val_dataset = Dataset(config, crop_size=None, hazy_flist=config.VAL_HAZY_FLIST,
                                       clean_flist=config.VAL_CLEAN_FLIST, augment=False, split='pair_test')

            # self.val_dataset = Dataset(config, crop_size=None, hazy_flist=config.TEST_HAZY_FLIST, augment=False, split='test')
            self.test_dataset = Dataset(config, crop_size=None, hazy_flist=config.TEST_HAZY_FLIST,
                                        clean_flist=config.TEST_CLEAN_FLIST, augment=False, split=self.config.TEST_MODE)
            self.sample_dataset = Dataset(config, crop_size=config.CROP_SIZE, clean_flist=config.TRAIN_CLEAN_FLIST,
                                          hazy_flist=config.TRAIN_HAZY_FLIST, augment=False, split='unpair')

            self.sample_iterator = self.sample_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')
        self.eval_path = os.path.join(config.PATH, 'eval')
        self.log_path = os.path.join(config.PATH, 'logs')

        # if config.RESULTS is not None:
        #     self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + self.model.name + '.dat')

    def load(self):
        self.model.load()

    def save(self, save_best=False, psnr=None, iteration=None):
        self.model.save(save_best, psnr, iteration)

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)
        epoch = self.model.epoch
        self.writer = SummaryWriter(self.log_path)
        self.loss_list = []
        highest_psrn = 0
        highest_meanT = 0
        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while (keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
            print('epoch:', epoch)

            index = 0

            for items in train_loader:
                self.model.train()
                self.model.net_seg.eval()
                # self.model.eval_()

                clean_images, hazy_images = self.cuda(items[0], items[1])

                if model == 1:

                    outputs, gen_loss, dis_loss, logs = self.model.process(clean_images, hazy_images)


                    psnr = self.psnr(self.postprocess(clean_images), self.postprocess(outputs))
                    logs.append(('psnr_cyc', psnr.item()))






                    iteration = self.model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                           ("epoch", epoch),
                           ("iter", iteration),
                       ] + logs

                index += 1
                progbar.add(len(clean_images), values=logs if self.config.VERBOSE else [x for x in logs])

                self.write_tensorboard(logs, iteration)
                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

            # update epoch for scheduler
            self.model.epoch = epoch
            self.model.update_scheduler()
            self.writer.close()
        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            drop_last=False,
            shuffle=False,
            num_workers=4,
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        psnrs = []
        ts = []
        with torch.no_grad():
            for items in val_loader:
                iteration += 1


                if model == 1 and self.val_dataset.split == 'pair_test':
                    clean_images, hazy_images = self.cuda(items[0], items[1])

                    h, w = hazy_images.shape[2:4]

                    hazy_input_images = self.pad_input(hazy_images)
                    predicted_results = self.model.forward_h2c(hazy_input_images)

                    predicted_results = self.crop_result(predicted_results, h, w)

                    psnr = self.psnr(self.postprocess(predicted_results), self.postprocess(clean_images))

                    psnrs.append(psnr.item())
                    logs = []
                    logs.append(('psnr', psnr.item()))


                logs = [("it", iteration), ] + logs
                progbar.add(len(hazy_images), values=logs)

        return np.mean(psnrs) if self.val_dataset.split == 'pair_test' else np.mean(ts)

    def test(self):
        model = self.config.MODEL
        self.model.eval()
        create_dir(self.results_path)
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0

        psnrs = []
        times = []

        with torch.no_grad():
            for items in test_loader:
                if self.test_dataset.split == 'pair_test':

                    name = self.test_dataset.load_name(index)[:-4] + '.png'

                    clean_images, hazy_images = self.cuda(items[0], items[1])
                    index += 1

                    # print(hazy_images.shape)
                    if model == 1:
                        ## check if the input size is multiple of 4
                        h, w = hazy_images.shape[2:4]

                        hazy_input_images = self.pad_input(hazy_images)
                        predicted_results = self.model.forward_h2c(hazy_input_images)

                        predicted_results = self.crop_result(predicted_results, h, w)

                        psnr = self.psnr(self.postprocess(predicted_results), self.postprocess(clean_images))
                        psnrs.append(psnr.item())
                        print('PSNR:', psnr)

                        predicted_results = self.postprocess(predicted_results)[0]

                        path = os.path.join(self.results_path, self.model.name)
                        create_dir(path)
                        save_name = os.path.join(path, name)
                        imsave(predicted_results, save_name)
                        print(save_name)




                elif self.test_dataset.split == 'hazy':

                    name = self.test_dataset.load_name(index)[:-4] + '.png'

                    if self.config.USE_GUIDED_FILTER == 1:
                        use_GF=  True
                    else:
                        use_GF = False

                    hazy_images = items.to(self.config.DEVICE)
                    index += 1

                    if model == 1:
                        torch.cuda.empty_cache()
                        ## check if the input size is multiple of 4
                        h, w = hazy_images.shape[2:4]
                        print(hazy_images.shape)

                        hazy_input_images = self.pad_input(hazy_images)
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        predicted_results = self.model.forward_h2c(hazy_input_images, use_GF=use_GF)
                        end.record()
                        torch.cuda.synchronize()
                        times.append(start.elapsed_time(end))
                        predicted_results = self.crop_result(predicted_results, h, w)


                        predicted_results = self.postprocess(predicted_results)[0]

                        path = os.path.join(self.results_path, self.model.name)
                        create_dir(path)
                        save_name = os.path.join(path, name)
                        imsave(predicted_results, save_name)
                        print(save_name)



                    if model == 2:

                        ## check if the input size is multiple of 4
                        h, w = hazy_images.shape[2:4]

                        if h * w > 2000 * 3000:
                            continue
                        hazy_input_images = self.pad_input(hazy_images)

                        predicted_A, predicted_beta = self.model.para_estimator(hazy_input_images)

                        for i in range(5):
                            beta_in = predicted_beta + -0.2 + 0.1 * i
                            predicted_results = self.model.forward_h2c_given_parameter(hazy_input_images, predicted_A,
                                                                                       beta_in)
                            predicted_results = self.crop_result(predicted_results, h, w)

                            images_gather = stitch_images(
                                self.postprocess(hazy_images),
                                self.postprocess(predicted_results),
                                img_per_row=1
                            )

                            path = os.path.join(self.results_path, self.model.name + '_gather')
                            create_dir(path)

                            beta_name = format(beta_in.item(), '.3f')
                            save_name = os.path.join(path, name[:-4] + str(beta_name) + '.png')

                            images_gather.save(save_name)

                elif self.test_dataset.split == 'clean':

                    name = self.test_dataset.load_name(index)[:-4] + '.png'

                    clean_images = items.to(self.config.DEVICE)
                    index += 1

                    if model == 1:

                        ## check if the input size is multiple of 4
                        h, w = clean_images.shape[2:4]

                        if h * w > 2000 * 3000:
                            continue

                        clean_input_images = self.pad_input(clean_images)
                        predicted_depth = self.model.forward_depth(clean_input_images)
                        # if use_guided_filter:
                        #     predicted_depth = self.model.net_c2h.transmission_estimator.get_refined_transmission(clean_input_images, predicted_depth)

                        for beta_in in [0.6, 1.2, 1.8]:
                            predicted_results = self.model.forward_c2h_given_parameters(clean_input_images,
                                                                                        predicted_depth, beta_in)
                            predicted_results = self.crop_result(predicted_results, h, w)


                            predicted_results = self.postprocess(predicted_results)[0]

                            path = os.path.join(self.results_path, self.model.name + '_haze')
                            create_dir(path)
                            name_s = name[:-4] + '_' + str(beta_in) + '.png'
                            save_name = os.path.join(path, name_s)
                            imsave(predicted_results, save_name)
                            print(index, save_name)

                        predicted_depth = self.crop_result(predicted_depth, h, w)
                        predicted_depth = self.minmax_depth(predicted_depth)
                        predicted_depth = self.generate_color_map(predicted_depth, [h, w])

                        predicted_depth = predicted_depth[0]
                        path = os.path.join(self.results_path, self.model.name + '_depth')
                        create_dir(path)
                        name_s = name[:-4] + '_depth' + '.png'
                        save_name = os.path.join(path, name_s)
                        imsave(predicted_depth, save_name)
                        print(index, save_name)




            print('AVG times:' + str(np.mean(times)))
            print('Total PSNR_' + ('YCbCr:' if self.config.PSNR == 'YCbCr' else 'RGB:'), np.mean(psnrs))
            print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.sample_dataset) == 0:
            return
        self.model.eval()

        model = self.config.MODEL

        items = next(self.sample_iterator)
        clean_images, hazy_images = self.cuda(items[0], items[1])

        # inpaint with edge model / joint model
        with torch.no_grad():
            iteration = self.model.iteration

            if model == 1:
                h, w = hazy_images.shape[2:4]
                ## check if the input size is multiple of 4
                clean_images_h2c, t_h2c = self.model.forward_h2c(hazy_images, requires_t=True)
                segment_h2c = self.model.forward_segment(clean_images_h2c)
                hazy_images_h2c2h = self.model.forward_c2h_cycle(clean_images_h2c,segment_h2c,t_h2c)
                # print(clean_images.shape)
                # print(depth_c.shape)

                segment_c = self.model.forward_segment(clean_images)
                hazy_images_c2h = self.model.forward_c2h_random(clean_images,segment_c)
                clean_images_c2h2c = self.model.forward_h2c(hazy_images_c2h)

                seg_direct_h = self.model.forward_segment(hazy_images)
                print(seg_direct_h.shape)
                seg_direct_c2h = self.model.forward_segment(hazy_images_c2h)

                seg_direct_c = self.model.forward_segment(clean_images)
                seg_direct_h2c = self.model.forward_segment(clean_images_h2c)


                seg_direct_h = decode_segment_result(seg_direct_h,dataset=self.config.SEG_DATASET)
                seg_direct_c2h = decode_segment_result(seg_direct_c2h,dataset=self.config.SEG_DATASET)
                seg_direct_c = decode_segment_result(seg_direct_c,dataset=self.config.SEG_DATASET)
                seg_direct_h2c = decode_segment_result(seg_direct_h2c,dataset=self.config.SEG_DATASET)

                coarse_h2c_t = self.model.transmission_estimator.direct_get_transmission(hazy_images).clamp(0.1,1)
                coarse_h2c_A = self.model.transmission_estimator.get_atmosphere_light_new(hazy_images)
                coarse_h2c = self.model.transmission_estimator.get_radiance(hazy_images,coarse_h2c_A,coarse_h2c_t).clamp(0.01,1)
                seg_direct_coarse_h2c = self.model.forward_segment(coarse_h2c)
                seg_direct_coarse_h2c = decode_segment_result(seg_direct_coarse_h2c,dataset=self.config.SEG_DATASET)



                images_sample = stitch_images(
                    self.postprocess(hazy_images),
                    self.postprocess(seg_direct_h,size=[256,256]),
                    self.postprocess(coarse_h2c),
                    self.postprocess(seg_direct_coarse_h2c, size=[256, 256]),
                    self.postprocess(clean_images_h2c),
                    self.postprocess(seg_direct_h2c,size=[256,256]),
                    self.postprocess(hazy_images_h2c2h),

                    self.postprocess(clean_images),
                    self.postprocess(seg_direct_c,size=[256,256]),
                    self.postprocess(hazy_images_c2h),
                    self.postprocess(seg_direct_c2h,size=[256,256]),
                    self.postprocess(clean_images_c2h2c),
                    img_per_row=1,
                )





            if self.log_path:
                self.writer.add_image('sample_image', np.array(images_sample), iteration, dataformats='HWC')


    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def cuda_list(self, args):
        return [item.to(self.config.DEVICE) for item in args]

    def lr_schedule_cosdecay(self, t, T, init_lr):
        lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
        return lr

    def postprocess(self, img, size=None):
        # [0, 1] => [0, 255]
        if size is not None:
            img = torch.nn.functional.interpolate(img, size, mode='nearest')
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def generate_color_map(self, imgs, size=[256, 256]):
        # N 1 H W -> N H W 3 color map
        if torch.max(imgs) > 1 or torch.min(imgs) < 0:
            imgs = self.minmax_depth(imgs, blur=True)
        imgs = (imgs * 255.0).int().squeeze(1).cpu().numpy().astype(np.uint8)
        N, height, width = imgs.shape

        colormaps = np.full((N, size[0], size[1], 3), 1)

        for i in range(imgs.shape[0]):
            colormaps[i] = cv2.resize((cv2.applyColorMap(imgs[i], cv2.COLORMAP_HOT)), (size[1], size[0]))

        # transfer to tensor than to gpu
        # firstly the channel BGR->RGB
        colormaps = colormaps[..., [2, 1, 0]]

        # than to tensor, to gpu
        colormaps = torch.from_numpy(colormaps).cuda()

        return colormaps

    def crop_result(self, result, input_h, input_w, times=32):
        crop_h = crop_w = 0

        if input_h % times != 0:
            crop_h = times - (input_h % times)

        if input_w % times != 0:
            crop_w = times - (input_w % times)

        if crop_h != 0:
            result = result[..., :-crop_h, :]

        if crop_w != 0:
            result = result[..., :-crop_w]
        return result

    def pad_input(self, input, times=32):
        input_h, input_w = input.shape[2:]
        pad_h = pad_w = 0

        if input_h % times != 0:
            pad_h = times - (input_h % times)

        if input_w % times != 0:
            pad_w = times - (input_w % times)

        input = torch.nn.functional.pad(input, (0, pad_w, 0, pad_h), mode='reflect')

        return input

    def minmax_depth(self, depth, blur=True):
        n, c, h, w = depth.shape

        if blur:
            depth = F.pad(depth, [4, 4, 4, 4], 'reflect')
            depth = kornia.filters.median_blur(depth, (9, 9))
            depth = depth[:, :, 3:h - 3, 3:w - 3]

        D_max = torch.max(depth.reshape(n, c, -1), dim=2, keepdim=True)[0].unsqueeze(3)
        D_min = torch.min(depth.reshape(n, c, -1), dim=2, keepdim=True)[0].unsqueeze(3)

        depth = (depth - D_min) / (D_max - D_min + 0.01)
        return depth

    def write_tensorboard(self, logs, iteration):
        iteration = int(iteration)
        if self.loss_list == []:
            self.loss_list = [0] * len(logs)
        if iteration % self.config.LOG_INTERVAL == 0:
            for i, item in enumerate(logs):
                if item[0] not in ['it', 'epoch']:
                    self.writer.add_scalar(item[0], self.loss_list[i] / int(self.config.LOG_INTERVAL), iteration)
            self.loss_list = [0] * len(logs)
        else:
            for i, item in enumerate(logs):
                if item[0] not in ['it', 'epoch']:
                    self.loss_list[i] += float(item[1])
