""" 
End2End optical design with only 5 lines of code.

Technical Paper:
    Xinge Yang, Qiang Fu and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," ArXiv preprint 2023.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""
import os
import yaml
import wandb
import shutil
import logging
import random
import string
import numpy as np
import cv2 as cv
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from deeplens import *

def config():
    # ==> Config
    with open('configs/1_end2end_5lines.yml') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # ==> Result folder
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for i in range(4))
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_name = current_time + '-End2End-5-lines-' + random_string
    result_dir = f'./results/{exp_name}'
    os.makedirs(result_dir, exist_ok=True)
    args['result_dir'] = result_dir

    if args['seed'] is None:
        seed = random.randint(0, 100)
        args['seed'] = seed
    set_seed(args['seed'])
    
    # ==> Log
    set_logger(result_dir)
    logging.info(f'EXP: {args["EXP_NAME"]}')
    if not args['DEBUG']:
        raise Exception('Add your wandb logging config here.')

    # ==> Device
    num_gpus = torch.cuda.device_count()
    args['num_gpus'] = num_gpus
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    args['device'] = device
    logging.info(f'Using {num_gpus} {torch.cuda.get_device_name(0)} GPU(s)')

    # ==> Save config and original code
    with open(f'{result_dir}/config.yml', 'w') as f:
        yaml.dump(args, f)

    shutil.copy('1_end2end_5lines.py', f'{result_dir}/1_end2end_5lines.py')
        
    return args


def validate(net, lens, epoch, args, val_loader):
    # Complete quantiative evaluation
    return


def end2end_train(lens, net, args):
    device = args['device']
    result_dir = args['result_dir']

    # ==> Dataset
    train_set = ImageDataset(args['train']['train_dir'], lens.sensor_res)
    train_loader = DataLoader(train_set, batch_size=args['train']['bs'])

    # ==> Network optimizer
    batchs = len(train_loader)
    epochs = args['train']['epochs']
    net_optim = torch.optim.AdamW(net.parameters(), lr=args['network']['lr'], betas=(0.9, 0.98), eps=1e-08)
    net_sche = torch.optim.lr_scheduler.CosineAnnealingLR(net_optim, T_max=epochs*batchs, eta_min=0, last_epoch=-1)
    
    # ==> Lens optimizer
    # ========================================
    # Line 2: get lens optimizers
    # ========================================
    lens_lrs = [float(i) for i in args['lens']['lr']]
    lens_optim = lens.get_optimizer(lr=lens_lrs)
    lens_sche= torch.optim.lr_scheduler.CosineAnnealingLR(lens_optim, T_max=epochs*batchs, eta_min=0, last_epoch=-1)

    # ==> Criterion
    cri_l2 = nn.L1Loss()

    # ==> Log
    logging.info(f'Start End2End optical design.')
    lens.write_lens_json(f'{result_dir}/epoch0.json')
    lens.analysis(f'{result_dir}/epoch0', render=False, zmx_format=True)
    
    # ==> Training
    for epoch in range(args['train']['epochs'] + 1):

        # ==> Train 1 epoch
        for img_org in tqdm(train_loader):            
            img_org = img_org.to(device)

            # => Render image
            # ========================================
            # Line 3: plug-and-play diff-rendering
            # ========================================
            img_render = lens.render(img_org)

            # => Image restoration
            img_rec = net(img_render)

            # => Loss
            L_rec = cri_l2(img_rec, img_org)

            # => Back-propagation
            net_optim.zero_grad()
            # ========================================
            # Line 4: zero-grad
            # ========================================
            lens_optim.zero_grad()

            L_rec.backward()

            net_optim.step()
            # ========================================
            # Line 5: step
            # ========================================
            lens_optim.step()

            if not args['DEBUG']:
                wandb.log({"loss_class": L_rec.detach().item()})

        net_sche.step()
        lens_sche.step()
        
        logging.info(f'Epoch{epoch+1} finishs.')


        # ==> Evaluate
        if epoch % 1 == 0:
            net.eval()
            with torch.no_grad():
                # => Save data and simple evaluation
                lens.write_lens_json(f'{result_dir}/epoch{epoch}.json')
                lens.analysis(f'{result_dir}/epoch{epoch}', render=False, zmx_format=True)

                torch.save(net.state_dict(), f'{result_dir}/net_epoch{epoch}.pth')
                
                # => Qualitative evaluation
                img1 = cv.cvtColor(cv.imread(f'./datasets/cat.png'), cv.COLOR_BGR2RGB)
                img1 = cv.resize(img1, args['train']['img_res']).astype(np.float32)
                img1 = torch.from_numpy(img1/255.).permute(2, 0, 1).unsqueeze(0).to(device)
                img1 = normalize_ImageNet(img1)
                
                img1_render = lens.render(img1)
                psnr_render = batch_PSNR(img1, img1_render)
                ssim_render = batch_SSIM(img1, img1_render)
                save_image(denormalize_ImageNet(img1_render), f'{result_dir}/img1_render_epoch{epoch}.png')
                img1_rec = net(img1_render)
                psnr_rec = batch_PSNR(img1, img1_rec)
                ssim_rec = batch_SSIM(img1, img1_rec)
                save_image(denormalize_ImageNet(img1_rec), f'{result_dir}/img1_rec_epoch{epoch}.png')

                logging.info(f'Epoch [{epoch}/{args["train"]["epochs"]}], PSNR_render: {psnr_render:.4f}, SSIM_render: {ssim_render:.4f}, PSNR_rec: {psnr_rec:.4f}, SSIM_rec: {ssim_rec:.4f}')

                # => Quantitative evaluation
                # validate(net, lens, epoch, args, val_loader)
                
            net.train()


if __name__=='__main__':
    args = config()
    
    # ========================================
    # Line 1: load a lens
    # ========================================
    lens = GeoLens(filename=args['lens']['path'], sensor_res=args['train']['img_res'])
    net = ImageRestorationNet()
    net = net.to(lens.device)
    if args['network']['pretrained']:
        net.load_state_dict(torch.load(args['network']['pretrained']))
    
    end2end_train(lens, net, args)