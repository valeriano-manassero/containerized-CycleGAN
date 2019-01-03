import itertools

import os
import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from options import Options
from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset
from torchvision.utils import save_image


def train(opt):
    netg_a2b = Generator(opt.input_nc, opt.output_nc)
    netg_b2a = Generator(opt.output_nc, opt.input_nc)
    netd_a = Discriminator(opt.input_nc)
    netd_b = Discriminator(opt.output_nc)

    if opt.cuda:
        netg_a2b.cuda()
        netg_b2a.cuda()
        netd_a.cuda()
        netd_b.cuda()

    netg_a2b.apply(weights_init_normal)
    netg_b2a.apply(weights_init_normal)
    netd_a.apply(weights_init_normal)
    netd_b.apply(weights_init_normal)

    criterion_gan = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    optimizer_g = torch.optim.Adam(itertools.chain(netg_a2b.parameters(), netg_b2a.parameters()), lr=opt.lr,
                                   betas=(0.5, 0.999))
    optimizer_d_a = torch.optim.Adam(netd_a.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_d_b = torch.optim.Adam(netd_b.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g,
                                                       lr_lambda=LambdaLR(opt.n_epochs,
                                                                          opt.epoch,
                                                                          opt.decay_epoch).step)
    lr_scheduler_d_a = torch.optim.lr_scheduler.LambdaLR(optimizer_d_a,
                                                         lr_lambda=LambdaLR(opt.n_epochs,
                                                                            opt.epoch,
                                                                            opt.decay_epoch).step)
    lr_scheduler_d_b = torch.optim.lr_scheduler.LambdaLR(optimizer_d_b,
                                                         lr_lambda=LambdaLR(opt.n_epochs,
                                                                            opt.epoch,
                                                                            opt.decay_epoch).step)

    tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_a = tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_b = tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    target_real = Variable(tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    fake_a_buffer = ReplayBuffer()
    fake_b_buffer = ReplayBuffer()

    transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
                   transforms.RandomCrop(opt.size),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

    logger = Logger(opt.n_epochs, len(dataloader))

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            real_a = Variable(input_a.copy_(batch['A']))
            real_b = Variable(input_b.copy_(batch['B']))

            optimizer_g.zero_grad()

            same_b = netg_a2b(real_b)
            loss_identity_b = criterion_identity(same_b, real_b) * 5.0

            same_a = netg_b2a(real_a)
            loss_identity_a = criterion_identity(same_a, real_a) * 5.0

            fake_b = netg_a2b(real_a)
            pred_fake = netd_b(fake_b)
            loss_gan_a2b = criterion_gan(pred_fake, target_real)

            fake_a = netg_b2a(real_b)
            pred_fake = netd_a(fake_a)
            loss_gan_b2a = criterion_gan(pred_fake, target_real)

            recovered_a = netg_b2a(fake_b)
            loss_cycle_aba = criterion_cycle(recovered_a, real_a) * 10.0

            recovered_b = netg_a2b(fake_a)
            loss_cycle_bab = criterion_cycle(recovered_b, real_b) * 10.0

            loss_g = loss_identity_a + loss_identity_b + loss_gan_a2b + loss_gan_b2a + loss_cycle_aba + loss_cycle_bab
            loss_g.backward()

            optimizer_g.step()

            optimizer_d_a.zero_grad()

            pred_real = netd_a(real_a)
            loss_d_real = criterion_gan(pred_real, target_real)

            fake_a = fake_a_buffer.push_and_pop(fake_a)
            pred_fake = netd_a(fake_a.detach())
            loss_d_fake = criterion_gan(pred_fake, target_fake)

            loss_d_a = (loss_d_real + loss_d_fake) * 0.5
            loss_d_a.backward()

            optimizer_d_a.step()

            optimizer_d_b.zero_grad()

            pred_real = netd_b(real_b)
            loss_d_real = criterion_gan(pred_real, target_real)

            fake_b = fake_b_buffer.push_and_pop(fake_b)
            pred_fake = netd_b(fake_b.detach())
            loss_d_fake = criterion_gan(pred_fake, target_fake)

            loss_d_b = (loss_d_real + loss_d_fake) * 0.5
            loss_d_b.backward()

            optimizer_d_b.step()

            logger.log({'loss_g': loss_g, 'loss_g_identity': (loss_identity_a + loss_identity_b),
                        'loss_g_gan': (loss_gan_a2b + loss_gan_b2a),
                        'loss_g_cycle': (loss_cycle_aba + loss_cycle_bab), 'loss_d': (loss_d_a + loss_d_b)})

        lr_scheduler_g.step()
        lr_scheduler_d_a.step()
        lr_scheduler_d_b.step()

        torch.save(netg_a2b.state_dict(), '/output/netg_a2b.pth')
        torch.save(netg_b2a.state_dict(), '/output/netg_b2a.pth')
        torch.save(netd_a.state_dict(), '/output/netd_a.pth')
        torch.save(netd_b.state_dict(), '/output/netd_b.pth')


def test(opt):
    netg_a2b = Generator(opt.input_nc, opt.output_nc)
    netg_b2a = Generator(opt.output_nc, opt.input_nc)

    if opt.cuda:
        netg_a2b.cuda()
        netg_b2a.cuda()

    netg_a2b.load_state_dict(torch.load('/output/netg_A2B.pth'))
    netg_b2a.load_state_dict(torch.load('/output/netg_B2A.pth'))

    netg_a2b.eval()
    netg_b2a.eval()

    tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_a = tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_b = tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    transforms_ = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), batch_size=opt.batchSize,
                            shuffle=False, num_workers=opt.n_cpu)

    if not os.path.exists('/output/A'):
        os.makedirs('/output/A')
    if not os.path.exists('/output/B'):
        os.makedirs('/output/B')

    for i, batch in enumerate(dataloader):
        real_a = Variable(input_a.copy_(batch['A']))
        real_b = Variable(input_b.copy_(batch['B']))

        fake_b = 0.5*(netg_a2b(real_a).data + 1.0)
        fake_a = 0.5*(netg_b2a(real_b).data + 1.0)

        save_image(fake_a, '/output/A/%04d.png' % (i+1))
        save_image(fake_b, '/output/B/%04d.png' % (i+1))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

    sys.stdout.write('\n')


if __name__ == "__main__":
    main_opt = Options()
    main_opt.mode = int(os.getenv('OPTIONS_MODE', 0))
    main_opt.epoch = int(os.getenv('OPTIONS_EPOCH', 0))
    main_opt.n_epochs = int(os.getenv('OPTIONS_N_EPOCHS', 200))
    main_opt.batchSize = int(os.getenv('OPTIONS_BATCHSIZE', 1))
    main_opt.dataroot = os.getenv('OPTIONS_MODE', '/dataset/')
    main_opt.lr = float(os.getenv('OPTIONS_LEARNING_RATE', 0.0002))
    main_opt.decay_epoch = int(os.getenv('OPTIONS_DECAY_EPOCH', 100))
    main_opt.size = int(os.getenv('OPTIONS_CROP_SIZE', 256))
    main_opt.input_nc = int(os.getenv('OPTIONS_INPUT_NC', 3))
    main_opt.output_nc = int(os.getenv('OPTIONS_OUTPUT_NC', 3))
    main_opt.cuda = bool(os.getenv('OPTIONS_MODE', True))
    main_opt.n_cpu = int(os.getenv('OPTIONS_N_CPU', 8))

    if torch.cuda.is_available() and not main_opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with CUDA=True")

    if main_opt.mode == 0:
        train(main_opt)
    elif main_opt.mode == 1:
        test(main_opt)
    else:
        print("ERROR: Unknown mode, exiting")
