class Options:
    mode = 0
    epoch = 0
    n_epochs = 200
    batchSize = 1
    dataroot = '/dataset/'
    lr = 0.0002
    decay_epoch = 100
    size = 256
    input_nc = 3
    output_nc = 3
    cuda = True
    n_cpu = 8
    generator_A2B = '/output/netG_A2B.pth'
    generator_B2A = '/output/netG_B2A.pth'
