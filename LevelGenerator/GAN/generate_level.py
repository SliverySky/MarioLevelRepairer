# This generator program expands a low-dimentional latent vector into a 2D array of Tiles.
# Each line of input should be an array of z vectors (which are themselves arrays of floats -1 to 1)
# Each line of output is an array of 32 levels (which are arrays-of-arrays of integer tile ids)

import torch
from torch.autograd import Variable

import numpy as np
from LevelGenerator.GAN.dcgan import Generator
# import matplotlib.pyplot as plt
from utils.level_process import *
from utils.visualization import *
from root import rootpath


def getLevel(noise, to_string, name, size):
    model_to_load = name
    batch_size = 1
    image_size = 32 * size
    ngf = 64
    nz = 32
    z_dims = 10  # number different titles
    generator = Generator(nz, ngf, image_size, z_dims)
    generator.load_state_dict(torch.load(model_to_load, map_location=lambda storage, loc: storage))
    latent_vector = torch.FloatTensor(noise).view(batch_size, nz, 1, 1)
    with torch.no_grad():
        levels = generator(Variable(latent_vector))
    im = levels.data.cpu().numpy()
    im = np.argmax(im, axis=1)
    im = littleLevel(im[0], size)
    if to_string:
        return arr_to_str(im[0:14, 0:28])
    else:
        return im[0:14, 0:28]


if __name__ == '__main__':
    lv = getLevel(np.random.randn(1, 32), False, './generator.pth', 1)
    saveLevelAsImage(lv, rootpath + '\lv0')
