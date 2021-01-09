import torch
import torch.nn as nn
import torch.nn.parallel

'''
some code are copied from https://github.com/TheHedgeify/DagstuhlGAN
'''

class Generator(nn.Module):
    def __init__(self, ns_size, final_depth, out_size, out_depth, n_add=0, ngpu=1):
        # ns_size: length of noise vector

        # final_depth: depth of last hidden layer
        # out_size； objective size of the output matrix
        # out_depth: depth of output
        # n_add: number of additional stride-convolution layers

        super(Generator, self).__init__()
        self.ngpu = ngpu
        assert out_size % 16 == 0, 'output size must be divided by 16'
        main = nn.Sequential()

        # Projection_layer: a transpose convolution layer to project and reshape for noise vector(1*1*ns_size).
        # Layer scale: 4 * 4 * l0_depth
        # proNet = nn.ConvTranspose2d(ns_size, l0_depth, 4, 1, 0, bias=False)
        # Compute depth of first hidden layer
        tmp = 4
        first_depth = final_depth
        while tmp < out_size//2:
            tmp *= 2
            first_depth *= 2

        main.add_module('pro:Net', nn.ConvTranspose2d(ns_size, first_depth, 4, 1, 0, bias=False))
        # print('projection layer size:' + proNet.size())
        main.add_module('pro:BN', nn.BatchNorm2d(first_depth))
        main.add_module('pro:AcFunc', nn.ReLU(True))

        # Transpose convolution layers to expand to the objective size.
        prev_depth = first_depth
        prev_size = 4
        cnt = 1
        while prev_size < out_size//2:
            main.add_module('exp%d:Net' % cnt,
                            nn.ConvTranspose2d(prev_depth, prev_depth//2, 4, 2, 1, bias=False))
            main.add_module('exp%d:BN' % cnt, nn.BatchNorm2d(prev_depth//2))
            main.add_module('exp%d:AcFunc' % cnt, nn.ReLU(True))
            cnt += 1
            prev_depth //= 2
            prev_size *= 2
        # Additional stride-convolution layers
        for i in range(n_add):
            main.add_module('add%d:Net' % (i+1),
                            nn.Conv2d(prev_depth, prev_depth, 3, 1, 1, bias=False))
            main.add_module('add%d:BN' % (i+1), nn.BatchNorm2d(prev_depth // 2))
            main.add_module('add%d:AcFunc' % (i+1), nn.ReLU(True))

        # Output layer which use transpose convolution
        main.add_module('out:Net', nn.ConvTranspose2d(prev_depth, out_depth, 4, 2, 1, bias=False))
        main.add_module('out:AcFunc', nn.ReLU())

        self.main = main

    def forward(self, inpt):
        if isinstance(inpt.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inpt, range(self.ngpu))
        else:
            output = self.main(inpt)
        return output
class Discriminator(nn.Module):
    def __init__(self, in_size, in_depth, first_depth, n_add=0, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        assert in_size % 16 == 0, "isize has to be a multiple of 16"
        main = nn.Sequential()


        # First layer, without BN
        main.add_module('init:Net', nn.Conv2d(in_depth, first_depth, 4, 2, 1, bias=False))
        main.add_module('init:AcFunc', nn.LeakyReLU(0.2))

        # Additional layers
        for i in range(n_add):
            main.add_module('add%d:Net' % (i+1), nn.Conv2d(in_depth, in_depth, 3, 1, 1, bias=False))
            main.add_module('add%d:BN' % (i+1), nn.BatchNorm2d(in_depth))
            main.add_module('add%d:AcFunc' % (i+1), nn.LeakyReLU(0.2))

        # Shrink layers
        prev_size = in_size
        prev_depth = first_depth
        cnt = 1
        while prev_size > 4*2:
            main.add_module('shrk:Net%d' % cnt, nn.Conv2d(prev_depth, prev_depth*2, 4, 2, 1, bias=False))
            main.add_module('shrk:BN%d' % cnt, nn.BatchNorm2d(prev_depth*2))
            main.add_module('shrk:AcFunc%d' % cnt, nn.LeakyReLU(0.2))
            prev_depth *= 2
            prev_size //= 2
            cnt += 1
        # Last layer
        main.add_module('out:Net', nn.Conv2d(prev_depth, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, inpt):
        if isinstance(inpt.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inpt, range(self.ngpu))
        else:
            output = self.main(inpt)
        output = output.mean(0)
        return output.view(1)
        # egin_vector=[0.]*inpt.shape[0]
        # cnt=0
        # for i in range(inpt.shape[0]):
            # tmp=judge_level(inpt[i])

            # cnt=cnt+tmp
            # if(tmp>0):egin_vector[i]=0.5
            # else:egin_vector[i]=1
        # print(cnt)
        # output_new = output[0] * egin_vector[0]
        # for i in range(1, len(egin_vector)):
        #     output_new = output_new + output[i] * egin_vector[i]
        # return output_new.view(1)


class Generator_nobn(nn.Module):
    def __init__(self, ns_size, final_depth, out_size, out_depth, n_add=0, ngpu=1):
        # ns_size: length of noise vector
        # l0_depth: length and width of layer0
        # final_depth: depth of last hidden layer
        # out_size； objective size of the output matrix
        # n_add: number of additional stride-convolution layers

        super(Generator, self).__init__()
        self.ngpu = ngpu
        assert out_size % 16 == 0, 'output size must be divided by 16'
        main = nn.Sequential()

        # Projection_layer: a transpose convolution layer to project and reshape for noise vector(1*1*ns_size).
        # Layer scale: 4 * 4 * l0_depth
        # proNet = nn.ConvTranspose2d(ns_size, l0_depth, 4, 1, 0, bias=False)
        # Compute depth of first hidden layer
        tmp = 4
        first_depth = final_depth
        while tmp < out_size // 2:
            tmp *= 2
            first_depth *= 2

        main.add_module('pro:Net', nn.ConvTranspose2d(ns_size, first_depth, 4, 1, 0, bias=False))
        # print('projection layer size:' + proNet.size())
        main.add_module('pro:AcFunc', nn.ReLU(True))

        # Transpose convolution layers to expand to the objective size.
        prev_depth = first_depth
        prev_size = 4
        cnt = 1
        while prev_size < out_size // 2:
            main.add_module('exp%d:Net' % cnt,
                            nn.ConvTranspose2d(prev_depth, prev_depth // 2, 4, 2, 1, bias=False))
            main.add_module('exp%d:AcFunc' % cnt, nn.ReLU(True))
            cnt += 1
            prev_depth //= 2
            prev_size *= 2

        # Additional stride-convolution layers
        for i in range(n_add):
            main.add_module('add%d:Net' % (i + 1),
                            nn.Conv2d(prev_depth, prev_depth, 3, 1, 1, bias=False))
            main.add_module('add%d:AcFunc' % (i + 1), nn.ReLU(True))

        # Output layer which use transpose convolution
        main.add_module('out:Net', nn.ConvTranspose2d(prev_depth, out_depth, 4, 2, 1, bias=False))
        main.add_module('out:AcFunc', nn.ReLU())

        self.main = main

    def forward(self, inpt):
        if isinstance(inpt.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inpt, range(self.ngpu))
        else:
            output = self.main(inpt)
        return output


class Discriminator_nobn(nn.Module):
    def __init__(self, in_size, in_depth, first_depth, n_add=0, ngpu=1):
        super(Discriminator_nobn, self).__init__()
        self.ngpu = ngpu
        assert in_size % 16 == 0, "isize has to be a multiple of 16"
        main = nn.Sequential()

        # First layer, without BN
        main.add_module('init:Net', nn.Conv2d(in_depth, first_depth, 4, 2, 1, bias=False))
        main.add_module('init:AcFunc', nn.LeakyReLU(0.2))

        # Additional layers
        for i in range(n_add):
            main.add_module('add%d:Net' % (i+1), nn.Conv2d(in_depth, in_depth, 3, 1, 1, bias=False))
            # main.add_module('add%d:BN' % (i+1), nn.BatchNorm2d(in_depth))
            main.add_module('add%d:AcFunc' % (i+1), nn.LeakyReLU(0.2))

        # Shrink layers
        prev_size = in_size
        prev_depth = first_depth
        cnt = 1
        while prev_size > 4*2:
            main.add_module('pyramid:Net%d' % cnt, nn.Conv2d(prev_depth, prev_depth*2, 4, 2, 1, bias=False))
            # main.add_module('init:BN%d', nn.BatchNorm2d(prev_depth*2))
            main.add_module('pyramid:AcFunc%d' % cnt, nn.LeakyReLU(0.2))
            prev_depth *= 2
            prev_size //= 2
            cnt += 1

        # Last layer
        main.add_module('out:Net', nn.Conv2d(prev_depth, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, inpt):
        if isinstance(inpt.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inpt, range(self.ngpu))
        else:
            output = self.main(inpt)
        output = output.mean(0)
        return output.view(1)

























