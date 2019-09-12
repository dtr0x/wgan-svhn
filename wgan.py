import numpy as np
import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
import classify_svhn
from generator import Generator

NOISE_DIM = 100
CONVT_SIZE = 128
LAMBDA = 10
DISC_ITERS = 5
BATCH_SIZE = 64
NUM_EPOCHS = 100

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, CONVT_SIZE, 3, 2, padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(CONVT_SIZE, 2 * CONVT_SIZE, 3, 2, padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * CONVT_SIZE, 4 * CONVT_SIZE, 3, 2, padding = 1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(64 * CONVT_SIZE, 1)

    def forward(self, input):
        out = self.main(input)
        out = out.view(-1, 64 * CONVT_SIZE)
        out = self.linear(out)
        return out

G = Generator(NOISE_DIM)
D = Discriminator()

# optimizers
adam_d = optim.Adam(D.parameters(), lr=1e-4)
adam_g = optim.Adam(G.parameters(), lr=1e-4)

# set gpu flag
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
if use_cuda:
    D = D.cuda(gpu)
    G = G.cuda(gpu)

# values to set gradient sign
one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

# calculates mean of gradient penalty over a batch
def grad_penalty(D, real_data, fake_data):
    dim = real_data.size(0)
    alpha = torch.rand(dim, 1)
    alpha = alpha.expand(dim, int(real_data.nelement()/dim)).contiguous().view(dim, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    z_values = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        z_values = z_values.cuda(gpu)
    z_values = autograd.Variable(z_values, requires_grad=True)

    d_interp = D(z_values)

    gradients = autograd.grad(outputs=d_interp, inputs=z_values,
                              grad_outputs=torch.ones(d_interp.size()).cuda(gpu) if use_cuda else torch.ones(
                                  d_interp.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# generate grid of sample images
def gen_samples(i, G):
    noise = torch.randn(64, NOISE_DIM)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    samples = G(noisev)
    samples = samples.view(-1, 3, 32, 32)
    torchvision.utils.save_image(samples, './gan_data/samples/sample_{}.png'.format(i), nrow=8, padding=2)

# Dataset iterator
directory = "svhn/"
train_loader, valid_loader, test_loader = classify_svhn.get_data_loader(directory, BATCH_SIZE)

for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(train_loader):
        if i == 0 or i % (DISC_ITERS + 1): # update discriminator
            D.zero_grad()
            real_data, _ = data
            dim = real_data.size(0)

            if use_cuda:
                real_data = real_data.cuda(gpu)
            real_data = autograd.Variable(real_data)

            D_real = D(real_data)
            D_real = D_real.mean()
            D_real.backward(mone)

            # train with fake
            noise = torch.randn(dim, NOISE_DIM)
            if use_cuda:
                noise = noise.cuda(gpu)
            noise = autograd.Variable(noise)
            fake = autograd.Variable(G(noise).data)
            D_fake = D(fake)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # train with gradient penalty
            gp = grad_penalty(D, real_data.data, fake.data)
            gp.backward()

            D_loss = D_fake - D_real + gp
            adam_d.step()
        else: # update generator
            G.zero_grad()

            noise = torch.randn(BATCH_SIZE, NOISE_DIM)
            if use_cuda:
                noise = noise.cuda(gpu)
            noise = autograd.Variable(noise)
            fake = G(noise)
            D_fake = D(fake)
            D_fake = D_fake.mean()
            D_fake.backward(mone)
            G_loss = -D_fake
            adam_g.step()

    gen_samples(epoch, G)

# save model
torch.save(G.state_dict(), "./gan_data/model/wgan.pt")
