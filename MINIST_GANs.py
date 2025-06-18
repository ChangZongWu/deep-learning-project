import torch
import fastai
from fastcore.xtras import image_size
from pydantic import Discriminator
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from IPython.display import Image
from torchvision.utils import save_image
import cv2
from IPython.display import FileLink
import os
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mnist = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose([
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])
    )
    def denorm(x):
        out = (x + 1) / 2
        return out.clamp(0, 1)
    batch_size = 100
    data_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
    image_size = 784
    hidden_size = 256
    Discriminator = nn.Sequential(
        nn.Linear(image_size, hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, 1),
        nn.Sigmoid())
    Discriminator.to(device)
    latent_size = 64
    Generator = nn.Sequential(
        nn.Linear(latent_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, image_size),
        nn.Tanh())

    y = Generator(torch.randn(2, latent_size))
    gen_imgs = denorm(y.reshape(-1, 28, 28)).detach()
    Generator.to(device)
    criterion = nn.BCELoss()
    d_optim = torch.optim.Adam(Discriminator.parameters(), lr=0.0002)
    g_optim = torch.optim.Adam(Generator.parameters(), lr=0.0002)
    def reset_grad():
        d_optim.zero_grad()
        g_optim.zero_grad()
    def train_discriminator(images):
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        outputs = Discriminator(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = Generator(z)
        outputs = Discriminator(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optim.step()
        return d_loss, real_score, fake_score

    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = Generator(z)
    Discriminator(fake_images)
    def train_generator():
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = Generator(z)
        labels = torch.ones(batch_size, 1).to(device)
        outputs = Discriminator(fake_images)
        g_loss = criterion(outputs, labels)
        reset_grad()
        g_loss.backward()
        g_optim.step()
        return g_loss, fake_images
    sample_dir = "samples"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    for images, _ in data_loader:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'), nrow=10)
        break

    sample_vectors = torch.randn(batch_size,latent_size).to(device)
    def save_fake_images(index):
        fake_images = Generator(sample_vectors)
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
        save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)
    save_fake_images(0)

    num_epochs = 500
    total_step = len(data_loader)
    d_losses, g_losses, real_scores, fake_scores = [], [], [], []
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            images = images.reshape(batch_size,-1).to(device)
            d_loss, real_score, fake_score = train_discriminator(images)
            g_loss, fake_images = train_generator()
            if (i+1) % 200 == 0:
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                real_scores.append(real_score.mean().item())
                fake_scores.append(fake_score.mean().item())
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'.format(epoch+1, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item()))
        save_fake_images(epoch+1)
    # vid_fname = 'mnist_gan.avi'
    # files = [os.path.join(sample_dir,f) for f in os.listdir(sample_dir) if 'fake_images' in f]
    # files.sort()
    # out = cv2.VideoWriter(vid_fname, cv2.VideoWriter_fourcc(*'MP4V'), 8, (302, 302))
    # [out.write(cv2.imread(fname)) for fname in files]
    # out.release()
    # FileLink('gan_mnist.avi')
    # Change this section at the end of your code
    vid_fname = 'mnist_gan.avi'  # Make sure filename is consistent
    files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'fake_images' in f]
    files.sort()

    # Read first image to get dimensions
    first_img = cv2.imread(files[0])
    height, width, layers = first_img.shape

    # Use XVID codec instead of MP4V for better compatibility with AVI
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(vid_fname, fourcc, 8, (width, height))

    # Write images to video
    for fname in files:
        out.write(cv2.imread(fname))

    out.release()
    FileLink(vid_fname)  # Use the same filename defined above

