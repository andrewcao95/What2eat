__author__ = 'Wendong Xu'
'''
Training strategy of the `DRAGAN-like SRGAN`.
Have some difference in loss calculating.
I weighted label's loss and tag's loss with half of lambda_adv.
The label_criterion was also different.
'''
import argparse
from networks.generator import Generator
from networks.discriminator import Discriminator
from data_loader import FoodDataset
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.autograd import Variable, grad
import utils
import os
import torchvision.utils as vutils
import logging
import time
import torch.nn.functional as F


__DEBUG__ = True

# have GPU or not.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet-GAN")
# TODO:
parser.add_argument('--food_tag_dat_path', type=str, default='/home/kirai_wendong/proj/food-1000/ingredient/det_ingrs.dat', help='avatar with tag\'s list path')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.5, help='adam optimizer\'s paramenter')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size for each epoch')
parser.add_argument('--lr_update_cycle', type=int, default=50000, help='cycle of updating learning rate')
parser.add_argument('--max_epoch', type=int, default=500, help='training epoch')
parser.add_argument('--num_workers', type=int, default=0, help='number of data loader processors')
parser.add_argument('--noise_size', type=int, default=128, help='number of G\'s input')
parser.add_argument('--lambda_adv', type=float, default=34.0, help='adv\'s lambda')
parser.add_argument('--lambda_gp', type=float, default=0.5, help='gp\'s lambda')
# TODO:
parser.add_argument('--model_dump_path', type=str, default='../../models', help='model\'s save path')
parser.add_argument('--verbose', type=bool, default=True, help='output verbose messages')
# TODO:
parser.add_argument('--tmp_path', type=str, default='../../training_temp/', help='path of the intermediate files during training')
parser.add_argument('--verbose_T', type=int, default=50, help='steps for saving intermediate file')
# TODO:
parser.add_argument('--logfile', type=str, default='../../training.log', help='logging path')


##########################################
# Load params
#
opt = parser.parse_args()
food_tag_dat_path = opt.food_tag_dat_path
learning_rate = opt.learning_rate
beta_1 = opt.beta_1
batch_size= opt.batch_size
lr_update_cycle = opt.lr_update_cycle
max_epoch = opt.max_epoch
num_workers= opt.num_workers
noise_size = opt.noise_size
lambda_adv = opt.lambda_adv
lambda_gp = opt.lambda_gp
model_dump_path = opt.model_dump_path
verbose = opt.verbose
tmp_path= opt.tmp_path
verbose_T = opt.verbose_T
logfile = opt.logfile

logger = logging.getLogger()
logger.setLevel(logging.INFO)
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log = logging.FileHandler(logfile, mode='w+')
log.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
log.setFormatter(formatter)

plog = logging.StreamHandler()
plog.setLevel(logging.INFO)
plog.setFormatter(formatter)

logger.addHandler(log)
logger.addHandler(plog)

logger.info('Currently use {} for calculating'.format(device))
if __DEBUG__:
  batch_size = 16
  num_workers = 0
#
#
##########################################


tag_size = 4500

def initital_network_weights(element):
  if hasattr(element, 'weight'):
    element.weight.data.normal_(.0, .02)


def adjust_learning_rate(optimizer, iteration):
  lr = learning_rate * (0.1 ** (iteration // lr_update_cycle))
  return lr


class SRGAN():
  def __init__(self):
    logger.info('Set Data Loader')
    self.dataset = FoodDataset(transform=transforms.Compose([ToTensor()]))
    self.data_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers, drop_last=True)
    checkpoint, checkpoint_name = self.load_checkpoint(model_dump_path)
    if checkpoint == None:
      logger.info('Don\'t have pre-trained model. Ignore loading model process.')
      logger.info('Set Generator and Discriminator')
      self.G = Generator(tag=tag_size).to(device)
      self.D = Discriminator(tag=tag_size).to(device)
      logger.info('Initialize Weights')
      self.G.apply(initital_network_weights).to(device)
      self.D.apply(initital_network_weights).to(device)
      logger.info('Set Optimizers')
      self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
      self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
      self.epoch = 0
    else:
      logger.info('Load Generator and Discriminator')
      self.G = Generator(tag=tag_size).to(device)
      self.D = Discriminator(tag=tag_size).to(device)
      logger.info('Load Pre-Trained Weights From Checkpoint'.format(checkpoint_name))
      self.G.load_state_dict(checkpoint['G'])
      self.D.load_state_dict(checkpoint['D'])
      logger.info('Load Optimizers')
      self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
      self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
      self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
      self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
      self.epoch = checkpoint['epoch']
    logger.info('Set Criterion')
    # self.label_criterion = nn.BCEWithLogitsLoss().to(device)
    # self.tag_criterion = nn.BCEWithLogitsLoss().to(device)


  def load_checkpoint(self, model_dir):
    models_path = utils.read_newest_model(model_dir)
    if len(models_path) == 0:
      return None, None
    models_path.sort()
    new_model_path = os.path.join(model_dump_path, models_path[-1])
    if torch.cuda.is_available():
      checkpoint = torch.load(new_model_path)
    else:
      checkpoint = torch.load(new_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    return checkpoint, new_model_path


  def train(self):
    iteration = -1
    label = Variable(torch.FloatTensor(batch_size, 1)).to(device)
    logging.info('Current epoch: {}. Max epoch: {}.'.format(self.epoch, max_epoch))
    while self.epoch <= max_epoch:
      msg = {}
      adjust_learning_rate(self.optimizer_G, iteration)
      adjust_learning_rate(self.optimizer_D, iteration)
      for i, (avatar_tag, avatar_img) in enumerate(self.data_loader):
        iteration += 1
        if avatar_img.shape[0] != batch_size:
          logging.warn('Batch size not satisfied. Ignoring.')
          continue
        if verbose:
          if iteration % verbose_T == 0:
            msg['epoch'] = int(self.epoch)
            msg['step'] = int(i)
            msg['iteration'] = iteration
        avatar_img = Variable(avatar_img).to(device)
        # 1. Training D
        # 1.1. use really image for discriminating
        self.D.zero_grad()
        label_p = self.D(avatar_img)
        label.data.fill_(1.0)

        # 1.2. real image's loss
        # real_label_loss = self.label_criterion(label_p, label)
        real_label_loss = F.binary_cross_entropy(label_p, label)
        real_loss_sum = real_label_loss
        real_loss_sum.backward()
        if verbose:
          if iteration % verbose_T == 0:
            msg['discriminator real loss'] = float(real_loss_sum)

        # 1.3. use fake image for discriminating
        g_noise, fake_tag = utils.fake_generator(batch_size, noise_size, device)
        fake_feat = torch.cat([g_noise, fake_tag], dim=1)
        fake_img = self.G(fake_feat).detach()
        fake_label_p = self.D(fake_img)
        label.data.fill_(.0)

        # 1.4. fake image's loss
        # fake_label_loss = self.label_criterion(fake_label_p, label)
        fake_label_loss = F.binary_cross_entropy(fake_label_p, label)
        # TODO:
        fake_loss_sum = fake_label_loss
        fake_loss_sum.backward()
        if verbose:
          if iteration % verbose_T == 0:
            print('predicted fake label: {}'.format(fake_label_p))
            msg['discriminator fake loss'] = float(fake_loss_sum)

        # 1.6. update optimizer
        self.optimizer_D.step()

        # 2. Training G
        # 2.1. generate fake image
        self.G.zero_grad()
        g_noise, fake_tag = utils.fake_generator(batch_size, noise_size, device)
        fake_feat = torch.cat([g_noise, fake_tag], dim=1)
        fake_img = self.G(fake_feat)
        fake_label_p = self.D(fake_img)
        label.data.fill_(1.0)

        # 2.2. calc loss
        # label_loss_g = self.label_criterion(fake_label_p, label)
        label_loss_g = F.binary_cross_entropy(fake_label_p, label)
        loss_g = label_loss_g
        loss_g.backward()
        if verbose:
          if iteration % verbose_T == 0:
            msg['generator loss'] = float(loss_g)

        # 2.2. update optimizer
        self.optimizer_G.step()

        if verbose:
          if iteration % verbose_T == 0:
            logger.info('------------------------------------------')
            for key in msg.keys():
              logger.info('{} : {}'.format(key, msg[key]))
        # save intermediate file
        if iteration % 10000 == 0:
          torch.save({
            'epoch': self.epoch,
            'D': self.D.state_dict(),
            'G': self.G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
          },'{}/checkpoint_{}.tar'.format(model_dump_path, str(iteration).zfill(8)))
          logger.info('Checkpoint saved in: {}'.format('{}/checkpoint_{}.tar'.format(model_dump_path, str(iteration).zfill(8))))

        if iteration % verbose_T == 0:
          vutils.save_image(avatar_img.data.view(batch_size, 3, avatar_img.size(2), avatar_img.size(3)),
                            os.path.join(tmp_path, 'real_image_{}.png'.format(str(iteration).zfill(8))))
          g_noise, fake_tag = utils.fake_generator(batch_size, noise_size, device)
          fake_feat = torch.cat([g_noise, fake_tag], dim=1)
          fake_img = self.G(fake_feat)
          vutils.save_image(fake_img.data.view(batch_size, 3, avatar_img.size(2), avatar_img.size(3)),
                            os.path.join(tmp_path, 'fake_image_{}.png'.format(str(iteration).zfill(8))))
          logger.info('Saved intermediate file in {}'.format(os.path.join(tmp_path, 'fake_image_{}.png'.format(str(iteration).zfill(8)))))
      # dump checkpoint
      torch.save({
        'epoch': self.epoch,
        'D': self.D.state_dict(),
        'G': self.G.state_dict(),
        'optimizer_D': self.optimizer_D.state_dict(),
        'optimizer_G': self.optimizer_G.state_dict(),
      }, '{}/checkpoint_{}.tar'.format(model_dump_path, str(self.epoch).zfill(4)))
      logger.info('Checkpoint saved in: {}'.format('{}/checkpoint_{}.tar'.format(model_dump_path, str(self.epoch).zfill(4))))
      self.epoch += 1


if __name__ == '__main__':
  if not os.path.exists(model_dump_path):
    os.mkdir(model_dump_path)
  if not os.path.exists(tmp_path):
    os.mkdir(tmp_path)
  gan = SRGAN()
  gan.train()
