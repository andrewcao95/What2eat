import argparse
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import os
from networks.generator import Generator
from networks.discriminator import Discriminator
import utils
import numpy as np
from sklearn import preprocessing
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="")
parser.add_argument('--food_tag_dat_path', type=str, default='../../', help='food with tag\'s list path')
parser.add_argument('--tmp_path', type=str, default='../../data/training_temp/', help='path of the intermediate files during training')
parser.add_argument('--model_dump_path', type=str, default='../../resource/gan_models', help='model\'s save path')

opt = parser.parse_args()
tmp_path= opt.tmp_path
model_dump_path = opt.model_dump_path


def load_checkpoint(model_dir):
  models_path = utils.read_newest_model(model_dir)
  if len(models_path) == 0:
    return None, None
  models_path.sort()
  new_model_path = os.path.join(model_dump_path, models_path[-1])
  checkpoint = torch.load(new_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
  return checkpoint, new_model_path


def generate(G, file_name, tags):
  '''
  Generate fake image.
  :param G:
  :param file_name:
  :param tags:
  :return: img's tensor and file path.
  '''
  # g_noise = Variable(torch.FloatTensor(1, 128)).to(device).data.normal_(.0, 1)
  # g_tag = Variable(torch.FloatTensor([utils.get_one_hot(tags)])).to(device)
  g_noise, g_tag = utils.fake_generator(1, 128, device)

  img = G(torch.cat([g_noise, g_tag], dim=1))
  vutils.save_image(img.data.view(1, 3, 128, 128),
                    os.path.join(tmp_path, '{}.png'.format(file_name)))
  print('Saved file in {}'.format(os.path.join(tmp_path, '{}.png'.format(file_name))))
  return img.data.view(1, 3, 128, 128), os.path.join(tmp_path, '{}.png'.format(file_name))


grad_list = []


def get_grad(module, inputdata, output):
  grad_list.append(output)


def image_backward_D(G, D):
  import matplotlib.pyplot as plt
  D.reduce_block_1.conv_1.register_backward_hook(get_grad)
  # g_noise, g_tag = utils.fake_generator(1, 128, device)
  # img = G(torch.cat([g_noise, g_tag], dim=1))

  # Using a specific noise tensor to generator a specific image.
  test_noise = [[]]
  img = G(torch.Tensor(test_noise).to(device))

  t_img = vutils.make_grid(img.data.view(1, 3, 128, 128)).numpy()
  t_img = np.transpose(t_img, (1, 2, 0))
  t_img[t_img<0] = 0
  # min_max_scaler = preprocessing.MinMaxScaler()
  # t_img[..., 0] = min_max_scaler.fit_transform(t_img[..., 0])
  # t_img[..., 1] = min_max_scaler.fit_transform(t_img[..., 1])
  # t_img[..., 2] = min_max_scaler.fit_transform(t_img[..., 2])
  plt.imshow(t_img)
  plt.show()
  label_p, tag_p = D(img)
  label = Variable(torch.FloatTensor(1, 1.0)).to(device)
  lbl_criterion = nn.BCEWithLogitsLoss().to(device)
  loss = lbl_criterion(label_p, label) * 10000
  loss.backward()
  print(grad_list[0][0].shape)



if __name__ == '__main__':
  checkpoint, _ = load_checkpoint(model_dump_path)

  G = Generator().to(device)
  G.load_state_dict(checkpoint['G'])

  D = Discriminator().to(device)
  D.load_state_dict(checkpoint['D'])

  # generate(G, 'test', ['white hair'])
  image_backward_D(G, D)