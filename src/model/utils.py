import os
import random
import torch
from torch.autograd import Variable
import pickle
tag_shape = 4500
food_pickle_path = '../../../food_w2c/'

with open('./label_img.pkl', 'rb') as fp:
  label_img = pickle.load(fp)

LO = 0
HI = len(label_img) - 1


def read_newest_model(model_dump_path):
  for rt, dirs, files in os.walk(model_dump_path):
    return files


def fake_tag():
  _tag = [0 for _ in range(tag_shape)]
  _tag = Variable(torch.FloatTensor(_tag)).view(1, -1)
  _tag.data.normal_(.0, 1)
  
  label_id = label_img[random.randint(LO, HI)][1]
  path = os.path.join(food_pickle_path, str(label_id) + '.pkl')
  w2v = pickle.load(open(path, 'rb'), encoding='iso-8859-1')
  _tag = Variable(torch.FloatTensor(w2v)).view(1, -1)
  return _tag


def fake_generator(batch_size, noise_size, device):
  noise = Variable(torch.FloatTensor(batch_size, noise_size)).to(device)
  noise.data.normal_(.0, 1)
  tag = torch.cat([fake_tag() for i in range(batch_size)], dim=0)
  tag = Variable(tag).to(device)
  return noise, tag


if __name__ == '__main__':
  pass
