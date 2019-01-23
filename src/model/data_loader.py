__author__ = 'Wendong Xu'
import os
from torch.utils.data import Dataset
import PIL.Image as Image
from scipy import misc
import pickle
import re


# TODO: modify
food_pickle_path = '../../data/det_ingrs.dat'
data_root_path = '../../data/train/'
filename_ptn = re.compile('(.+)\..+')

def get_filename_picklefile(file_root_path):
  for rt, dirs, files in os.walk(file_root_path):
    ret = []
    for each in files:
      try:
        tmp = filename_ptn.findall(each)[0]
      except:
        print('file {} don\'t have this file. skip'.format(each))
      with open(os.path.join(file_root_path, each), 'rb') as fp:
        ret.append([tmp, pickle.load(fp)])
    return ret


def get_food_with_tag(food_pickle_path):
  # image name, seq_vector
  with open(food_pickle_path, 'rb') as fp:
    food_list = pickle.load(fp)
    return food_list


def img_resize(img_path, aim_size=128):
  img = misc.imread(img_path)
  img = misc.imresize(img, [aim_size, aim_size, 3])
  return img


def get_image_filepath(data_root_path, img_filename):
  data_root_path = os.path.join(data_root_path, img_filename[0])
  data_root_path = os.path.join(data_root_path, img_filename[1])
  data_root_path = os.path.join(data_root_path, img_filename[2])
  data_root_path = os.path.join(data_root_path, img_filename[3])
  data_root_path = os.path.join(data_root_path, img_filename)
  return data_root_path


class FoodDataset(Dataset):
  def __init__(self, transform=None, target_transform=None):
    self.food_list = get_filename_picklefile(data_root_path)
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    image_path = get_image_filepath(data_root_path, self.food_list[index][0])
    image = img_resize(image_path)
    if self.transform is not None:
      image = self.transform(image)
    # seq_vector, image
    return self.food_list[index][1], image

  def __len__(self):
    return len(self.food_list)


if __name__ == '__main__':
  # vec = [['0000dfbc52.jpg', [1,2,3,4,5,6,7,8,9]]]
  # with open('./det_ingrs.dat', 'wb') as fp:
  #   pickle.dump(vec, fp)
  x = get_food_with_tag(food_pickle_path)
  x = img_resize(get_image_filepath(data_root_path, x[0][0]))
