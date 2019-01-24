__author__ = 'Wendong Xu'
import os
from torch.utils.data import Dataset
import PIL.Image as Image
from scipy import misc
import pickle
import re
import json


# TODO: modify
food_pickle_path = '../../../food_w2c/'
data_root_path = '../../../food_w2c/'
img_root_path = '../../../food-1000/train/'
filename_ptn = re.compile('(.+)\..+')


json_id_img_path = '/home/kirai_wendong/proj/food-1000/introduction/layer2.json'


def get_image_filepath(data_root_path, img_filename):
  data_root_path = os.path.join(data_root_path, img_filename[0])
  data_root_path = os.path.join(data_root_path, img_filename[1])
  data_root_path = os.path.join(data_root_path, img_filename[2])
  data_root_path = os.path.join(data_root_path, img_filename[3])
  data_root_path = os.path.join(data_root_path, img_filename)
  data_root_path += '.jpg'
  return data_root_path


def get_img_to_id():
  label_img = []
  with open(json_id_img_path, 'r') as fp:
    js = json.load(fp)
    for line in js:
      label_id = line['id']
      if not os.path.exists(os.path.join(data_root_path, label_id+'.pkl')):
        # print(os.path.join(data_root_path, label_id+'.pkl'))
        # print('pkl {} not find. skip.'.format(label_id))
        continue
      # print(os.path.join(data_root_path, label_id+'.pkl'))
      for img in line['images']:
        if not os.path.exists(get_image_filepath(img_root_path, img['id'][:-4])):
          continue
        label_img.append([img['id'][:-4], line['id']])
  return label_img


# label_img = get_img_to_id()
# with open('./label_img.pkl', 'wb') as fp:
#   pickle.dump(label_img, fp)
with open('./label_img.pkl', 'rb') as fp:
  label_img = pickle.load(fp)


def get_filename_picklefile(file_root_path):
  for rt, dirs, files in os.walk(file_root_path):
    ret = []
    for each in files:
      try:
        tmp = filename_ptn.findall(each)[0]
      except:
        print('file {} don\'t have this file. skip'.format(each))
      ret.append([tmp, os.path.join(file_root_path, each)])
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


class FoodDataset(Dataset):
  def __init__(self, transform=None, target_transform=None):
    self.food_list = label_img
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    # print(index, len(self.food_list))
    image_path = get_image_filepath(img_root_path, self.food_list[index][0])
    image = img_resize(image_path)
    if self.transform is not None:
      image = self.transform(image)
    # seq_vector, image
    path = os.path.join(food_pickle_path, self.food_list[index][1]+'.pkl')
    w2v = pickle.load(open(path, 'rb'), encoding='iso-8859-1')
    # print(w2v.shape)
    return w2v.astype('float32'), image

  def __len__(self):
    return len(self.food_list)


if __name__ == '__main__':
  # vec = [['0000dfbc52.jpg', [1,2,3,4,5,6,7,8,9]]]
  # with open('./det_ingrs.dat', 'wb') as fp:
  #   pickle.dump(vec, fp)
  x = get_food_with_tag(food_pickle_path)
  x = img_resize(get_image_filepath(data_root_path, x[0][0]))
