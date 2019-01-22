__author__ = 'Wendong Xu'
import os
from torch.utils.data import Dataset
import PIL.Image as Image
import pickle


# TODO: modify
food_npz_path = '../../resource/det_ingrs.npy'
data_root_path = '../data/'

def get_food_with_tag(food_npz_path):
  # image name, seq_vector
  food_list = np.load(food_npz_path)
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
  os.path.join(data_root_path, img_filename)
  return data_root_path


class FoodDataset(Dataset):
  def __init__(self, transform=None, target_transform=None):
    self.food_list = get_food_with_tag()
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
  food_tag_dat_path = '../../resource/avatar_with_tag.dat'
  FoodDataset(food_tag_dat_path)
