__author__ = 'Wendong Xu'
import os
import re


def get_image_path(path_name):
  ret = []
  for rt, dirs, files in os.walk(path_name):
    for dir in dirs:
      tmp = []
      for _, _, file in os.walk(path_name+dir+'/'):
        for f in file:
          tmp.append([rt+dir+'/'+f, f])
      ret.append([tmp, dir])
  return ret



if __name__ == '__main__':
  print(get_image_path('../../train/'))