import json
import numpy as np
import pickle
import os
import os.path as osp
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
import re

path_to_model =  '/home/zhiyuyin/workspace/word2vector/m.pkl'
path_to_json =  '/data1/zhiyuyin/dataset/recipe1M/det_ingrs.json'
path_to_imgs = './imgs'
if not osp.exists(path_to_imgs):
  os.system('mkdir ' + path_to_imgs)

# load the model
#model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#model.save_word2vec_format('GoogleNews-vectors-negative300.txt', binary=False)
with open(path_to_model,'rb') as fm:
  model = pickle.load(fm)

# load json
with open(path_to_json,'r') as fj:
  samples=json.load(fj)
n_samples=len(samples)
print('%d samples for processing'%n_samples)

# word2vector
digit_ptn = re.compile('\d+')
seg_ptn =  re.compile('\ |\.|,|;|\|\(|\)|\{|\}|\[|\]|"')
for sample in samples:
  vector = []
  img_id = sample['id']
  #print(img_id)
  ingredients = sample['ingredients']
  if len(ingredients) > 15:
    continue
  for ingredient in ingredients:
    words = re.split('\ |\.|,|;|\|\(|\)|\{|\}|\[|\]|"', ingredient['text'].lower())
    #print(words)
    
    if len(words)==1:
      try:
        vector.append(model[words[0]])
      except:
        vector.append(np.zeros((300,),dtype=np.float))
    else:
      # statues1
      ret = []
      for word in words: 
        if len(digit_ptn.findall(word)) != 0:
          continue
        if len(seg_ptn.findall(word)) != 0:
          continue
        ret.append(word)
      tmp = []
      for r in ret:
        try:
          tmp.append(model[r.lower()])
        except:
          tmp.append(np.zeros((300,),dtype=np.float))
      vector.append(np.mean(tmp,axis=0))
      #target = [i+' ' for i in ret[0:-1]]
      #target.append(ret[-1])
      #target = '.'.join(target)
      #vector.append(model[target])
  try:
    vector = np.concatenate(vector,axis=0)
  except:
    continue
  vector = np.lib.pad(vector,(0,4500-vector.shape[0]), 'constant',constant_values=(0.0))
  np.reshape(vector,[15,300])
  
  path_to_save = osp.join(path_to_imgs, img_id+'.pkl') 
  with open(path_to_save,'wb') as f:
    pickle.dump(vector,f)
      
  
  

