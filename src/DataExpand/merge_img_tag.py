__author__ = 'Wendong Xu'
import json
import os
import multiprocessing as mp
import copy


def label_cross(a, b):
  c_a = set(a)
  c_b = set(b)
  return c_a.intersection(c_b)


def preprocess_label_cross(json_input_path, json_output_path, thred=3):
  pool = mp.Pool(max(mp.cpu_count() - 2, 1))

  with open(json_input_path, 'r') as fp:
    input_json = json.load(fp)
    output_json = copy.deepcopy(input_json)

    def _process_one_label(i, label_i):
      output_json[i]['ingredients'] = []
      output_json[i]['valid'] = []
      tmp_ingredients = set([])
      for j in range(0, len(input_json)):
        label_j = []
        if i == j:
          continue
        for k in range(0, len(input_json[j]['valid'])):
          if input_json[j]['valid'][k] == True:
            label_j.append(input_json[j]['ingredients'][k]['text'])
        label_x = label_cross(label_i, label_j)
        if len(label_x) >= thred:
          for lbl in label_x:
            tmp_ingredients.add(lbl)
      for each in tmp_ingredients:
        tmp = dict()
        tmp['text'] = each
        output_json[i]['ingredients'].append(tmp)
      print('{} has finished'.format(i))

    for i in range(0, len(input_json)):
      label_i = []
      for k in range(0, len(input_json[i]['valid'])):
        if input_json[i]['valid'][k] == True:
          label_i.append(input_json[i]['ingredients'][k]['text'])
      _process_one_label(i, label_i)
      # pool.apply_async(_process_one_label, (label_i, ))
      # pool.close()
      # pool.join()
  with open(json_output_path, 'w') as fp:
    json.dump(output_json, fp)


if __name__ == '__main__':
  # preprocess_label_cross('../../data/det_ingrs_fake.json', '../../data/det_ingrs_fake_output.json')
  preprocess_label_cross('../../data/det_ingrs_fake.json', '../../data/det_ingrs_fake_output.json')
