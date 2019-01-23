import os


def read_newest_model(model_dump_path):
  for rt, dirs, files in os.walk(model_dump_path):
    return files


if __name__ == '__main__':
  pass
