import os
import shutil

mv_path = '/home/kirai_wendong/aborted'
img_root_path = '/home/kirai_wendong/proj/food-1000/train/'
dat_root_path = '/home/kirai_wendong/proj/food_w2c/'


def get_image_file(path, filename):
	path = os.path.join(path, filename[0])
	path = os.path.join(path, filename[1])
	path = os.path.join(path, filename[2])
	path = os.path.join(path, filename[3])
	path = os.path.join(path, filename)
	path += '.jpg'
	return path


def get_filename_pickfile(file_root_path):
	for rt, dirs, files in os.walk(file_root_path):
		ret = []
		for each in files:
			print(get_image_file(img_root_path, each[:-4]))
			try:
				fp = open(get_image_file(img_root_path, each[:-4]), 'rb')
				ret.append(each[:-4])
			except:
				print('don\'t have {}. ignore'.format(each))
				shutil.move(os.path.join(file_root_path, each), mv_path)
				continue

if __name__ == '__main__':
	get_filename_pickfile(dat_root_path)
