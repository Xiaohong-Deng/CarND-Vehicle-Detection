import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.utils import shuffle
import pipeline_helpers
# %matplotlib inline

colorspace = 'YUV'
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = "ALL"

def load_model():
  try:
    with open('model.p', mode='rb') as f:
      clf = pickle.load(f)
  except FileNotFoundError:

    clf = LinearSVC()

def load_training_data():
  features_vehicles, labels_vehicles = load_data()
  features_non_vehicles, labels_non_vehicles = load_data(is_vehicle=False)

  features = np.concatenate((features_vehicles, features_non_vehicles))
  labels = np.concatenate((labels_vehicles, labels_non_vehicles))

  return features, labels

def load_data(is_vehicle=True):
  if is_vehicle:
    img_fns = get_vehicle_fns()
  else:
    img_fns = get_non_vehicle_fns()

  imgs = load_imgs(img_fns)

  features = extract_features(imgs, img_format='PNG', color_space=colorspace, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=False, hist_feat=False)
  if is_vehicle:
    labels = [1] * len(features)
  else:
    labels = [0] * len(features)

  return features, labels

def get_vehicle_fns(prefix='./vehicle-detection-vehicles/vehicles/',
                      subfolders=['GTI_Far/', 'GTI_Left/', 'GTI_MiddleClose/', 'GTI_Right/', 'KITTI_extracted/'],
                      indices_per_folder=[{'start': 0, 'end': 974}, {'start': 9, 'end': 974}, {'start': 0, 'end': 494}, {'start': 1, 'end': 5969}],
                      index_len=4, padding='0'):
  img_fns = []

  for idx in range(len(subfolders) - 1):
    sf = subfolders[idx]
    indices = indices_per_folder[idx]
    for i in range(indices['start'], indices['end'] + 1):
      index = str(i)
      num_of_padding = index_len - len(index)
      paddings = padding * num_of_padding
      filename = prefix + sf + 'image' + paddings + index + '.png'
      img_fns.append(filename)

  for i in range(indices['start'], indices['end'] + 1):
    filename = prefix + subfolders[-1] + str(i) + '.png'
    img_fns.append(filename)

  return img_fns

def get_non_vehicle_fns(prefix='./vehicle-detection-non-vehicles/non-vehicles/',
                          subfolders=['Extras/', 'GTI/'],
                          file_prefix=['extra', 'image'],
                          indices_per_folder=[{'start': 1, 'end': 5766}, {'start': 1, 'end': 3900}]):
  img_fns = []

  for idx in range(len(subfolders)):
    sf = subfolders[idx]
    fp = file_prefix[idx]
    indices = indices_per_folder[idx]
    for i in range(indices['start'], indices['end'] + 1):
      filename = prefix + sf + fp + str(i) + '.png'
      img_fns.append(filename)

  return img_fns

def load_imgs(img_fns):
  try:
    with open('images.p', mode='rb') as f:
      imgs = pickle.load(f)
  except FileNotFoundError:
    imgs = []
    for fn in img_fns:
      image = mpimg.imread(fn)
      imgs.append(image)

    with open('images.p', mode='wb') as f:
      pickle.dump(imgs, f)

  return imgs

filename = './vehicle-detection-vehicles/vehicles/GTI_Far/image0000.png'
image = mpimg.imread(filename)

print(np.max(image))
scaled = np.uint8(image * 255)
print(scaled.shape)

plt.figure(figsize=(1,1))
plt.imshow(scaled)
