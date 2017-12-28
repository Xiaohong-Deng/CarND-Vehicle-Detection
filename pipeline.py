import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage.feature import hog
from sklearn.utils import shuffle
import pipeline_helpers as ph
from skimage.feature import hog
from importlib import reload
%matplotlib inline

colorspace = 'YUV'
orient = 9
pix_per_cell = 16
cell_per_block = 2
hog_channel = "ALL"
block_per_row = 64 / pix_per_cell - cell_per_block + 1
feat_per_sample = (block_per_row ** 2) * (cell_per_block ** 2) * orient

def load_model():
  try:
    with open('model.p', mode='rb') as f:
      clf_params = pickle.load(f)
      clf, feat_scaler = clf_params['clf'], clf_params['feat_scaler']
  except FileNotFoundError:
    features, labels = load_training_data()
    print('number of samples: ', len(features))
    print('shape of features: ', features[0].shape)
    print('shape of labels: ', labels.shape)
    feat_scaler = StandardScaler().fit(features)
    scaled_feats = feat_scaler.transform(features)
    features, labels = shuffle(scaled_feats, labels)
    features_train, features_valid, labels_train, labels_valid = train_test_split(features, labels, test_size=0.2)
    clf = LinearSVC()

    # random_search = GridSearchCV(clf, param_grid=param_dist)
    # random_search.fit(features, labels)
    # best_C = random_search.best_params_['C']

    t = time.time()
    clf.fit(features_train, labels_train)
    t2 = time.time()
    acc = round(clf.score(features_valid, labels_valid), 4)
    print(round(t2-t, 2), 'Seconds to train SVC...')
    print('Test Accuracy of SVC = ', acc)

    if acc >= 0.95:
      clf_params = {}
      clf_params['clf'] = clf
      clf_params['feat_scaler'] = feat_scaler
      with open('model.p', mode='wb') as f:
        pickle.dump(clf_params, f)

  return clf, feat_scaler

def load_training_data():
  imgs, labels = load_imgs()

  features = ph.extract_features(imgs, img_format='PNG', color_space=colorspace, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=False, hist_feat=False)


  return features, labels

# 8792 images of vehicles
# 8968 images of non-vehicles
def get_img_fns(prefix='./vehicle-detection-vehicles/vehicles/',
                    subfolders=['GTI_Far', 'GTI_Left', 'GTI_MiddleClose', 'GTI_Right', 'KITTI_extracted']):
  img_fns = []

  for sf in subfolders:
    file_pattern = prefix + sf + '/*.png'
    img_fns.extend(glob.glob(file_pattern))

  return img_fns

def load_imgs():
  try:
    with open('images.p', mode='rb') as f:
      img_params = pickle.load(f)
      imgs = img_params['imgs']
      labels = img_params['labels']
  except FileNotFoundError:
    img_fns = []

    img_fns.extend(get_img_fns())
    labels_vehicle = np.ones(len(img_fns))

    img_fns.extend(get_img_fns(prefix='./vehicle-detection-non-vehicles/non-vehicles/',
                                  subfolders=['Extras', 'GTI']))
    labels_non_vehicle = np.zeros(len(img_fns) - labels_vehicle.shape[0])

    labels = np.hstack((labels_vehicle, labels_non_vehicle))

    imgs = []
    for fn in img_fns:
      image = mpimg.imread(fn)
      imgs.append(image)

    img_params = {}
    img_params['imgs'] = imgs
    img_params['labels'] = labels
    with open('images.p', mode='wb') as f:
      pickle.dump(img_params, f)

  return imgs, labels

# image = cv2.imread('./vehicle-detection-vehicles/vehicles/GTI_Far/image0000.png')
# print(type(np.max(image)))
# print(np.min(image))
# yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
# print(type(np.max(yuv_image)))
clf, feat_scaler = load_model()
image = mpimg.imread('./test_images/test1.jpg')
img_tosearch = ph.get_img_tosearch(image, 410, 656)
hogs = ph.get_image_hog(img_tosearch, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
bboxes = ph.one_shot_sliding_window(hogs, img_tosearch, 410, clf, feat_scaler, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block)
bbimg = ph.draw_boxes(image, bboxes)
# filename = './vehicle-detection-vehicles/vehicles/GTI_Far/image0000.png'
# image = mpimg.imread(filename)
#
# print(np.max(image))
# scaled = np.uint8(image * 255)
# print(scaled.shape)
#
plt.figure(figsize=(1280,720))
plt.imshow(bbimg)
