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
from moviepy.editor import VideoFileClip
from importlib import reload
import argparse
# %matplotlib inline

colorspace = 'YUV'
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = "ALL"
block_per_row = 64 / pix_per_cell - cell_per_block + 1
feat_per_sample = (block_per_row ** 2) * (cell_per_block ** 2) * orient
scale_params = {1: (410, 506, 64), 1.5: (410, 650, 64), 2: (410, 650, 64)}

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

# this method is used in hard-negative mining
def conf_score_thresh(X, y_true, y_pred):
  feat_fp = X[(y_pred - y_true)==1]
  feat_tp = X[(y_pred == 1) & (y_true == 1)]
  conf_scores_fp = clf.decision_function(feat_fp)
  conf_scores_tp = clf.decision_function(feat_tp)
  thresh = np.percentile(conf_scores_fp, 75)

  return thresh

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

def process_image(image, heat=False, heat_thresh=2):
# image = cv2.imread('./vehicle-detection-vehicles/vehicles/GTI_Far/image0000.png')
# print(type(np.max(image)))
# print(np.min(image))
# yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
# print(type(np.max(yuv_image)))
# reload(ph)
  clf, feat_scaler = load_model()
  bbimage = ph.multi_scale_sliding_window(image, clf, feat_scaler, scale_params, heat=heat, heat_thresh=heat_thresh, orient=orient,
                                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
  return bbimage
# img_tosearch = ph.get_img_tosearch(image, 410, 650)
# hogs = ph.get_image_hog(img_tosearch, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
# bboxes = ph.one_shot_sliding_window(hogs, img_tosearch, clf, feat_scaler, 410, pix_per_cell=pix_per_cell,
#                                     cell_per_block=cell_per_block)
# bbimg = ph.draw_boxes(image, bboxes)
# filename = './vehicle-detection-vehicles/vehicles/GTI_Far/image0000.png'
# image = mpimg.imread(filename)
#
# print(np.max(image))
# scaled = np.uint8(image * 255)
# print(scaled.shape)
#
# clf.decision_function().shape
# features, labels = load_training_data()
# features = feat_scaler.transform(features)
# labels_pred = clf.predict(features)

# conf_scores_fp = clf.decision_function(feat_fp)
# conf_scores_tp = clf.decision_function(feat_tp)
# print('mean confidence for false positives is: ', np.mean(conf_scores_fp))
# print('median confidence for false positives is: ', np.median(conf_scores_fp))
# print('median confidence for true positive is: ', np.median(conf_scores_tp))
# print('min confidence for true positive is: ', np.min(conf_scores_tp))
# np.percentile(conf_scores_fp, 60)
# np.percentile(conf_scores_tp, 3)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--image", help="if specified, generate bounding boxed images from './test_images' to './output_images'",
                      action="store_true")
  parser.add_argument("-v", "--video", help="if specified, generate a bounding boxed video from './project_video.mp4'",
                      action="store_true")
  parser.add_argument("-hm", "--heatmap", help="if specified, generate heatmap based bounding boxed images from './test_images' to './output_images'",
                      action="store_true")
  args = parser.parse_args()

  if args.image:
    fp = './test_images/*.jpg'
    fns = glob.glob(fp)
    count = 1

    for fn in fns:
      image = mpimg.imread(fn)
      bbimage = process_image(image)
      out_img_name = './output_images/bboxed_test' + str(count) + '.jpg'
      cv2.imwrite(out_img_name, cv2.cvtColor(bbimage, cv2.COLOR_RGB2BGR))
      count += 1

  if args.video:
    clip = VideoFileClip("./test_video.mp4")
    bbvideo = clip.fl_image(process_image)

    bbvideo.write_videofile('./bboxed_test_video.mp4', audio=False)

  if args.heatmap:
    fp = './test_images/*.jpg'
    fns = glob.glob(fp)
    count = 1

    for fn in fns:
      image = mpimg.imread(fn)
      bbimage = process_image(image, heat=True)
      out_img_name = './output_images/heat_bboxed_test' + str(count) + '.jpg'
      cv2.imwrite(out_img_name, cv2.cvtColor(bbimage, cv2.COLOR_RGB2BGR))
      count += 1
  # image = mpimg.imread('./output_images/bboxed_test1.jpg')
