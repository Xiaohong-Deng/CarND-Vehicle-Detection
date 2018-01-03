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
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import argparse
# %matplotlib inline

colorspace = 'YUV'
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = "ALL"
block_per_row = 64 / pix_per_cell - cell_per_block + 1
feat_per_sample = (block_per_row ** 2) * (cell_per_block ** 2) * orient
# scale, (image search area, window) dictionary, values are 3-tuple containing search start, search end and window size
scale_params = {1: (410, 506, 64), 1.5: (410, 650, 64), 2: (410, 650, 64)}

class Box():
  """
  A class used to track the box information defined by heatmaps from the last n frames
  """
  def __init__(self):
    # heatmaps for the last n frames
    self.heatmaps = []
    # sum of the last n heatmaps
    self.acc_heatmap = None
    # average of the last n heatmap
    self.avg_heatmap = None

def update_box(box, heatmap, n_frames):
  """
  Update Box() instance

  Input
  -----
  box : the Box() instance needed to be updated

  heatmap : heatmap matrix, likely to be the one for the current frame

  n_frames : number of frames the Box() instance need to track, track the last n_frames frames
  """
  num_tracked_frames = len(box.heatmaps)

  if num_tracked_frames == n_frames:
    box.acc_heatmap -= box.heatmaps.pop(0)
  box.heatmaps.append(heatmap)

  if box.acc_heatmap is None:
    box.acc_heatmap = heatmap
    box.avg_heatmap = heatmap
  else:
    box.acc_heatmap += heatmap
    if num_tracked_frames == n_frames:
      box.avg_heatmap = box.acc_heatmap // n_frames
    else:
      box.avg_heatmap = box.acc_heatmap // (num_tracked_frames + 1)

def load_model():
  """
  Load the Linear SVM classifier and the normalizer generated from the data used to train the classifier

  Output
  -----
  clf : the Linear SVM classifier

  feat_scaler : normalizer used to preprocess all the data fed to the classifier
  """
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
  """
  Return the proper threshold used to decide whether or not you should reject
  a positive predicted by Linear SVM classifier

  Input
  -----
  X: features fed to classifier to do the prediction

  y_true : true labels associated with X

  y_pred : predicted labels associated with X

  Output
  -----
  thresh : a real number representing the confidence score of a prediction
  """
  feat_fp = X[(y_pred - y_true)==1]
  feat_tp = X[(y_pred == 1) & (y_true == 1)]
  conf_scores_fp = clf.decision_function(feat_fp)
  conf_scores_tp = clf.decision_function(feat_tp)
  thresh = np.percentile(conf_scores_fp, 75)

  return thresh

def load_training_data():
  """
  Load all the HOG features of the training images and the labels of the images

  Output
  -----
  features :
  """
  imgs, labels = load_imgs()

  features = ph.extract_features(imgs, img_format='PNG', color_space=colorspace, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=False, hist_feat=False)


  return features, labels

# 8792 images of vehicles
# 8968 images of non-vehicles
def get_img_fns(prefix='./vehicle-detection-vehicles/vehicles/',
                    subfolders=['GTI_Far', 'GTI_Left', 'GTI_MiddleClose', 'GTI_Right', 'KITTI_extracted']):
  """
  Extract all the image file names from the designated directory

  Output
  -----
  img_fns : a list of image file names
  """
  img_fns = []

  for sf in subfolders:
    file_pattern = prefix + sf + '/*.png'
    img_fns.extend(glob.glob(file_pattern))

  return img_fns

def load_imgs():
  """
  Load all the image matrices and the corresponding labels

  Output
  -----
  imgs : images in the form of numpy arrays

  labels : labels that consists of 0 and 1, indicating car and non-car
  """
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
  """
  Draw bounding boxes around cars in the images

  Input
  -----
  image : raw image user wants to detect cars on

  heat : whether or not use heatmaps to draw bounding boxes

  heat_thresh : threshold used to decide if a pixel is false-positive

  Output
  -----
  bbimage : the image identical to the input image except that detected cars have colored rectangle boxes around them
  """
  clf, feat_scaler = load_model()
  bboxes = ph.multi_scale_sliding_window(image, clf, feat_scaler, scale_params, orient=orient,
                                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
  # either use heatmaps to shake off false-positives or not
  if heat:
    bbimage = ph.draw_heat_boxes(image, bboxes, heat_thresh)
  else:
    bbimage = ph.draw_boxes(image, bboxes)

  return bbimage

def process_frames(box=None, heat_thresh=1, n_frames=20):
  """
  Draw bounding boxes around cars in the video

  Input
  -----
  box : a Box() instance

  heat_thresh : heatmap threshold used on heatmap for a single frame

  n_frames : number of frames the Box() instance should keep track of

  Output
  -----
  process_frame : method used to process a single frame
  """
  if box is None:
    box = Box()

  def process_frame(image):
    clf, feat_scaler = load_model()
    bboxes = ph.multi_scale_sliding_window(image, clf, feat_scaler, scale_params, orient=orient,
                                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = ph.add_heat(heat, bboxes)

    update_box(box, heat, n_frames)

    # use a mix of average heatmap and the current frame heatmap to decide the bounding boxes
    # for the current frame heatmap we first use heatmap on it to shake off false-positives
    heatmap = (0.6 * box.avg_heatmap + 0.4 * ph.apply_threshold(np.copy(box.heatmaps[-1]), 2)).astype(np.int_)
    heatmap[heatmap < heat_thresh] = 0

    heatmap = np.clip(heatmap, 0, 255)
    labels = label(heatmap)
    draw_img = ph.draw_labeled_bboxes(np.copy(image), labels)

    return draw_img

  return process_frame

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
    clip = VideoFileClip("./project_video.mp4")
    bbvideo = clip.fl_image(process_frames())

    bbvideo.write_videofile('./bboxed_project_video.mp4', audio=False)

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
