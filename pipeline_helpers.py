import matplotlib.image as mpimg
import numpy as np
import cv2
import pdb
from skimage.feature import hog
from scipy.ndimage.measurements import label
# Define a function to return HOG features and visualization
def convert_color(img, conv='RGB2YCrCb'):
  """
  Convert image to specified color space

  Input
  -----
  image : the input image

  conv : specified source color space and the destinated color space

  Output
  -----
  converted image
  """
  if conv == 'RGB2YCrCb':
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
  if conv == 'BGR2YCrCb':
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
  if conv == 'RGB2LUV':
    return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
  if conv== 'RGB2YUV':
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def get_img_tosearch(img, ystart, ystop, conv='RGB2YUV', img_format='JPG'):
  """
  Trim the image to keep the interested area only

  Input
  -----
  img : the input image

  ystart : starting y coordinate of the interested area in the input image

  ystop : stopping y coordinate of the interested area in the input image

  conv : specified source color space and the destinated color space

  img_format : the format of the input image

  Output
  -----
  img_tosearch : images trimmed down to the interested area only
  """
  img_tosearch = img[ystart:ystop, :, :]
  if img_format == 'PNG':
    img_tosearch = np.uint8(img_tosearch * 255)
  img_tosearch = convert_color(img_tosearch, conv=conv)

  return img_tosearch

def get_image_hog(img_tosearch, orient=9,
                    pix_per_cell=8, cell_per_block=2):
  """
  Return HOG features of the input image

  Input
  -----
  img_tosearch : the input image

  orient : the number of bins for each cell

  pix_per_cell : number of pixels per cell

  cell_per_block : number of cells per block

  Output
  -----
  hogs : flattenned HOG features from all 3 channels
  """

  ch1 = img_tosearch[:, :, 0]
  ch2 = img_tosearch[:, :, 1]
  ch3 = img_tosearch[:, :, 2]

  hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
  hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
  hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

  hogs = (hog1, hog2, hog3)
  return hogs

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
  """
  Extract HOG features from a single channel

  Input
  -----
  img : single channel image

  orient : the number of bins for each cell

  pix_per_cell : number of pixels per cell

  cell_per_block : number of cells per block

  vis : boolean value used to decide if you want the HOG visulaized image

  feature_vec : boolean value used to decide if you want flattenned HOG features

  Output
  -----
  features : HOG features
  """
  # Call with two outputs if vis==True
  if vis == True:
    features, hog_image = hog(img, orientations=orient,
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block),
                              transform_sqrt=True,
                              visualise=vis, feature_vector=feature_vec)
    return features, hog_image
  # Otherwise call with one output
  else:
    features = hog(img, orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   transform_sqrt=False,
                   visualise=vis, feature_vector=feature_vec)

    return features

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
  # Use cv2.resize().ravel() to create the feature vector
  features = cv2.resize(img, size).ravel()
  # Return the feature vector
  return features

# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
  # Compute the histogram of the color channels separately
  channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
  channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
  channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
  # Concatenate the histograms into a single feature vector
  hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
  # Return the individual histograms, bin_centers and feature vector
  return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, img_format='JPG', color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
  """
  Extract features ranged from bin spatial, color histogram and HOG

  Input
  -----
  imgs : a list of input images

  img_format : the format of input images

  color_space : color space you want to convert the images to

  spatial_size : the size you want to resize the images to

  hist_bins : number of bins to apply to color histogram feature extraction

  orient : the number of bins for each cell

  pix_per_cell : number of pixels per cell

  cell_per_block : number of cells per block

  spatial_feat : boolean value used to decide if extract spatial features

  hist_feat : boolean value used to decide if extract color histogram features

  hog_feat : boolean value used to decide if extract HOG features

  Output
  -----
  features : possibly mixed, flattenned features
  """
    # Create a list to append feature vectors to
  features = []
  # Iterate through the list of images
  for image in imgs:
    image_features = []
    if img_format == 'PNG':
      image = np.uint8(image * 255)
    # Read in each one by one
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
      if color_space == 'HSV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
      elif color_space == 'LUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
      elif color_space == 'HLS':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
      elif color_space == 'YUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
      elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
      feature_image = np.copy(image)

    if spatial_feat == True:
      spatial_features = bin_spatial(feature_image, size=spatial_size)
      image_features.append(spatial_features)
    if hist_feat == True:
      # Apply color_hist()
      hist_features = color_hist(feature_image, nbins=hist_bins)
      image_features.append(hist_features)
    if hog_feat == True:
    # Call get_hog_features() with vis=False, feature_vec=True
      if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
          hog_features.append(get_hog_features(feature_image[:,:,channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
      else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                      pix_per_cell, cell_per_block, vis=False, feature_vec=True)
      # Append the new feature vector to the features list
      image_features.append(hog_features)
    features.append(np.concatenate(image_features))
  # Return list of feature vectors
  return features

def one_shot_sliding_window(hogs, img_tosearch, clf, feat_scaler, ystart, scale, window_size=64, pix_per_cell=8,
                              cell_per_block=2, cells_per_step=2):
  """
  Return bounding box coordinates for the input image based on the image HOG features

  Input
  -----
  hogs : HOG features of the input image

  img_tosearch : the input image

  clf : Linear SVM classifier

  feat_scaler : normalizer generated from the training data the classifier was trained on

  ystart : the y coordinate where the method start slide the window on the original image, not img_tosearch

  scale : the number used to scale the window size

  window_size : the width and height of the sliding window

  pix_per_cell : number of pixels per cell

  cell_per_block : number of cells per block

  cells_per_step : number of cells each time the method needs to cross when moving the window
  """
  bboxes = []

  hog1, hog2, hog3 = hogs

  nxblocks = (img_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
  nyblocks = (img_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1

  nblocks_per_window = (window_size // pix_per_cell) - cell_per_block + 1
  nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
  nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

  for xb in range(nxsteps):
    for yb in range(nysteps):
      ypos = yb * cells_per_step
      xpos = xb * cells_per_step

      hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
      hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
      hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
      hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

      xleft = xpos * pix_per_cell
      ytop = ypos * pix_per_cell

      # sub_img = cv2.resize(img_tosearch[ytop:ytop + window_size, xleft:xleft + window_size], (64, 64))
      features = feat_scaler.transform(hog_features.reshape(1, -1))
      prediction = clf.predict(features)

      if prediction == 1 and clf.decision_function(features)[0] > 0.6:
        xleft_draw = np.int(xleft * scale)
        ytop_draw = np.int(ytop * scale)
        win_draw = np.int(window_size * scale)
        bbox = ((xleft_draw, ytop_draw + ystart), (xleft_draw + win_draw, ytop_draw + ystart + win_draw))
        bboxes.append(bbox)

  return bboxes

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
  """
  Draw bounding boxes according to coordinates given

  Input
  -----
  img : the input image

  bboxes : a list of coordinates indicating the bounding boxes

  color : bounding box color

  thick : weight of bounding box

  Output
  -----
  imcopy : a copy of the input image that has bounding boxes drawn
  """
  # Make a copy of the image
  imcopy = np.copy(img)
  # Iterate through the bounding boxes
  for bbox in bboxes:
    # Draw a rectangle given bbox coordinates
    cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
  # Return the image copy with boxes drawn
  return imcopy

def img_scaled(image, scale):
  """
  Return scaled image

  Input
  -----
  image : the input image

  scale : the scaling number applied to the input image
  """
  imshape = image.shape
  img_resized = cv2.resize(image, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

  return img_resized

# scale params should be of the form {scale_coeff1: (ystart1, ystop1, window_size1), [scale_coeff2: (ystart, ystop, window_size2), ...]}
def multi_scale_sliding_window(image, clf, feat_scaler, scale_params, orient=9, pix_per_cell=8, cell_per_block=2):
  """
  Apply sliding window multiple time with different window size and search area, return bounding box coordinates found

  Input
  -----
  image : the input image

  clf : Linear SVM classifier

  feat_scaler : normalizer generated from the training data the classifier was trained on

  scale_params : a dictionary in which scales are keys and (ystart, ystop, window_size) tuple are values

  orient : the number of bins for each cell

  pix_per_cell : number of pixels per cell

  cell_per_block : number of cells per block

  Output
  -----
  multi_bboxes : bounding box coordinates
  """
  multi_bboxes = []
  for scale, params in scale_params.items():
    ystart, ystop, window_size = params[0], params[1], params[2]
    img_tosearch = get_img_tosearch(image, ystart, ystop)
    if scale != 1:
      img_tosearch = img_scaled(img_tosearch, scale)
    hogs = get_image_hog(img_tosearch, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
    bboxes = one_shot_sliding_window(hogs, img_tosearch, clf, feat_scaler, ystart, scale, window_size=window_size, pix_per_cell=pix_per_cell,
                                            cell_per_block=cell_per_block)
    multi_bboxes.extend(bboxes)

  return multi_bboxes

def draw_heat_boxes(image, bboxes, threshold):
  """
  Draw bounding boxes around detected objects according to heatmaps

  Input
  -----
  image : the input image

  bboxes : bounding box coordinates of the detect objects

  threshold : heatmap threshold used to decide if a pixel is a false-positive

  Output
  -----
  draw_img : a copy of the input image with bounding boxes drawn
  """
  heat = np.zeros_like(image[:, :, 0]).astype(np.float)
  heat = add_heat(heat, bboxes)
  heat = apply_threshold(heat, threshold)
  heatmap = np.clip(heat, 0, 255)
  labels = label(heatmap)
  draw_img = draw_labeled_bboxes(np.copy(image), labels)

  return draw_img

def add_heat(heatmap, bbox_list):
  # Iterate through list of bboxes
  for box in bbox_list:
    # Add += 1 for all pixels inside each bbox
    # Assuming each "box" takes the form ((x1, y1), (x2, y2))
    heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

  # Return updated heatmap
  return heatmap

def apply_threshold(heatmap, threshold):
  # Zero out pixels below the threshold
  heatmap[heatmap < threshold] = 0
  # Return thresholded map
  return heatmap

def draw_labeled_bboxes(img, labels):
  # Iterate through all detected cars
  for car_number in range(1, labels[1]+1):
    # Find pixels with each car_number label value
    nonzero = (labels[0] == car_number).nonzero()
    # Identify x and y values of those pixels
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Define a bounding box based on min/max x and y
    bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
    # Draw the box on the image
    cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
  # Return the image
  return img

# Non-Maxima-Suppression for removing overlapped multiple detections
# here we use one of the bounding boxes that overlap with each other
# to represent that group of bounding boxes so we don't see overlapped
# boxes in the image
def non_max_suppression_slow(boxes, overlapThresh):
  if len(boxes == 0):
    return []

  pick = []

  # boxes in the form of np.array([[x1, y1, x2, y2], ...])
  x1 = boxes[:, 0]
  y1 = boxes[:, 1]
  x2 = boxes[:, 2]
  y2 = boxes[:, 3]

  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)

  while len(idxs) > 0:
    # pick the last lower right y coordinate as the reference or the pivot
    # compute overlapped area by comparing others to the one whose lower right
    # y coordinate is the last one
    last = len(idxs) - 1
    i = idxs[last]
    # i is the index of the bbox we pick to represent the overlapped bboxes in this round
    pick.append(i)
    suppress = [last]

    # start compare overlapped area
    for pos in range(last):
      j = idxs[pos]

      xx1 = max(x1[i], x1[j])
      yy2 = max(y1[i], y1[j])
      xx2 = min(x2[i], x2[j])
      yy2 = min(y2[i], y2[j])

      w = max(0, xx2 - xx1 + 1)
      h = max(0, yy2 - yy1 + 1)

      # ratio of overlapped area to referenced bbox
      overlap = float(w * h) / area[j]

      # if overlapped area is large enough, we suppress the bbox to be compared to ref
      if overlap > overlapThresh:
        suppress.append(pos)

    # suppress contains indices of idxs, which contains indices of y2 in the ascending order with respect to
    # y2 value. it means you remove all the overlapped bboxes found in this round and will not compare them
    # in the following rounds
    idxs = np.delete(idxs, suppress)

  return boxes[pick]

def non_max_suppression_fast(boxes, overlapThresh):
  if len(boxes) == 0:
    return []

  if boxes.dtype.kind == 'i':
    boxes = boxes.astype('float')

  pick = []

  x1 = boxes[:, 0]
  y1 = boxes[:, 1]
  x2 = boxes[:, 2]
  y2 = boxes[:, 3]

  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)

  while len(idxs) > 0:
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)

    # this compares two arrays, here x1[i] is broadcasted to an array
    # return a array containing element-wise comparison results
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    overlap = (w * h) / area[idxs[:last]]

    idxs = np.delete(idxs, np.concatenate(([last],
                                           np.where(overlap > overlapThresh)[0])))

  return boxes[pick].aastype(int)
