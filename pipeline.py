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
from pipeline_helpers.py

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
