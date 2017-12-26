import unittest
import matplotlib.image as mpimg
from os import remove
from pipeline import *
from pipeline_helpers import *

class TestPipeline(unittest.TestCase):
  def setup(self):
    pass

  def test_get_vehicle_fns(self):
    fns = get_vehicle_fns()
    self.assertEqual(fns[0], './vehicle-detection-vehicles/vehicles/GTI_Far/image0000.png')
    self.assertEqual(fns[-1], './vehicle-detection-vehicles/vehicles/KITTI_extracted/5969.png')

  def test_get_non_vehicle_fns(self):
    fns = get_non_vehicle_fns()
    self.assertEqual(fns[0], './vehicle-detection-non-vehicles/non-vehicles/Extras/extra1.png')
    self.assertEqual(fns[-1], './vehicle-detection-non-vehicles/non-vehicles/GTI/image3900.png')

  def test_load_imgs(self):
    imgs = load_imgs(['./vehicle-detection-vehicles/vehicles/GTI_Far/image0000.png'])
    self.assertEqual(imgs[0].shape, (64, 64, 3))
    remove('./images.p')

  def test_get_hog_features(self):
    image = np.uint8(mpimg.imread('./vehicle-detection-vehicles/vehicles/GTI_Far/image0000.png') * 255)
    features = get_hog_features(image[:, :, 0], orient, pix_per_cell, cell_per_block)
    self.assertEqual(np.ravel(features).shape[0], feat_per_sample)

if __name__ == '__main__':
  unittest.main()
