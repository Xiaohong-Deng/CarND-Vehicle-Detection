import unittest
from os import remove
from pipeline import *

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

if __name__ == '__main__':
  unittest.main()
