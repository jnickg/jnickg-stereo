import os
from os import listdir
from os.path import isfile, join
import unittest
from ..algorithm import fun
import numpy as np
import hashlib as hl
import cv2 as cv
import filecmp

test_fmt = '.bmp'
expected_output = "expected"
actual_output = "actual"

def run_test_on(func, image_files, output_file):
  image_buffers = []
  for f in image_files:
    with open(f) as fh:
      image_buffers.append(np.fromfile(fh, dtype=np.uint8))

  avg_output = func(image_buffers, fmt=test_fmt)
  output_fh = open(output_file, 'wb+')
  output_fh.write(avg_output)
  output_fh.close()

def buffers_all_same(buffers):
  hashes = []
  for b in buffers:
    new_digest = hl.md5(b).hexdigest()
    hashes.append(new_digest)
  return hashes[1:] == hashes[:-1]

def files_all_same(file_names):
  hashes = []
  for f in file_names:
    with open(f, 'rb') as fh:
      new_digest = hl.md5(fh.read()).hexdigest()
      print("{0} digest: {1}".format(f, new_digest))
      hashes.append(new_digest)
  return hashes[1:] == hashes[:-1]

def images_all_same(image_files):
  hashes = []
  image_buffers = []
  for f in image_files:
    with open(f) as fh:
      cvimg = cv.imdecode(np.fromfile(fh, dtype=np.uint8), cv.IMREAD_UNCHANGED)
      new_digest = hl.md5(cvimg).hexdigest()
      print("{0} digest: {1}".format(f, new_digest))
      hashes.append(new_digest)
  return hashes[1:] == hashes[:-1]


class TestFunAlgorithm(unittest.TestCase):
  def setUp(self):
    test_path = join(os.path.dirname(os.path.realpath(__file__)), 'res')
    print('Enumerating test resources in: ' + test_path)

    self.output_file = join(test_path, actual_output + test_fmt)
    print("Saving output file to: " + self.output_file)

    self.expected_file = join(test_path, expected_output + test_fmt)
    print("Comparing output file to: " + self.expected_file)

    self.image_files = [join(test_path, f) for f in listdir(test_path) if isfile(join(test_path, f))]
    if (self.expected_file in self.image_files):
      self.image_files.remove(self.expected_file)
    print("Testing with input files:")

    for f in self.image_files:
      print('\t' + f)

  def tearDown(self):
    try:
      os.remove(self.output_file)
      pass
    except:
      pass

  def test_fun(self):
    run_test_on(fun.average_pels, self.image_files, self.output_file)
    self.assertTrue(isfile(self.output_file))
    self.assertTrue(images_all_same([self.output_file, self.expected_file]))

if __name__ == '__main__':
    unittest.main()