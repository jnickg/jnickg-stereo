import os
from os import listdir
from os.path import isfile, join
import unittest
from ..algorithm import average_pels
from ..algorithm import stereo
import numpy as np
import hashlib as hl
import cv2 as cv
import filecmp
import tempfile

CONFIG_OUTPUT_DUMP_DIR = tempfile.gettempdir()
test_fmt = '.bmp'
expected_output = "expected"
actual_output = "actual"

def run_test_on(func, image_files, output_file, save_path=None):
  image_buffers = []
  for f in image_files:
    with open(f) as fh:
      image_buffers.append(np.fromfile(fh, dtype=np.uint8))

  p = {
    "fmt": test_fmt,
    "save_path": save_path,
    "verbose": True,
    "debug": True
  }
  output = func(image_buffers, params=p)

  return output

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


class TestAveragePels(unittest.TestCase):
  def setUp(self):
    print("TestAveragePels:")
    test_path = join(os.path.dirname(os.path.realpath(__file__)), 'res/average_pels')
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
      print("Ensuring file is removed: " + self.output_file)
      os.remove(self.output_file)
      pass
    except:
      pass

  def test_average_pels(self):
    output = run_test_on(average_pels.average_pels, self.image_files, self.output_file)
    self.assertTrue(len(output) == 1)
    encoded, filename = output[0]
    self.assertTrue(encoded is not None and len(encoded) > 0)
    self.assertTrue(filename is not None)

class TestStereo(unittest.TestCase):
  def setUp(self):
    print("TestStereo:")
    self.test_path = join(os.path.dirname(os.path.realpath(__file__)), 'res/stereo')
    print('Enumerating test resources in: ' + self.test_path)

    self.output_file = join(self.test_path, actual_output + test_fmt)
    print("Saving output file to: " + self.output_file)

    self.expected_file = join(self.test_path, expected_output + test_fmt)
    print("Comparing output file to: " + self.expected_file)

    self.start_files = [join(self.test_path, f) for f in listdir(self.test_path) if isfile(join(self.test_path, f))]
    self.image_files = self.start_files
    if (self.expected_file in self.image_files):
      self.image_files.remove(self.expected_file)
    print("Testing with input files:")

    for f in self.image_files:
      print('\t' + f)

  def tearDown(self):
    global CONFIG_OUTPUT_DUMP_DIR
    current_files = [join(self.test_path, f) for f in listdir(self.test_path) if isfile(join(self.test_path, f))]
    dump_dir = CONFIG_OUTPUT_DUMP_DIR
    if not os.path.isdir(dump_dir):
      os.makedirs(dump_dir)
    print(f"Moving test artifacts to: {dump_dir} ...")
    for f in [f for f in current_files if f not in self.start_files]:
      try:
        base = os.path.basename(f)
        dest = os.path.join(dump_dir, base)
        os.replace(f, dest)
      except:
        print(f"Failed to move file: {f}")
    try:
      os.remove(self.output_file)
    except:
      pass
    print(f"MOVED TEST ARTIFACTS TO: {dump_dir}.")

  def test_rectify(self):
    outputs = run_test_on(stereo.rectify, self.image_files, self.output_file, save_path=self.test_path)
    for file_name in outputs:
      #self.assertTrue(isfile(file_name), f"Could not find {file_name}")
      pass

if __name__ == '__main__':
    unittest.main()