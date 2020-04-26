import cv2 as cv
import numpy as np

def average_pels(image_files, fmt='.bmp'):
  """Read the given image files, and average the co-local pixel values for all of them.

  If no dest_file is specified, a bitmap-encoded image is returned. If a dest_file is specified,
  a file of the same format as the inputs is written at that location, and None is returned.
  """
  output = {}

  # Read all images in
  cvimgs = []
  for img_f in image_files:
    cvimg = cv.imdecode(img_f, cv.IMREAD_UNCHANGED)
    if cvimg is None:
      raise ValueError("Failed to decode provided image")
    cvimgs.append(cvimg)

  # Validate they are the same size (TODO normalize image spaces?)
  it = iter(cvimgs)
  first_img = next(it)
  h_expected = len(first_img)
  w_expected = len(first_img[0])
  if not all ((len(i) == h_expected and len(i[0]) == w_expected) for i in it):
    raise ValueError("Images must all be the same size.")

  # Average the pixel values for all images
  num_images = len(cvimgs)
  totals = np.zeros_like(cvimgs[0])
  for cvi in cvimgs:
    np.add(totals, cvi, out=totals)
  totals = np.divide(totals, num_images).astype(int)
  _, output = cv.imencode(fmt, totals)

  return output
