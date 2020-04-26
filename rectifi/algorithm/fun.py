import cv2 as cv
import numpy as np

def average_pels(image_buffers, fmt='.bmp'):
  """Read the given image files, and average the co-local pixel values for all of them.

  If no dest_file is specified, a bitmap-encoded image is returned. If a dest_file is specified,
  a file of the same format as the inputs is written at that location, and None is returned.

  Parameters:
    image_buffers (array): Array of image buffers, as in calling numpy.fromfile('some/file', dtype=np.uint8) for each.
    fmt: A valid image extension (default: '.bmp') representing the format to which the output buffer will be encoded

  Returns:
    A byte array representing an image file, encoded in the given format
  """
  output = {}

  # Read all images in
  cvimgs = []
  for imgbuf in image_buffers:
    cvimg = cv.imdecode(imgbuf, cv.IMREAD_UNCHANGED)
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
  totals = np.zeros_like(cvimgs[0]).astype(float)
  for cvi in cvimgs:
    cvi = cvi / num_images
    #np.divide(cvi, num_images, out=cvi)
    np.add(totals, cvi, out=totals)
  _, output = cv.imencode(fmt, totals)

  return output
