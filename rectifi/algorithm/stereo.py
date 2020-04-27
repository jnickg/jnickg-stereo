import cv2 as cv
import numpy as np

def rectify(image_buffers):
  """Rectifies the given images

  Parameters:
    image_buffers (array): Array of image buffers, as in calling numpy.fromfile('some/file', dtype=np.uint8) for each.
    fmt: A valid image extension (default: '.bmp') representing the format to which the output buffer will be encoded

  Returns:
    TBD
  """
  output = {}

  #
  # Read all images in
  #
  cvimgs = []
  for imgbuf in image_buffers:
    cvimg = cv.imdecode(imgbuf, cv.IMREAD_COLOR)
    if cvimg is None:
      raise ValueError("Failed to decode provided image")
    cvimgs.append(cvimg)

  #
  # Validate they are the same size (TODO normalize image spaces?)
  #
  it = iter(cvimgs)
  first_img = next(it)
  h_expected = len(first_img)
  w_expected = len(first_img[0])
  if not all ((len(i) == h_expected and len(i[0]) == w_expected) for i in it):
    raise ValueError("Images must all be the same size.")

  #
  # Capture chessboard corners from the calibration matrix we assume to be in
  # the photo. If any of the photos don't have a matrix, dump it and see if we
  # can continue without them
  # See 717
  #

  #cv.findChessboardCorners
  #cv.drawChessboardCorners

  #
  # Calibrate images using the chessboard, to remove distortions.
  # See 718
  #

  #cv.calibrateCamera

  #
  # Compute Fundamental matrix between images. This matrix relates the two
  # images together (in pixel space rather than physical space, as E does),
  # so we will need to do this for all images.
  # See 719
  #

  #cv.undistortPoints
  #cv.undistortPointsIter

  # All found chessboard corners must be inliers, so they must satisfy epipolar
  # constraints. Thus, use the 8-point algorithm because it's fastest & most
  # accurate (in this case)
  #cv.findFundamentalMat

  #
  # Check return value of findFundamentalMat to see if we are forming a
  # degenerate configuration
  #

  # TODO what is the Python equivalent of cv::noArray() ???

  #
  # Build the undistort map
  # See 720
  #

  #cv.initUndistortRectifyMap

  #
  # Compute epipolar lines using the fundamental matrix F calculated above.
  # See 721
  #

  #cv.computeCorrespondEpilines

  #
  # Calibrate between the images
  # See 723
  #
  #cv.stereoCalibrate

  #
  # If we succeeded in calibrating, proceed with calibrated rectification
  # See 732
  #
  #cv.initUndistortRectifyMap
  #cv.stereoRectify

  #
  # If we failed in calibrating (e.g. failed to find chessboards), try
  # uncalibrated rectification.
  # See 729
  #
  #cv.stereoRectifyUncalibrated

  return output
