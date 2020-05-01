import cv2 as cv
import numpy as np

dflt_params = {
  "fmt":".bmp",
  "verbose": True,
  "chess_sz":(6,9),
  "chess_col":6,
  "chess_row":9,
  "chess_total":54,
  "criteria": (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
}

def get_param(key, params):
  return params.get(key, dflt_params.get(key, None))

def rectify(image_buffers, params=dflt_params):
  """Rectifies the given images

  Parameters:
    image_buffers (array): Array of image buffers, as in calling numpy.fromfile('some/file', dtype=np.uint8) for each.
    fmt: A valid image extension (default: '.bmp') representing the format to which the output buffer will be encoded

  Returns:
    TBD
  """
  output = {}

  #
  print("+++Reading input images, decoding to CV format")
  #
  cvimgs = []
  for imgbuf in image_buffers:
    cvimg = cv.imdecode(imgbuf, cv.IMREAD_GRAYSCALE)
    if cvimg is None:
      raise ValueError("Failed to decode provided image")
    cvimgs.append(cvimg)

  #
  print("+++Validating image sizes...") # (TODO normalize image spaces?)
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
  print("+++Finding chessboard corners...")
  #
  chessboard_size = get_param("chess_sz", params)
  found = [cv.findChessboardCorners(i, chessboard_size) for i in cvimgs]
  if not all (f[0] is True for f in found):
    raise ValueError("Unable to find calibration points for all images.")
  if get_param("verbose", params):
    for f in found:
      print("Found points: {0}".format(f[1]))
  # TODO something with this???
  #cv.drawChessboardCorners

  #
  # Calibrate images using the chessboard, to remove distortions.
  # See 718
  print("+++Calibrating images...")
  for i, pt in zip(cvimgs, [f[1] for f in found]):
    objP = np.zeros((get_param("chess_total",params), 3), np.float32)
    objP[:,:2] = np.mgrid[0:get_param("chess_row", params), 0:get_param("chess_col", params)].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    objpoints.append(objP)
    cv.cornerSubPix(i, pt, (11, 11), (-1, -1), get_param("criteria", params))
    imgpoints.append(pt)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, i.shape[::-1], None, None)
    if get_param("verbose", params):
      print("Successfully calibrated! ret:{0} mtx:{1} dist:{2} rvecs:{3} tvecs:{4}".format(ret, mtx, dist, rvecs, tvecs))

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

  raise ValueError("HIT END OF TEST IMPLEMENTATION")
  return output
