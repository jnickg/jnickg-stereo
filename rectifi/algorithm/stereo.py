import cv2 as cv
import numpy as np
import os.path as path

dflt_params = {
  "fmt":".bmp",
  "verbose": True,
  "chess_sz":(6,9),
  "chess_col":6,
  "chess_row":9,
  "chess_total":54,
  "criteria": (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
  "save_path":"./",
  "verbose":True
}

def get_param(key, params):
  return params.get(key, dflt_params.get(key, None))

def print_message(message, isVerbose=True, params=dflt_params):
  doVerbose = get_param("verbose", params)
  if (isVerbose is True and doVerbose is True):
    print(message)

def cv_save(filename, data, params=dflt_params):
  save_to = path.join(get_param("save_path", params), filename)
  cv.imwrite(save_to, data)

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
  img_h = len(first_img)
  img_w = len(first_img[0])
  if not all ((len(i) == img_h and len(i[0]) == img_w) for i in it):
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
  for f in found:
    print_message(f"Found points: {f[1]}", isVerbose=True, params=params)
  if (get_param("verbose", params)):
    idx_chess = 0
    for img, f in zip(cvimgs, found):
      drawable = img.copy()
      cv.drawChessboardCorners(drawable, chessboard_size, f[1], f[0])
      cv_save(f"chessboard_{idx_chess}.bmp", drawable, params=params)
      idx_chess += 1

  #
  # Calibrate images using the chessboard, to remove distortions.
  # See 718
  print("+++Calibrating images...")
  #
  calibration = None
  # Taken from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
  objP = np.zeros((get_param("chess_total",params), 3), np.float32)
  objP[:,:2] = np.mgrid[0:get_param("chess_col", params), 0:get_param("chess_row", params)].T.reshape(-1, 2)
  print_message(f"Using Object Points: {objP}", isVerbose=True, params=params)
  objpoints = []
  imgpoints = []
  # Here we assume all images had a chessboard found, which may not always be true if we change this in the future
  for i, pt in zip(cvimgs, [f[1] for f in found]):
    objpoints.append(objP)
    pt_refined = cv.cornerSubPix(i, pt, (11, 11), (-1, -1), get_param("criteria", params))
    imgpoints.append(pt_refined)

  ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, i.shape[::-1], None, None)
  print_message(f"Successfully calibrated! ret:{ret} mtx:{mtx} dist:{dist} rvecs:{rvecs} tvecs:{tvecs}", isVerbose=True, params=params)
  calibration = [{
    "ret":ret,
    "mtx":mtx,
    "dist":dist,
    "rvecs":rvecs,
    "tvecs":tvecs} for i in cvimgs]

  print("+++Undistorting images...")
  undistorted = []
  idx_undistort = 0
  for (i, cal) in zip(cvimgs, calibration):
    # Taken from: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    refined_mtx, roi = cv.getOptimalNewCameraMatrix(cal["mtx"], cal["dist"], (img_w, img_h), 0, newImgSize=(img_w, img_h))
    print_message(f"Calculated refined camera matrix and ROI. MTX: {refined_mtx}, ROI: {roi}", isVerbose=True, params=params)
    undist = cv.undistort(i, cal["mtx"], cal["dist"], None, None)
    x,y,w,h = roi
    undist = undist[y:y+h, x:x+w]
    undistorted.append(undist)
    if (get_param("verbose", params)):
      cv_save(f"undistorted_{idx_undistort}.bmp", undist, params=params)
    idx_undistort += 1 

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
