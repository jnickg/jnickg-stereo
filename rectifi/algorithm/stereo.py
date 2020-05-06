import cv2 as cv
import numpy as np
import os.path as path
import itertools as itt

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
    print(f"\n\n{message}")

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
  chessboard_size = get_param("chess_sz", params)


  #
  print_message("+++Reading input images, decoding to processing format...", isVerbose=False)
  #
  cvimgs = []
  for imgbuf in image_buffers:
    cvimg = cv.imdecode(imgbuf, cv.IMREAD_GRAYSCALE)
    if cvimg is None:
      raise ValueError("Failed to decode provided image")
    cvimgs.append(cvimg)

  #
  print_message("+++Validating input images...", isVerbose=False)
  #
  cvimg_iter = iter(cvimgs)
  first_img = next(cvimg_iter)
  img_h = len(first_img)
  img_w = len(first_img[0])
  if not all ((len(i) == img_h and len(i[0]) == img_w) for i in cvimg_iter):
    raise ValueError("Images must all be the same size.") # (TODO normalize image spaces?)
  if not all ((f is True) for f, _ in [cv.findChessboardCorners(i, chessboard_size, flags=cv.CALIB_CB_FAST_CHECK) for i in cvimgs]):
    raise ValueError("Not all images have chessboard patterns")

  #
  # Capture chessboard corners from the calibration matrix we assume to be in
  # the photo. If any of the photos don't have a matrix, dump it and see if we
  # can continue without them
  # See 717
  print_message("+++Finding chessboard corners...", isVerbose=False)
  #
  found = [cv.findChessboardCorners(i, chessboard_size) for i in cvimgs]
  if not all (f[0] is True for f in found):
    raise ValueError("Unable to find calibration points for all images.")
  if (get_param("verbose", params)):
    idx_chess = 0
    for img, f in zip(cvimgs, found):
      print_message(f"Found points for image {idx_chess}:\n{f[1]}", isVerbose=True, params=params)
      drawable = img.copy()
      cv.drawChessboardCorners(drawable, chessboard_size, f[1], f[0])
      cv_save(f"chessboard_{idx_chess}.bmp", drawable, params=params)
      idx_chess += 1

  #
  # Calibrate images using the chessboard, to remove distortions.
  # See 718
  print_message("+++Calibrating camera based on images...", isVerbose=False)
  #

  # Taken from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
  objP = np.zeros((get_param("chess_total",params), 3), np.float32)
  objP[:,:2] = np.mgrid[0:get_param("chess_col", params), 0:get_param("chess_row", params)].T.reshape(-1, 2)
  print_message(f"Using Object Points:\n{objP}", isVerbose=True, params=params)
  objpoints = []
  imgpoints = []
  # Here we assume all images had a chessboard found, which may not always be true if we change validation above
  for i, pt in zip(cvimgs, [f[1] for f in found]):
    objpoints.append(objP)
    pt_refined = cv.cornerSubPix(i, pt, (11, 11), (-1, -1), get_param("criteria", params))
    imgpoints.append(pt_refined)

  ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, i.shape[::-1], None, None)
  print_message(f"Successfully calibrated!\nret: {ret}\nmtx:\n{mtx}\ndist:\n{dist}\nrvecs:\n{rvecs}\ntvecs:\n{tvecs}", isVerbose=True, params=params)

  print_message("+++Undistorting images...", isVerbose=False)
  undistorted = []
  idx_undistort = 0
  for i in cvimgs:
    # Taken from: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    refined_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (img_w, img_h), 0, newImgSize=(img_w, img_h))
    print_message(f"Calculated refined camera matrix and ROI.\nMTX: {refined_mtx},\nROI: {roi}", isVerbose=True, params=params)
    undist = cv.undistort(i, mtx, dist, None, None)
    x,y,w,h = roi
    undist = undist[y:y+h, x:x+w]
    undistorted.append(undist)
    if (get_param("verbose", params)):
      cv_save(f"undistorted_{idx_undistort}.bmp", undist, params=params)
    idx_undistort += 1

  # Might have useful stuff: https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/cbunch7/index.html

  #
  print_message("+++Extracting feature points from undistorted images...", isVerbose=False)
  # Inspired by: https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
  #
  feature_finder = cv.ORB_create()
  feature_results = []
  for i in undistorted:
    kp = feature_finder.detect(i, None)
    kp, des = feature_finder.compute(i, kp)
    feature_results.append((kp, des))

  if (get_param("verbose", params)):
    idx_feature = 0
    for i, kp, des in zip(undistorted, [kp for kp, _ in feature_results], [des for _, des in feature_results]):
      print_message(f"Found {len(kp)} keypoints in image {idx_feature}.", params=params)
      print_message(f"Computed {len(des)} descriptors in image {idx_feature}:\n{des}", params=params)
      kp_img = cv.drawKeypoints(i, kp, None, color=(0,102,255), flags=0)
      cv_save(f"keypoints_{idx_feature}.bmp", kp_img, params=params)
      idx_feature += 1

  #
  # For every pair of images & points, calculate the fundamental matrix (homographies)
  # Taken from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
  print_message("+++Matching feature points between image pairs...", isVerbose=False)
  #
  for pair in itt.combinations(zip(undistorted, feature_results), 2):
    left_side, right_side = pair
    left_img = left_side[0]
    left_kp  = left_side[1][0]
    left_des = left_side[1][1]
    right_img = right_side[0]
    right_kp  = right_side[1][0]
    right_des = right_side[1][1]
    FLANN_INDEX_KDTREE = 1 # for SIFT
    FLANN_INDEX_LSH = 6 # for ORB
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)

    matcher = cv.FlannBasedMatcher(index_params, search_params)
    knn_matches = matcher.knnMatch(left_des, right_des, k=2)
    print_message(f"Matcher found {len(knn_matches)} matches.", params=params)
    knn_matches = [match for match in knn_matches if len(match) == 2]
    print_message(f"Kept {len(knn_matches)} matches after removing non-pairs.", params=params)

    good = []
    left_pts = []
    right_pts = []

    # ratio test as per Lowe's paper
    idx_match = 0
    for _, match_tuple in enumerate(knn_matches):
      idx_match += 1
      m, n = match_tuple
      if m.distance < 0.8 * n.distance:
        good.append(m)
        right_pts.append(left_kp[m.trainIdx].pt)
        left_pts.append(right_kp[m.queryIdx].pt)
    print_message(f"Found {len(good)} good matches between images", params=params)
  #
  # Find homographies between pairs of images. See 663
  #
  #cv.findHomography
  #cv.RANSAC
  #

  #
  # Compute Fundamental matrix between images. This matrix relates the two
  # images together (in pixel space rather than physical space, as E does),
  # so we will need to do this for all images.
  # See 719
  print_message("+++Computing fundamental matrix between images...", isVerbose=False)
  #

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
