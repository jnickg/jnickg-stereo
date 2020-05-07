import sys
import time
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
  "verbose":True,
  "do_resize":True,
  "max_wide_len":1000
}

def get_param(key, params):
  return params.get(key, dflt_params.get(key, None))

def print_message(message, isVerbose=True, params=dflt_params, do_newline=True):
  doVerbose = get_param("verbose", params)
  if (isVerbose is False or doVerbose is True):
    if (do_newline is True):
      print(f"\n\n{message}")
    else:
      print(message, end=" ")
    sys.stdout.flush()

def cv_save(filename, data, params=dflt_params):
  save_to = path.join(get_param("save_path", params), filename)
  print_message(f"Writing to file {save_to}...", params=params)
  cv.imwrite(save_to, data)

# Taken from https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
def cv_drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

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
    print_message("OK", params=params, do_newline=False)
    cvimgs.append(cvimg)

  #
  print_message("+++Validating input images...", isVerbose=False)
  #
  cvimg_iter = iter(cvimgs)
  first_img = next(cvimg_iter)
  scale_factor = get_param("max_wide_len", params=params) / max(first_img.shape[1], first_img.shape[0])
  img_h_orig = len(first_img)
  img_w_orig = len(first_img[0])
  if not all ((len(i) == img_h_orig and len(i[0]) == img_w_orig) for i in cvimg_iter):
    raise ValueError("Images must all be the same size.") # (TODO normalize image spaces?)
  print_message("Image shapes OK", params=params)
  if (get_param("do_resize", params=params)):
    cvimgs = [cv.resize(i, (int(i.shape[1] * scale_factor), int(i.shape[0] * scale_factor)), interpolation=cv.INTER_AREA) for i in cvimgs]
    img_h = cvimgs[0].shape[0]
    img_w = cvimgs[0].shape[1]
    print_message(f"Resized images from {img_w_orig}x{img_h_orig} to {img_w}x{img_h}")
  else:
    img_h = img_h_orig
    img_w = img_w_orig
  print_message(f"Image sizes OK: {img_w}x{img_h}", params=params)
  if not all ((f is True) for f, _ in [cv.findChessboardCorners(i, chessboard_size, flags=cv.CALIB_CB_FAST_CHECK) for i in cvimgs]):
    raise ValueError("Not all images have chessboard patterns")
  print_message("Image contents OK", params=params)
  
  time.sleep(1.0)

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
  objpoints = []
  imgpoints = []
  # Here we assume all images had a chessboard found, which may not always be true if we change validation above
  for i, pt in zip(cvimgs, [f[1] for f in found]):
    objpoints.append(objP)
    pt_refined = cv.cornerSubPix(i, pt, (11, 11), (-1, -1), get_param("criteria", params))
    imgpoints.append(pt_refined)

  ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, i.shape[::-1], None, None)
  print_message(f"Successfully calibrated!\nret: {ret}\nmtx:\n{mtx}\ndist:\n{dist}\nrvecs:\n{rvecs}\ntvecs:\n{tvecs}", isVerbose=True, params=params)
  # Taken from: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
  refined_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (img_w, img_h), 0, newImgSize=(img_w, img_h))
  print_message(f"Calculated refined camera matrix and ROI.\nMTX: {refined_mtx},\nROI: {roi}", isVerbose=True, params=params)
  
  print_message("+++Undistorting images...", isVerbose=False)
  undistorted = []
  idx_undistort = 0
  for i in cvimgs:
    undist = cv.undistort(i, mtx, dist, None, newCameraMatrix=refined_mtx)
    x,y,w,h = roi
    undist = undist[y:y+h, x:x+w]
    undistorted.append(undist)
    if (get_param("verbose", params)):
      cv_save(f"undistorted_{idx_undistort}.bmp", undist, params=params)
    idx_undistort += 1

  # Might have useful stuff: https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/cbunch7/index.html

  # Here it may be useful to segment the images and then look there for keypoints using a mask

  print_message("+++Attempting to resolve epilines with Chessboard points...", isVerbose=False)


  #
  print_message("+++Extracting feature points from undistorted images...", isVerbose=False)
  # Inspired by: https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
  #
  feature_finder = cv.ORB_create(nfeatures=1000)
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
  idx_left_match = 0
  idx_right_match = 1
  for pair in itt.combinations(zip(undistorted, feature_results), 2):
    (left_img, (left_kp, left_des)), (right_img, (right_kp, right_des))  = pair

    FLANN_INDEX_KDTREE = 1 # for SIFT
    FLANN_INDEX_LSH = 6 # for ORB
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)

    # https://docs.opencv.org/2.4/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html
    matcher = cv.FlannBasedMatcher(index_params, search_params)
    knn_matches = matcher.knnMatch(left_des, right_des, k=2)
    print_message(f"Matcher found {len(knn_matches)} matches.", params=params)
    knn_matches = [match for match in knn_matches if len(match) == 2]
    print_message(f"Kept {len(knn_matches)} matches after removing non-pairs.", params=params)

    good = []
    left_pts = []
    right_pts = []

    # ratio test as per Lowe's paper: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    idx_match = 0
    for _, match_tuple in enumerate(knn_matches):
      idx_match += 1
      m, n = match_tuple
      if m.distance < 0.8 * n.distance:
        good.append(m)
        right_pts.append(left_kp[m.trainIdx].pt)
        left_pts.append(right_kp[m.queryIdx].pt)
    print_message(f"Found {len(good)} good matches between images", params=params)

    left_pts = np.int32(left_pts)
    right_pts = np.int32(right_pts)
    F, mask = cv.findFundamentalMat(left_pts, right_pts, cv.FM_RANSAC)
    print_message(f"Found Fundamental Matrix F:\n{F}", params=params)

    # Select only inliers
    left_pts = left_pts[mask.ravel()==1]
    right_pts = right_pts[mask.ravel()==1]
    print_message(f"There are {len(left_pts)} inliers for left, {len(right_pts)} inliers for right.", params=params)

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(right_pts.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(left_pts.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)

    if (get_param("verbose", params)):
      left_lines, _  = cv_drawlines(left_img,right_img,lines1,left_pts,right_pts)
      right_lines, _ = cv_drawlines(right_img,left_img,lines2,right_pts,left_pts)
      cv_save(f"epilines_{idx_left_match}_{idx_right_match}.bmp", left_lines, params=params)
      cv_save(f"epilines_{idx_right_match}_{idx_left_match}.bmp", right_lines, params=params)


    # No threshold because we selected only inliers, above
    rv, H1, H2 = cv.stereoRectifyUncalibrated(left_pts, right_pts, F, (img_w, img_h), threshold=0.0)
    if rv is not True:
      raise ValueError("Failed to rectify images.")
    print_message(f"Calculated Homography Matrices:\nH1: {H1}\nH2: {H2}", params=params)

    # See last comment in https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#initundistortrectifymap
    # Also slide 25: http://ece631web.groups.et.byu.net/Lectures/ECEn631%2014%20-%20Calibration%20and%20Rectification.pdf
    R1 = np.linalg.inv(mtx).dot(H1).dot(mtx)
    map1_1, map1_2 = cv.initUndistortRectifyMap(mtx, dist, R1, refined_mtx, (img_w, img_h), cv.CV_32FC1)
    print_message(f"Rectify maps:\nmap1_1: {map1_1}\nmat1_2: {map1_2} ", params=params)
    R2 = np.linalg.inv(mtx).dot(H2).dot(mtx)
    map2_1, map2_2 = cv.initUndistortRectifyMap(mtx, dist, R2, refined_mtx, (img_w, img_h), cv.CV_32FC1)
    print_message(f"Rectify maps:\nmap2_1: {map2_1}\nmat2_2: {map2_2} ", params=params)

    if (get_param("verbose", params)):
      left_remap = cv.remap(left_img, map1_1, map1_2, cv.INTER_LANCZOS4)
      right_remap = cv.remap(right_img, map2_1, map2_2, cv.INTER_LANCZOS4)
      cv_save(f"remap_{idx_left_match}_{idx_right_match}.bmp", left_remap, params=params)
      cv_save(f"remap_{idx_right_match}_{idx_left_match}.bmp", right_remap, params=params)

    idx_left_match += 1
    idx_right_match += 1


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
