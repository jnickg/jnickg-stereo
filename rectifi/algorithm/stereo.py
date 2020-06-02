import sys
import time
import cv2 as cv
import numpy as np
import os.path as path
import itertools as itt

dflt_params = {
  "fmt":".bmp",
  "debug": True,
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

def print_message(message, is_verbose=True, is_debug=False, params=dflt_params, do_newline=True):
  do_verbose = get_param("verbose", params)
  do_debug = get_param("debug", params)
  if ((is_verbose is False or do_verbose is True) and (is_debug is False or do_debug is True)):
    if (do_newline is True):
      print(f"\n\n{message}")
    else:
      print(message, end=" ")
    sys.stdout.flush()

def cv_save(filename, data, params=dflt_params):
  if get_param("debug", params) is False:
    print_message("Skipping file save -- not in Debug Mode!", is_verbose=False)
    return
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
        #print(f"pt1: {pt1}")
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def sharpen_image(image, sharpen_thresh, p=dflt_params):
  b = 0.0
  while (b < sharpen_thresh):
    b = cv.Laplacian(image, cv.CV_64F).var()
    kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
    image = cv.filter2D(image, -1, kernel)
  return image

def get_matcher(kp_type='SIFT', params=dflt_params):
  search_params = dict(checks=50)

  index_params = None
  if kp_type == 'SIFT':
    FLANN_INDEX_KDTREE = 1 # for SIFT
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=16)
  elif kp_type == 'ORB':
    FLANN_INDEX_LSH = 6 # for ORB
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)

  # https://docs.opencv.org/2.4/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html
  return cv.FlannBasedMatcher(index_params, search_params)

def do_kp_correspondence(left_kp, left_des, right_kp, right_des, kp_type='SIFT', params=dflt_params):
  matcher = get_matcher(kp_type=kp_type, params=params)
  matches = matcher.match(left_des, right_des)
  print_message(f"Matcher found {len(matches)} matches.", params=params)
  match_distances = [m.distance for m in matches]

  print_message('Match distance: min:  %.3f' % min(match_distances), params=params)
  print_message('Match distance: mean: %.3f' % (sum(match_distances) / len(match_distances)), params=params)
  print_message('Match distance: max:  %.3f' % max(match_distances), params=params)

  flann_good = []
  flann_left_pts = []
  flann_right_pts = []

  flann_good = matcher.radiusMatch(left_des, right_des, max(2 * min(match_distances), 0.02))
  flann_good_printable = [(left_kp[m[0].queryIdx].pt, right_kp[m[0].trainIdx].pt) for m in flann_good if len(m) > 0]
  print_message(f"Found {len(flann_good_printable)} good matches between images using FLANN-based RADIUS matching.", params=params, is_verbose=False)
  print_message(f"Good matches:\n{flann_good_printable}", params=params, is_debug=True)
  flann_left_pts = [left_kp[m[0].queryIdx].pt for m in flann_good if len(m) > 0]
  flann_right_pts = [right_kp[m[0].trainIdx].pt for m in flann_good if len(m) > 0]

  return flann_left_pts, flann_right_pts

def do_kp_correspondence_knn(left_kp, left_des, right_kp, right_des, kp_type='SIFT', params=dflt_params):
  matcher = get_matcher(kp_type=kp_type, params=params)
  matches = matcher.knnMatch(left_des, right_des, k=2)
  print_message(f"Matcher found {len(matches)} matches.", params=params)

  flann_good = []
  flann_left_pts = []
  flann_right_pts = []

  # ratio test as per Lowe's paper: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
  for _, k_best_matches in enumerate(matches):
    match1, match2 = k_best_matches
    if match1.distance < 0.8 * match2.distance:
      flann_good.append(match1)
      flann_left_pts.append(left_kp[match1.queryIdx].pt)
      flann_right_pts.append(right_kp[match1.trainIdx].pt)
  good_printable = [(left_kp[match1.queryIdx].pt, right_kp[match1.trainIdx].pt) for m in flann_good]
  print_message(f"Found {len(flann_good)} good matches between images using FLANN-based Ratio-test matching.", params=params, is_verbose=False)
  print_message(f"Good matches:\n{good_printable}", params=params, is_debug=True)

  return flann_left_pts, flann_right_pts

def do_epilines(F, left_img, left_idx, left_pts, right_img, right_idx, right_pts, prefix='epilines', p=dflt_params):
  # Find epilines corresponding to points in right image (second image) and
  # drawing its lines on left image
  lines1 = cv.computeCorrespondEpilines(right_pts.reshape(-1,1,2), 2, F)
  lines1 = lines1.reshape(-1,3)
  print_message(f"Computed Epilines lines1:\n{lines1}", params=p)
  # Find epilines corresponding to points in left image (first image) and
  # drawing its lines on right image
  lines2 = cv.computeCorrespondEpilines(left_pts.reshape(-1,1,2), 1, F)
  lines2 = lines2.reshape(-1,3)
  print_message(f"Computed Epilines lines2:\n{lines2}", params=p)

  epiline_images = []
  left_lines, _  = cv_drawlines(left_img,right_img,lines1,left_pts,right_pts)
  epiline_images.append((left_lines, f"{prefix}_{left_idx}-{right_idx}.bmp"))
  right_lines, _ = cv_drawlines(right_img,left_img,lines2,right_pts,left_pts)
  epiline_images.append((right_lines, f"{prefix}_{right_idx}-{left_idx}.bmp"))
  if (get_param("verbose", p)):
    for img, filename in epiline_images:
      cv_save(filename, img, params=p)
  return epiline_images

def do_rectification(intrinsic_matrix,
                     distortion_coeffs,
                     refined_mtx,
                     rectify_F,
                     size,
                     left_img, left_idx, left_pts,
                     right_img, right_idx, right_pts,
                     prefix="remap", p=dflt_params):
  img_w, img_h = size
  # No threshold because we selected only inliers, above
  cal_success, H1, H2 = cv.stereoRectifyUncalibrated(
    left_pts,
    right_pts,
    rectify_F,
    (img_w, img_h),
    threshold=3.0)
  if cal_success is not True:
    raise ValueError("Failed to rectify images.")
  print_message(f"Calculated Homography Matrices:\nH1: {H1}\nH2: {H2}", params=p)

  # See last comment in https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#initundistortrectifymap
  # Also slide 25: http://ece631web.groups.et.byu.net/Lectures/ECEn631%2014%20-%20Calibration%20and%20Rectification.pdf
  R1 = np.linalg.inv(intrinsic_matrix).dot(H1).dot(intrinsic_matrix)
  R2 = np.linalg.inv(intrinsic_matrix).dot(H2).dot(intrinsic_matrix)
  print_message(f"R Matrices for undistort mappings\nR1: {R1}\nR2: {R2}")

  map1_1, map1_2 = cv.initUndistortRectifyMap(
    intrinsic_matrix,
    distortion_coeffs,
    R1,
    refined_mtx,
    (img_w, img_h),
    cv.CV_32FC1)
  map2_1, map2_2 = cv.initUndistortRectifyMap(
    intrinsic_matrix,
    distortion_coeffs,
    R2,
    refined_mtx,
    (img_w, img_h),
    cv.CV_32FC1)

  print_message(f"Rectify maps:\nmap1_1: {map1_1}\nmat1_2: {map1_2} ", params=p)
  print_message(f"Rectify maps:\nmap2_1: {map2_1}\nmat2_2: {map2_2} ", params=p)

  rectified_images = []
  left_remap = cv.remap(left_img, map1_1, map1_2, cv.INTER_LANCZOS4)
  rectified_images.append((left_remap, f"{prefix}_{left_idx}-{right_idx}.bmp"))
  right_remap = cv.remap(right_img, map2_1, map2_2, cv.INTER_LANCZOS4)
  rectified_images.append((right_remap, f"{prefix}_{right_idx}-{left_idx}.bmp"))
  if (get_param("verbose", p)):
    for img, filename in rectified_images:
      cv_save(filename, img, params=p)
  return rectified_images

def rectify(image_buffers, params=dflt_params):
  """Rectifies the given images

  Parameters:
    image_buffers (array): Array of image buffers, as in calling numpy.fromfile('some/file', dtype=np.uint8) for each.
    fmt: A valid image extension (default: '.bmp') representing the format to which the output buffer will be encoded

  Returns:
    TBD
  """
  output = []
  chessboard_size = get_param("chess_sz", params)

  #
  print_message("+++Reading input images, decoding to processing format...", is_verbose=False)
  #
  cvimgs = []
  for imgbuf in image_buffers:
    cvimg = cv.imdecode(imgbuf, cv.IMREAD_GRAYSCALE)
    if cvimg is None:
      raise ValueError("Failed to decode provided image")
    print_message("OK", params=params, do_newline=False)
    cvimgs.append(cvimg)

  #
  print_message("+++Validating input images...", is_verbose=False)
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

  print_message("+++Filtering for calibration & subject subsets...", is_verbose=False)
  cvimgs_chessboard = []
  cvimgs_subject = []
  for i, idx in zip(cvimgs, [i for i, n in enumerate(cvimgs)]):
    print_message(f"  Image {idx}:", params=params, do_newline=False)
    f, _ = cv.findChessboardCorners(i, chessboard_size, flags=cv.CALIB_CB_FAST_CHECK)
    if f is True:
      cvimgs_chessboard.append(i)
      print_message(f"CALIB\n", params=params, do_newline=False)
    else:
      cvimgs_subject.append(i)
      print_message(f"SUBJ\n", params=params, do_newline=False)
  if len(cvimgs_chessboard) < 10:
    print_message("WARNING: Need at least 10 chessboard-pattern images. Output quality may be undesirable.", params=params)
  print_message("Image contents OK", params=params)

  time.sleep(1.0)

  #
  # Capture chessboard corners from the calibration matrix we assume to be in
  # the photos that were filtered above.
  # See 717
  print_message("+++Finding calibration grid...", is_verbose=False)
  #
  found = [cv.findChessboardCorners(i, chessboard_size) for i in cvimgs_chessboard]
  if not all (f[0] is True for f in found):
    raise ValueError("Unable to find calibration points for all calibration images.")
  if (get_param("verbose", params)):
    idx_chess = 0
    for img, f in zip(cvimgs_chessboard, found):
      print_message(f"Found points for image {idx_chess}:\n{f[1]}", is_verbose=True, params=params)
      drawable = img.copy()
      cv.drawChessboardCorners(drawable, chessboard_size, f[1], f[0])
      cv_save(f"chessboard_{idx_chess}.bmp", drawable, params=params)
      idx_chess += 1

  #
  # Calibrate images using the chessboard, to remove distortions.
  # See 718
  print_message("+++Calibrating camera based on images...", is_verbose=False)
  #

  # Taken from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
  objP = np.zeros((get_param("chess_total",params), 3), np.float32)
  objP[:,:2] = np.mgrid[0:get_param("chess_col", params), 0:get_param("chess_row", params)].T.reshape(-1, 2)
  objpoints = []
  imgpoints = []
  # Here we assume all calibration images had a chessboard found, which may not always be true if we
  # change validation above
  for i, pt in zip(cvimgs_chessboard, [f[1] for f in found]):
    objpoints.append(objP)
    pt_refined = cv.cornerSubPix(i, pt, (11, 11), (-1, -1), get_param("criteria", params))
    imgpoints.append(pt_refined)

  calibrate_output = cv.calibrateCamera(
    objpoints,
    imgpoints,
    i.shape[::-1],
    None,
    None,
    rvecs=None,
    tvecs=None,
    flags=(cv.CALIB_ZERO_TANGENT_DIST | cv.CALIB_FIX_PRINCIPAL_POINT))
  calib_error_rms, intrinsic_matrix, distortion_coeffs, rotation_vecs, translation_vecs = calibrate_output
  print_message(
    f"Successfully calibrated!\n"
    f"rms: {calib_error_rms}\n"
    f"mtx:\n{intrinsic_matrix}\n"
    f"dist:\n{distortion_coeffs}\n"
    f"rvecs:\n{rotation_vecs}\n"
    f"tvecs:\n{translation_vecs}",
    is_verbose=True, params=params)
  # Taken from: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
  refined_mtx, roi = cv.getOptimalNewCameraMatrix(
    intrinsic_matrix,
    distortion_coeffs,
    (img_w, img_h),
    0,
    newImgSize=(img_w, img_h))
  img_orig_x_new, img_orig_y_new, img_w_new, img_h_new = roi
  print_message(f"Calculated refined camera matrix and ROI.\nMTX: {refined_mtx},\nROI: {roi}", is_verbose=True, params=params)
  
  #
  print_message("+++Undistorting images...", is_verbose=False)
  #
  undistorted = []
  idx_undistort = 0
  for i in cvimgs_subject:
    undist = cv.undistort(i, intrinsic_matrix, distortion_coeffs, None, newCameraMatrix=refined_mtx)
    x,y,w,h = roi
    undist = undist[y:y+h, x:x+w]
    undistorted.append(undist)
    if (get_param("verbose", params)):
      cv_save(f"undistorted_{idx_undistort}.bmp", undist, params=params)
    idx_undistort += 1

  #
  print_message("+++Extracting feature points from undistorted images...", is_verbose=False)
  # Inspired by: https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
  #
  feature_finder = cv.ORB_create(nfeatures=6000)
  feature_results = []
  for i in undistorted:
    kp = feature_finder.detect(i, None)
    kp, des = feature_finder.compute(i, kp)
    feature_results.append((kp, des))

  idx_feature = 0
  for i, kp, des in zip(undistorted, [kp for kp, _ in feature_results], [des for _, des in feature_results]):
    print_message(f"Found {len(kp)} keypoints in image {idx_feature}.", params=params)
    print_message(f"Computed {len(des)} descriptors in image {idx_feature}:\n{des}", params=params)
    kp_img = cv.drawKeypoints(i, kp, None, color=(0,102,255), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_img_filename = f"keypoints_{idx_feature}.bmp"
    output.append((kp_img, kp_img_filename))
    if (get_param("verbose", params)):
      cv_save(kp_img_filename, kp_img, params=params)
    idx_feature += 1

  #
  # For every pair of images & points, calculate the fundamental matrix (homographies)
  # Taken from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
  print_message("+++Matching feature points between image pairs, then attempting rectification...", is_verbose=False)
  # Below we use undistorted, NOT cvimgs_subject, because of the note in under:
  # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#stereorectifyuncalibrated
  #
  pair_counter = 0

  for pair in itt.combinations(zip(undistorted, feature_results, [i for i, n in enumerate(undistorted)]), 2):
    (left_img, (left_kp, left_des), left_idx), (right_img, (right_kp, right_des), right_idx)  = pair
    print_message(f"Working on images {left_idx} and {right_idx}...", params=params)

    left_matches, right_matches = do_kp_correspondence_knn(
      left_kp,
      left_des,
      right_kp,
      right_des,
      kp_type='ORB',
      params=params)

    left_matches = np.int32(left_matches)
    right_matches = np.int32(right_matches)
    fundamental_mtx, inlier_mask = cv.findFundamentalMat(
      left_matches,
      right_matches,
      method=cv.FM_RANSAC,
      ransacReprojThreshold=1.0,
      confidence=0.9999)
    print_message(f"Found Fundamental Matrix flann_F:\n{fundamental_mtx}", params=params)

    # If we pass seven matches into findFundamentalMat, we may have gotten three fundamental
    # matrices smashed into one. I guess that made sense to someone?
    # See note here: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#findfundamentalmat
    if len(left_matches) == 7:
      fundamental_mtx = fundamental_mtx[0:3,0:3]
      print_message(f"Truncated Fundamental Matrix flann_F:\n{fundamental_mtx}", params=params)

    left_matches = left_matches[inlier_mask.ravel()==1]
    right_matches = right_matches[inlier_mask.ravel()==1]
    print_message(f"There are {len(left_matches)} inliers for left, {len(right_matches)} inliers for right (FLANN matching).", params=params)

    # TODO Check return value of findFundamentalMat to see if we are forming a degenerate config

    epilines = do_epilines(
      fundamental_mtx,
      left_img, left_idx, left_matches,
      right_img, right_idx, right_matches,
      prefix=f"epilines_pair{pair_counter}", p=params)
    
    for tup in epilines:
      output.append(tup)

    remaps = do_rectification(
      intrinsic_matrix,
      distortion_coeffs,
      refined_mtx,
      fundamental_mtx,
      (img_w_new, img_h_new),
      left_img, left_idx, left_matches,
      right_img, right_idx, right_matches,
      prefix=f"remap_pair{pair_counter}", p=params)

    for tup in remaps:
      output.append(tup)

    pair_counter += 1

  return output
