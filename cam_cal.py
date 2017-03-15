import pickle
import cv2
import numpy as np
import glob
import os


# calibrate image size
img_size = (1280, 720)

# grid size
nx = 9
ny = 6

def cam_cali():

	# prepare object points, like [[0,0,0], [1,0,0], ... , [8,5,0]]
	objp = np.zeros((nx*ny,3), np.float32)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

	# List to store object points and image points from all calibrate images
	objpoints = [] # 3d points in real word
	imgpoints = [] # 2d points in image plane


	# Make a list of calibration images
	root_files = './camera_cal/calibration'
	
	for id in range(1,21):
		# read image
		img = cv2.imread(root_files + str(id) + '.jpg')
		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

		# If found, add object points and image points
		if ret == True:
		    print('Found corners in calibration' + str(id))
		    objpoints.append(objp)
		    imgpoints.append(corners)

		    # Draw and write the corner images
		    cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
		    write_name = './camera_cal/corner_found_' + str(id) + '.jpg' 
		    cv2.imwrite(write_name, img)

	# Camera calibration given object points and image points
	ret, mtx, dist, rvects, tvects = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

	# Save the camera calibartion result for later use
	dic_cam_cal = {}
	dic_cam_cal['mtx'] = mtx
	dic_cam_cal['dist'] = dist
	
	return dic_cam_cal

if __name__ == '__main__':
	cam_cali_file = './params/cam_cali_pickle.p'
	if not os.path.exists(cam_cali_file):
		dic_cam_cal = cam_cali()
		pickle.dump(dic_cam_cal, open(cam_cali_file, 'wb'))