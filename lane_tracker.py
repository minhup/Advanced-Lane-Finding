from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np 
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
from scipy import ndimage
#matplotlib inline

class lane_tracker():

	def __init__(self, Mynwindows, Mymargin, Myminpix, My_cal_mtx, My_cal_dist, pers_M, pers_Minv, My_sanity_thresh, My_ym = 1, My_xm = 1, Mysmooth_factor = 10):
		# number of windows
		self.nwindows = Mynwindows
		# margin of region interest 
		self.margin = Mymargin
		
		self.left_sanity_thresh = My_sanity_thresh
		self.right_sanity_thresh = My_sanity_thresh

		self.pass_left_check = True
		self.pass_right_check = True 
		# Set minimum number of pixels found to recenter window
		self.minpix = Myminpix
		# meters per pixel in x direction
		self.xm_per_pix = My_xm
		# meters per pixel in y direction
		self.ym_per_pix = My_ym
		# smooth factor for averaging lane line coefficients
		self.smooth_factor = Mysmooth_factor
		# was the line detected in the last iteration?
		self.detected = False  
		# x values of the last n fits of the line
		#self.last_lane = (np.zeros(3),np.zeros(3))
		self.last_lane = []

		self.recent_left_xfitted = []
		self.recent_right_xfitted = []


		self.recent_xfitted = [] 
		
		self.radius_of_curvature = [] 
		
		self.image_size = (1280,720)
		
		self.M = pers_M
		self.Minv = pers_Minv
		self.cal_mtx = My_cal_mtx 
		self.cal_dist = My_cal_dist

	def percentile(self,img,perc_thresh):
		thresh1 = int(perc_thresh)
		thresh2 = int((perc_thresh-thresh1)*100)

		flat = img.reshape(-1)
		perc = np.percentile(flat,thresh1)
		if thresh2 == 0:
			return perc
		else:
			flat = flat[flat >= perc]
			perc = np.percentile(flat,thresh2)
			return perc 

	def to_perc_binary(self, img, perc_thresh = 98, part = 1):
		binary = np.zeros_like(img)
		h = img.shape[0]//part
		thresh = np.ones_like(img)
		for i in range(part):
			perc = int(self.percentile(img[i*h:(i+1)*h],perc_thresh))
			thresh[i*h:(i+1)*h] *= perc
		binary[ img >= thresh ] = 255
		return binary

	
	def to_binary(self, img, roi=None, perc_L = 98, perc_B = 98, part = 1):
		LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
		L = LAB[:,:,0]
		B = LAB[:,:,2]

		
		if roi != None:
			L = L*roi 
			B = B*roi
		
		if perc_L != None:
			binary_L = self.to_perc_binary(L, perc_L, part)
		else:
			binary_L = np.zeros_like(L)

		if perc_B != None:
			binary_B = self.to_perc_binary(B, perc_B, part)
		else:
			binary_B = np.zeros_like(L)

		binary = np.zeros_like(L)
		binary[ (binary_L==255) | (binary_B==255) ] = 255
		return binary

	def first_track(self, img):

		image_size = (img.shape[1], img.shape[0])
		
		# perspective transform
		pers_img = cv2.warpPerspective(img,self.M,(img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR)
		
		binary = self.to_binary(pers_img, perc_L = 99.5, perc_B = 99.5, part = 1)

		
		# Take a histogram of the bottom half of the image
		histogram = np.sum(binary[binary.shape[0]//2:,:], axis=0)
		# Create an output image to draw on and  visualize the result
		out_img = np.dstack((binary, binary, binary))

		# Find the peak of the left and right halves of the histogram
		# These will be the starting point for the left and right lines
		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[230:midpoint]) + 230
		rightx_base = np.argmax(histogram[midpoint:1050]) + midpoint

		# Choose the number of sliding windows
		nwindows = self.nwindows
		# Set height of windows
		window_height = np.int(image_size[0]/nwindows)
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = binary.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Current positions to be updated for each window
		leftx_current = leftx_base
		rightx_current = rightx_base
		# Set the width of the windows +/- margin
		margin = self.margin 
		# Set minimum number of pixels found to recenter window
		minpix = self.minpix 
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []

		early_stop_left = False
		early_stop_right = False

		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = binary.shape[0] - (window+1)*window_height
			win_y_high = binary.shape[0] - window*window_height
			if not early_stop_left:
				win_xleft_low = leftx_current - margin
				win_xleft_high = leftx_current + margin
				good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
				left_lane_inds.append(good_left_inds)
				if len(good_left_inds) >= minpix:
					leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
					win_xleft_low = leftx_current - margin
					win_xleft_high = leftx_current + margin
					good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
					#left_lane_inds.append(good_left_inds)
					leftx_current = np.int(np.mean(nonzerox[good_left_inds]))



				if (win_xleft_low < -margin/2) or (win_xleft_high > image_size[0] + margin/2):
					early_stop_left = True

				cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),[255,0,0], 2) 	

			if not early_stop_right:
				win_xright_low = rightx_current - margin
				win_xright_high = rightx_current + margin
				# Identify the nonzero pixels in x and y within the window
				good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
				# Append these indices to the lists
				right_lane_inds.append(good_right_inds)
				# If you found > minpix pixels, recenter next window on their mean position
				if len(good_right_inds) > minpix:        
					rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
					win_xright_low = rightx_current - margin
					win_xright_high = rightx_current + margin
					# Identify the nonzero pixels in x and y within the window
					good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
					rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

				if (win_xright_low < -margin/2) or (win_xright_high > image_size[0] + margin/2):
					early_stop_right = True	
			
				# Draw the windows on the visualization image
				cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),[0,255,0], 2)

	
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)

		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds] 

		

		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

		self.last_lane = [left_fit, right_fit]
		self.detected = True

		for i in range(0,self.image_size[1]):
			out_img[i,max(min(int(left_fit[0]*i*i + left_fit[1]*i+left_fit[2]),self.image_size[0]-1),0)] = [255,0,0]
			out_img[i,max(min(int(right_fit[0]*i*i + right_fit[1]*i+right_fit[2]),self.image_size[0]-1),0)] = [0,255,0]

		return  out_img, left_fit, right_fit, pers_img

	def next_track(self, img):

		image_size = (img.shape[1], img.shape[0])
		
		# perspective transform
		pers_img = cv2.warpPerspective(img,self.M,image_size,flags=cv2.INTER_LINEAR)
		# define region of interest round last lane line
		ploty = np.linspace(0, image_size[1]-1, image_size[1])

		middle_lane_eq = np.mean(self.last_lane, axis = 0)

		middle_lane = middle_lane_eq[0]*ploty**2 + middle_lane_eq[1]*ploty + middle_lane_eq[2]
		

		left_roi = np.uint8(np.zeros(img.shape[:2]))
		right_roi = np.uint8(np.zeros(img.shape[:2]))


		for y in range(image_size[1]):
			middle = int( max( min( middle_lane[y],image_size[0]),0) )

			left_roi[y,:middle] = 1
			right_roi[y,middle:] = 1

			
		roi = np.ones(img.shape[:2])
		roi = cv2.warpPerspective(roi,self.M,image_size,flags=cv2.INTER_LINEAR)

		
		B = self.to_binary(pers_img, left_roi, perc_L = None, perc_B = 99, part = 1)
		
		L = self.to_binary(pers_img, right_roi, perc_L = 99.7, perc_B = None, part = 1)

		binary = np.zeros_like(L)
		binary[(B == 255) | (L == 255)] = 255

		out_img = np.dstack((binary, binary, binary))

		
		leftx_base = int( self.last_lane[0][0]*(image_size[1]-1)**2 + self.last_lane[0][1]*(image_size[1]-1) + self.last_lane[0][2]  )
		rightx_base = int( self.last_lane[1][0]*(image_size[1]-1)**2 + self.last_lane[1][1]*(image_size[1]-1) + self.last_lane[1][2]  )



		# Choose the number of sliding windows
		nwindows = self.nwindows
		# Set height of windows
		window_height = np.int(image_size[1]/nwindows)
		# Identify the x and y positions of all nonzero pixels in the image
		left_nonzero = B.nonzero()
		left_nonzeroy = np.array(left_nonzero[0])
		left_nonzerox = np.array(left_nonzero[1])
		# Current positions to be updated for each window

		right_nonzero = L.nonzero()
		right_nonzeroy = np.array(right_nonzero[0])
		right_nonzerox = np.array(right_nonzero[1])
		


		leftx_current = leftx_base
		rightx_current = rightx_base
		# Set the width of the windows +/- margin
		margin = self.margin 
		# Set minimum number of pixels found to recenter window
		minpix = self.minpix 
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []

		early_stop_left = False
		early_stop_right = False

		

		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = binary.shape[0] - (window+1)*window_height
			win_y_high = binary.shape[0] - window*window_height
			if not early_stop_left:
				offset = int(self.last_lane[0][0]*win_y_low*win_y_low + self.last_lane[0][1]*win_y_low  -self.last_lane[0][0]*win_y_high*win_y_high - self.last_lane[0][1]*win_y_high )
				win_xleft_low = leftx_current - margin
				win_xleft_high = leftx_current + margin
				good_left_inds = ((left_nonzeroy >= win_y_low) & (left_nonzeroy < win_y_high) & (left_nonzerox >= win_xleft_low) & (left_nonzerox < win_xleft_high)).nonzero()[0]
				left_lane_inds.append(good_left_inds)
				if len(good_left_inds) >= minpix:
					leftx_current = np.int(np.mean(left_nonzerox[good_left_inds]))
					
					win_xleft_low = leftx_current - margin
					win_xleft_high = leftx_current + margin
					good_left_inds = ((left_nonzeroy >= win_y_low) & (left_nonzeroy < win_y_high) & (left_nonzerox >= win_xleft_low) & (left_nonzerox < win_xleft_high)).nonzero()[0]
					leftx_current = np.int(np.mean(left_nonzerox[good_left_inds])) + offset
				
				else:
					leftx_current += offset 

				if (win_xleft_low < 0) or (win_xleft_high > image_size[0]):
					early_stop_left = True



				cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),[255,0,0], 2) 	

			if not early_stop_right:
				offset = int(self.last_lane[1][0]*win_y_low*win_y_low + self.last_lane[1][1]*win_y_low -self.last_lane[1][0]*win_y_high*win_y_high - self.last_lane[1][1]*win_y_high )
				win_xright_low = rightx_current - margin
				win_xright_high = rightx_current + margin
				good_right_inds = ((right_nonzeroy >= win_y_low) & (right_nonzeroy < win_y_high) & (right_nonzerox >= win_xright_low) & (right_nonzerox < win_xright_high)).nonzero()[0]
				right_lane_inds.append(good_right_inds)
				if len(good_right_inds) >= minpix:
					rightx_current = np.int(np.mean(right_nonzerox[good_right_inds]))
					win_xright_low = rightx_current - margin
					win_xright_high = rightx_current + margin
					good_right_inds = ((right_nonzeroy >= win_y_low) & (right_nonzeroy < win_y_high) & (right_nonzerox >= win_xright_low) & (right_nonzerox < win_xright_high)).nonzero()[0]
					rightx_current = np.int(np.mean(right_nonzerox[good_right_inds]))

				else:
					rightx_current += offset

				if (win_xright_low < 0) or (win_xright_high > image_size[0]):
					early_stop_right = True

				
				cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),[0,255,0], 2) 	




		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)

		# Extract left and right line pixel positions
		leftx = left_nonzerox[left_lane_inds]
		lefty = left_nonzeroy[left_lane_inds] 
		rightx = right_nonzerox[right_lane_inds]
		righty = right_nonzeroy[right_lane_inds]

		ratio = 0.03
		num_left_point = int(len(leftx)*ratio)
		num_right_point = int(len(rightx)*ratio)

		leftx = np.concatenate((leftx, np.ones(num_left_point)*leftx_base))
		lefty = np.concatenate((lefty, np.ones(num_left_point)*(image_size[1]-1)))

		rightx = np.concatenate((rightx, np.ones(num_left_point)*rightx_base))
		righty = np.concatenate((righty, np.ones(num_left_point)*(image_size[1]-1)))

		try:
			left_fit = np.polyfit(lefty, leftx, 2)
		except:
			left_fit = self.last_lane[0]
		
		try:
			right_fit = np.polyfit(righty, rightx, 2)
		except:
			right_fit = self.last_lane[1]

		
		if self.pass_left_check:
			left_thresh = self.left_sanity_thresh
		else:
			left_thresh = self.left_sanity_thresh
		if self.pass_right_check:
			right_thresh = self.right_sanity_thresh
		else:
			right_thresh = self.left_sanity_thresh
		
		self.pass_left_check, l_string = self.sanity_check(self.last_lane[0], left_fit, left_thresh)
		self.pass_right_check, r_string = self.sanity_check(self.last_lane[1], right_fit, right_thresh)


		if (self.pass_left_check, self.pass_right_check) == (True, False):
			right_fit = self.last_lane[1] + left_fit - self.last_lane[0]
			
			cv2.putText(out_img, 'Fail right sanity' + r_string,(610,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1) 

		elif (self.pass_left_check, self.pass_right_check) == (False, True):
			left_fit = self.last_lane[0] + right_fit - self.last_lane[1]
			cv2.putText(out_img, 'Fail left sanity' + l_string,(50,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

		elif (self.pass_left_check, self.pass_right_check) == (False, False):
			(left_fit, right_fit) = self.last_lane
			cv2.putText(out_img, 'Fail right sanity',(1000,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) 
			cv2.putText(out_img, 'Fail left sanity',(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

				
		self.recent_left_xfitted.append(left_fit)
		self.recent_right_xfitted.append(right_fit)

		if len(self.recent_left_xfitted) > self.smooth_factor:
			self.recent_left_xfitted.pop(0)
		if len(self.recent_right_xfitted) > self.smooth_factor:
			self.recent_right_xfitted.pop(0)

		mean_left_fitx = np.mean(self.recent_left_xfitted, axis = 0)
		mean_right_fitx	= np.mean(self.recent_right_xfitted, axis = 0)

		self.last_lane = (mean_left_fitx, mean_right_fitx)
		
		
		for i in range(0,self.image_size[1]):
			out_img[i,max(min(int(left_fit[0]*i*i + left_fit[1]*i+left_fit[2]),self.image_size[0]-1),0)] = [255,0,0]
			out_img[i,max(min(int(right_fit[0]*i*i + right_fit[1]*i+right_fit[2]),self.image_size[0]-1),0)] = [0,255,0]

		return out_img, mean_left_fitx, mean_right_fitx, pers_img
		
	def sanity_check(self, last_poly_fit, current_poly_fit, thresh):
		yvals = np.arange(0, self.image_size[1])


		current_lane = current_poly_fit[0]*yvals*yvals+current_poly_fit[1]*yvals+current_poly_fit[2]
		last_lane = last_poly_fit[0]*yvals*yvals+last_poly_fit[1]*yvals+last_poly_fit[2]
		diff_lane = np.abs(current_lane - last_lane)

		diff_lane[ (current_lane<0) | (current_lane>self.image_size[0]) | (last_lane<0) | (last_lane>self.image_size[0])] = 0

		result = True

		if np.max(diff_lane) > thresh:
			result = False
			return result, 'diff_lane' + str(np.max(diff_lane)) + ' : ' + str(np.argmax(diff_lane))

		if np.abs( current_lane[-1] - last_lane[-1] ) > thresh /3:
			result = False
			return result, 'base_point ' +  str( np.abs( current_lane[-1] - last_lane[-1] ) )

		return result, 'pass'
		


	def track(self,img):
		if self.detected:
			return self.next_track(img)
		else:
			return self.first_track(img)

	def prepare_output_blend(self, text1, text2, fist_windows, second_windows, img):

		h, w, c = img.shape

		# decide the size of thumbnail images
		thumb_ratio = 0.25
		thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

		# resize to thumbnails images from various stages of the pipeline
		thumb_first_windows = cv2.resize(fist_windows, dsize=(thumb_w, thumb_h))
		thumb_second_windows = cv2.resize(second_windows, dsize=(thumb_w, thumb_h))

		off_x, off_y = 20, 20

		# add a semi-transparent rectangle to highlight thumbnails on the left
		mask = cv2.rectangle(img.copy(), (0, 0), (w, 2*off_y + thumb_h), (0, 0, 0), thickness=cv2.FILLED)
		img_blend = cv2.addWeighted(src1=mask, alpha=0.2, src2=img, beta=0.8, gamma=0)

		# stitch thumbnails
		img_blend[off_y:off_y+thumb_h, off_x:off_x+thumb_w, :] = thumb_first_windows
		img_blend[off_y:off_y+thumb_h, 2*off_x+thumb_w:2*off_x+2*thumb_w, :] = thumb_second_windows
		#img_blend[off_y:off_y+thumb_h, 3*off_x+2*thumb_w:3*off_x+3*thumb_w, :] = thumb_labeling

		cv2.putText(img_blend, text1 ,(730,80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
		cv2.putText(img_blend, text2,(730,130),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

		return img_blend

	def process_image(self,img):
		image_size = (img.shape[1], img.shape[0])
		img = cv2.undistort(img, self.cal_mtx, self.cal_dist, None, self.cal_mtx)
		#image_size = (1280,720)

		window_width = 24
		
		out_img, left_fit, right_fit, pers_img = self.track(img)
		#left_fit, right_fit = self.track(img)

		yvals = np.linspace(0, image_size[1]-1, image_size[1])

		#window_height = 72
		#res_yvals = np.arange(image_size[1]-(window_height/2),0,-window_height)

		#left_fit = np.polyfit(res_yvals,leftx,2)
		left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
		left_fitx = np.array(left_fitx, np.int32)

		right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
		right_fitx = np.array(right_fitx, np.int32)

		left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0), 
		np.concatenate((yvals,yvals[::-1]),axis=0) )),np.int32)
		right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0),
		np.concatenate((yvals,yvals[::-1]),axis=0) )),np.int32)
		inner_lane = np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2),axis=0),
		np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

		road = np.zeros_like(img)
		road_bkg = np.zeros_like(img)

		cv2.fillPoly(road,[left_lane],color=[255,0,0])
		cv2.fillPoly(road,[right_lane],color=[0,0,255])
		cv2.fillPoly(road,[inner_lane],color=[0,255,0])
		cv2.fillPoly(road_bkg,[left_lane],color=[255,255,255])
		cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255])

		road_warped = cv2.warpPerspective(road,self.Minv,image_size,flags=cv2.INTER_LINEAR)
		road_warped_bkg = cv2.warpPerspective(road_bkg,self.Minv,image_size,flags=cv2.INTER_LINEAR)

		base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
		result = cv2.addWeighted(img, 1.0, road_warped, 1.0, 0.0)

		ym_per_pix = self.ym_per_pix
		xm_per_pix = self.xm_per_pix

		middle_fit = left_fit

		A = left_fit[0]*xm_per_pix/np.square(ym_per_pix)
		B = left_fit[1]*xm_per_pix/ym_per_pix
		C = left_fit[2]*xm_per_pix
		y = yvals[-1]*ym_per_pix

		eps = 1e-8
		curverad = ((1+(2*A*y+B)**2)**1.5)/(np.absolute(2*A) + eps)
		self.radius_of_curvature.append(curverad)
		if len(self.radius_of_curvature) > self.smooth_factor:
			self.radius_of_curvature.pop(0)
		curverad = np.mean(self.radius_of_curvature)


		camera_center = (left_fitx[-1] + right_fitx[-1])/2
		center_diff = (camera_center - image_size[0]/2)*xm_per_pix
		side_pos = 'left'
		if center_diff <= 0:
			side_pos = 'right'

		#cv2.putText(result, 'Radius of Curvature = '+str(round(curverad,3))+'(m)',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
		#cv2.putText(result, 'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

		cv2.putText(pers_img, 'Perspective transform',(550,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
		cv2.putText(out_img, 'Color filter and polyfit',(550,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)


		text1 = 'Radius of Curvature = '+str(round(curverad,3))+'(m)'
		text2 = 'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center'

		out = self.prepare_output_blend(text1, text2, pers_img, out_img, result)
		return out

		#return result, out_img, left_fit, right_fit, binary, pers_img
		#out = np.vstack((np.hstack((result, binary)), np.hstack((pers_img, out_img))))

		#return out 
		
if __name__ == '__main__':

	with  open('./params/pers_mtx.p','rb') as f:
		pers_mtx = pickle.load(f)
		M = pers_mtx['M']
		Minv = pers_mtx['Minv']

	with  open('./params/cam_cali_pickle.p','rb') as f:
		dic_cam_cal = pickle.load(f)
		cal_mtx = dic_cam_cal['mtx']
		cal_dist = dic_cam_cal['dist']

	Mynwindows = 10
	Mymargin = 50
	Myroi_margin = 75
	Myminpix = 200
	My_sanity_thresh = 100


	#tracker = lane_tracker(Mynwindows, Mymargin, Myminpix, cal_mtx, cal_dist, M, Minv, My_sanity_thresh,  My_ym = 25/720, My_xm = 9.84/1280, Mysmooth_factor = 10)

	def process_video(img):
	 	result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	 	return tracker.process_image(result)

	#Output_video = 'project_video_out_ver4.mp4'
	#Input_video = 'project_video.mp4'

	#Output_video = 'challenge_video_out_ver10.mp4'
	#Input_video = 'challenge_video.mp4'

	#Output_video = 'harder_challenge_video_out_ver3.mp4'
	#Input_video = 'harder_challenge_video.mp4'
	
	inputs = ['project_video.mp4', 'challenge_video.mp4']
	outputs = ['project_video_result.mp4', 'challenge_video_result.mp4']

	#inputs = ['challenge_video.mp4']
	#outputs = ['challenge_video_result.mp4']

	for i in range(len(inputs)):
		tracker = lane_tracker(Mynwindows, Mymargin, Myminpix, cal_mtx, cal_dist, M, Minv, My_sanity_thresh,  My_ym = 25/720, My_xm = 9.84/1280, Mysmooth_factor = 10)
		clip1 = VideoFileClip(inputs[i])
		#clip1 = VideoFileClip(inputs[i]).subclip(10,20)
		video_clip = clip1.fl_image(tracker.process_image)
		video_clip.write_videofile(outputs[i], audio=False)

	# clip1 = VideoFileClip(Input_video)
	# i = 0
	# for frame in clip1.iter_frames():
	# 	i+=1
	# 	#print('frame ',i)
	# 	name = './bbbb4/' + str(i) + '.jpg'
	# 	frame = tracker.process_image(frame)
	# 	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# 	cv2.imwrite(name, frame)
	    
