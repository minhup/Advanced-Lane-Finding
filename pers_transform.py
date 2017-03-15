import numpy as np
import cv2
import glob
import os
import pickle

with  open('./params/cam_cali_pickle.p','rb') as f:
    dic_cam_cal = pickle.load(f)
    mtx = dic_cam_cal['mtx']
    dist = dic_cam_cal['dist']

# Undistort image given camera calibartion parameters mtx and dist
def undistort(img, mtx,dist):
    undist_img = cv2.undistort(img, mtx, dist, None , mtx)
    return undist_img


def get_vanishing_point(straight_lines_image_files):
    
    sl_imgs = glob.glob(straight_lines_image_files)
    print(sl_imgs)
    
    term1 = np.zeros((1,2))
    term2 = np.zeros((2,2))

    for fname in sl_imgs:
        sl_img = undistort(cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB))

        preprocessImage = np.zeros_like(sl_img[:,:,0])
        gradx = abs_sobel_thresh(sl_img, orient = 'x', thresh=(50, 255))
        grady = abs_sobel_thresh(sl_img, orient = 'y', thresh = (25, 255))
        c_binary = color_threshold(sl_img, s_thresh=(100, 255), v_thresh=(50,255))
        preprocessImage[((gradx == 1) & (grady == 1) | (c_binary == 1) )] =  255

        vertices = np.array([[ [600,440],[710,440],[1150,720],[180,720] ]])
        roi = np.zeros_like(preprocessImage)
        cv2.fillPoly(roi, vertices, 1)

        roi = roi*preprocessImage
        canny = cv2.Canny(roi, 100, 200)

        rho = 1 
        theta = np.pi/180
        threshold = 50
        min_line_len = 50
        max_line_gap = 100

        lines = cv2.HoughLinesP(canny, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        for line in lines:
            for x1, y1, x2, y2 in line:
                n = np.array([[y2 - y1, x1 - x2]])
                n = n/np.linalg.norm(n)
                n = np.dot(n.T,n)

                term1 += np.dot(np.array([x1,y1]),n)
                term2 += n

    vanish_point = np.dot(term1,np.linalg.inv(term2))
    
          
    return (vanish_point[0][0], vanish_point[0][1])

def pers_matrix(t=60, b=35, width=450):
    
    vanishing_point = pickle.load(open('./test_images/vanish_point.p','rb'))
    
    top = vanishing_point[1] + t
    bottom = 720 - b
    
    #define source and destination targets
    p1 = [vanishing_point[0] - width/2, top]
    p2 = [vanishing_point[0] + width/2, top]
    
    p3_p4 = width*(720 - vanishing_point[1] - b)/t
    
    p3 = [vanishing_point[0] + p3_p4/2,bottom]
    p4 = [vanishing_point[0] - p3_p4/2,bottom]
    
    q1 = [0,0]
    q2 = [img_size[0],0]
    q3 = [img_size[0], img_size[1]]
    q4 = [0, img_size[1]]
    

    src = np.float32([p1,p2 ,p3,p4])

    dst = np.float32([q1,q2,q3,q4])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv


if __name__=='__main__':

    straight_lines_image_files = './test_images/straight_lines*.jpg'
    img_size = (1280, 720) 

    vanishing_point_file = './test_images/vanish_point.p'
    if not os.path.exists(vanishing_point_file):
        vanish_point = get_vanishing_point(vanishing_point_file)
        pickle.dump(vanish_point, open(vanishing_point_file, 'wb'))

    pers_mtx_file = './test_images/pers_mtx.p'
    if not os.path.exists(pers_mtx_file):
        M, Minv = pers_matrix()
        pers_mtx = {'M':M, 'Minv':Minv}
        pickle.dump(pers_mtx, open(pers_mtx_file, 'wb'))


