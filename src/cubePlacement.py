import cv2
import numpy as np
import imutils
import copy
from numpy.linalg import inv
from numpy.linalg import norm

# This function is to order the coordinates properly
def correctCoordinates(pts):
    # to correct the order of interpreting the coordinates

	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	diff = np.diff(pts, axis = 1)

	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

# This function is to draw the cube on the tag
def drawCube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    print("imgpts")
    print(imgpts)

    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,255),3)                      #floor

    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255,255,0),3)       #edges

    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)                        #top

    return img

# Calculates the homography matrix
def calculateHomography(p1, p2):
    a = []
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        a.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        a.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    a = np.asarray(a)
    u, s, vh = np.linalg.svd(a)
    l = vh[-1, :] / vh[-1, -1]
    h = np.reshape(l, (3, 3))
    return h

# To calculate the new projection matrix to transform from 3D to camera frame
def projectionMatrix(h):
    K=np.array([[1406.08415449821,0,0],[2.20679787308599, 1417.99930662800,0],[1014.13643417416, 566.347754321696,1]]).T
    h = inv(h)
    b_new = np.dot(inv(K),h)
    b1 = b_new[:,0].reshape(3,1)
    b2 = b_new[:,1].reshape(3,1)
    r3 = np.cross(b_new[:,0],b_new[:,1])

    b3 = b_new[:,2].reshape(3,1)
    lambda_val = 2/(norm((inv(K)).dot(b1))+norm((inv(K)).dot(b2)))
    r1 = lambda_val*b1
    r2 = lambda_val*b2
    r3 = (r3 * lambda_val * lambda_val).reshape(3, 1)
    t = lambda_val*b3
    r = np.concatenate((r1,r2,r3),axis=1)

    return r,t,K

# To make the homography transformation
def homography(image, pts):
    #correcting the order of the corner points that are obtained
    rect = correctCoordinates(pts)
    max_dim = 170

    dst = np.array([
        [0, 0],
        [max_dim - 1, 0],
        [max_dim - 1, max_dim - 1],
        [0, max_dim - 1]], dtype="float32")

    # obtaining the Homography matrix

    m1 = calculateHomography(rect, dst)
    r,t,K = projectionMatrix(m1)
    m2 = calculateHomography(dst, rect)
    warped_img = cv2.warpPerspective(image, m1, (max_dim, max_dim))

    return warped_img,r,t,K,rect



# Taking inputs from the user
opt = int(input("enter one from the option below \nenter 0 for Tag0 Video\nenter 1 for Tag1 Video\nenter 2 for Tag2 Video : \t"))

if opt == 0:
	cap = cv2.VideoCapture('Tag0.mp4')
elif opt == 1:
	cap = cv2.VideoCapture('Tag1.mp4')
elif opt == 2:
	cap = cv2.VideoCapture('Tag2.mp4')


# Iterating through all the frames in the video

while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # smooth
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[2:5]
    axis = np.float32([[0,0,0], [0,160,0], [160,160,0], [160,0,0],[0,0,-160],[0,160,-160],[160,160,-160],[160,0,-160] ])


# iterating through the contours
    for c in cnts:

# for making the contour a proper shape
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)


        if len(approx) == 4:
            corners = approx
            print(corners)
            break
        elif len(approx) != 4:
            corners = None

    tag_count = 0


    if corners is not None:
        tag_count =  tag_count + 1
        cv2.drawContours(frame, [corners], -1, (0, 255, 0), 1)

        cv2.imshow("Outline", frame)

        warped,r,t,K,rect = homography(frame, corners[:, 0])
        cv2.imshow("Warped", warped)

        imgpts,jac = cv2.projectPoints(axis,r,t,K,np.zeros((1,4)))
        img = drawCube(frame, imgpts)
        cv2.imshow('Points',img)

    if cv2.waitKey(1) & 0xff == 27:
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()






