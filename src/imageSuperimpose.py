import numpy as np
import cv2
import copy
import argparse

# To detect the tag ID
def tagDetect(image):

# thresholding the image
    ret,img_bw = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
    corner_pixel = 255
    cropped_img = img_bw[50:150,50:150]

    (h, w) = cropped_img.shape

    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    found = False

# cropping the image to needed segments
    block_1 = cropped_img[37,37]
    block_3 = cropped_img[62,37]
    block_2 = cropped_img[37,62]
    block_4 = cropped_img[62,62]
    white = 255

# interpreting the data blocks
    if block_3 == white:
        block_3 = 1
    else:
        block_3 = 0
    if block_4 == white:
        block_4 = 1
    else:
        block_4 = 0
    if block_2 == white:
        block_2 = 1
    else:
        block_2 = 0
    if block_1 == white:
        block_1 = 1
    else:
        block_1 = 0

# interpreting the rotation block
    if (cropped_img[85,85]==corner_pixel):
        return list([block_3,block_4,block_2,block_1]),"BR"
    elif (cropped_img[15,85]==corner_pixel):
        return list([block_4,block_2,block_1,block_3]),"TR"
    elif (cropped_img[15,15]==corner_pixel):
        return list([block_2,block_1,block_3,block_4]),"TL"
    elif (cropped_img[85,15]==corner_pixel):
        return list([block_1,block_3,block_4,block_2]),"BL"

    return None,None

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

	return rect

# Calculates the homography matrix
def calculateHomography(p, p1):
    A = []
    p2 = correctCoordinates(p)

    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    l = Vh[-1,:] / Vh[-1,-1]
    h = np.reshape(l,(3,3))

    return h
#Generates the corner points for all the interested contours
def generateContours(frame):
    test_img1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    test_blur = cv2.GaussianBlur(test_img1,(5,5),0)
    edge = cv2.Canny(test_blur, 75,200)
    edge1 = copy.copy(edge)
    countour_list = list()
    # Finds the contours as a tree structure

    r, cnts, h = cv2.findContours(edge1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    index = list()
    for hier in h[0]:
        if hier[3] != -1:
            index.append(hier[3])

    for c in index:

# for making the contour a proper shape
        peri = cv2.arcLength(cnts[c], True)
        approx = cv2.approxPolyDP(cnts[c], 0.02 * peri, True)

        if len(approx) > 4:
            peri1 = cv2.arcLength(cnts[c-1], True)
            corners = cv2.approxPolyDP(cnts[c-1], 0.02*peri1, True)
            countour_list.append(corners)

    new_contour_list = list()

    for contour in countour_list:
        if len(contour) == 4:
            new_contour_list.append(contour)
    final_contour_list = list()
#Capturing all the corners of the interested contours

    for element in new_contour_list:
        if cv2.contourArea(element) < 2500:
            final_contour_list.append(element)

    return(final_contour_list)
# To calculate the new projection matrix to transform from 3D to camera frame

# To orient the template image accoriding to the tag information
def correctOrient(location, maxDim):


    if(location=="BR"):
        p1 = np.array([
		[0, 0],
		[maxDim - 1, 0],
		[maxDim - 1, maxDim - 1],
		[0, maxDim - 1]], dtype = "float32")
        return (p1)
    elif(location=="TR"):
        p1 = np.array([
		[maxDim - 1, 0],
		[maxDim - 1, maxDim - 1],
		[0, maxDim - 1],
		[0, 0]], dtype = "float32")
        return (p1)
    elif(location=="TL"):
        p1 = np.array([
		[maxDim - 1, maxDim - 1],
		[0, maxDim - 1],
		[0, 0],
		[maxDim - 1, 0]], dtype = "float32")
        return (p1)

    elif(location=="BL"):
        p1 = np.array([
		[0, maxDim - 1],
		[0, 0],
		[maxDim - 1,0],
		[maxDim - 1, maxDim - 1]], dtype = "float32")
        return (p1)

# To make the homography transformation
def homography(frame, p1):
    final_contour_list = generateContours(frame)
    lena_list = list()

#detecting all the contours in the image
    for i in range(len(final_contour_list)):
        cv2.drawContours(frame, [final_contour_list[i]], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", frame)
#creating homography matrix for all the contours
        c_rez = final_contour_list[i][:, 0]
        H_matrix = calculateHomography(p1, correctCoordinates(c_rez))

        tag = cv2.warpPerspective(frame, H_matrix, (200, 200))

        cv2.imshow("Outline", frame)
        cv2.imshow("Tag after Homo",tag)
#masking lena over the orignal tag
        tag1 = cv2.cvtColor(tag,cv2.COLOR_BGR2GRAY)
        decoded,location =tagDetect(tag1)
        empty = np.full(frame.shape,0,dtype = 'uint8')
        if not location == None:
            p2 = correctOrient(location, 200)
            print(location)
            if not decoded == None:
                print(decoded)
            H_Lena = calculateHomography(correctCoordinates(c_rez), p2)
            lena_overlap = cv2.warpPerspective(lena_resized, H_Lena, (frame.shape[1], frame.shape[0]))
            print("Lena Overlap",lena_overlap )
            if not np.array_equal(lena_overlap,empty ):
                lena_list.append(lena_overlap.copy())

    mask = np.full(frame.shape,0,dtype = 'uint8')
    if lena_list != []:
        for lena in lena_list:
            print("mask",mask)
            print("lena",lena)
            temp = cv2.add(mask,lena.copy())
            mask = temp

        lena_gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        r,lena_bin = cv2.threshold(lena_gray,10,255,cv2.THRESH_BINARY)

        mask_inv = cv2.bitwise_not(lena_bin)

        mask_3d = frame.copy()
        mask_3d[:,:,0] = mask_inv
        mask_3d[:,:,1] = mask_inv
        mask_3d[:,:,2] = mask_inv
        img_masked = cv2.bitwise_and(frame,mask_3d)
        final_image = cv2.add(img_masked,mask)
        cv2.imshow("Lena",final_image)
        #cv2.waitKey(0)

    if cv2.waitKey(1) & 0xff == 27:
        cv2.destroyAllWindows()


# To take the user input
opt = int(input("enter one from the option below \nenter 0 for Tag0 Video\nenter 1 for Tag1 Video\nenter 2 for Tag2 Video\nenter 3 for multipleTags Video : \t"))


if opt == 0:
	cap = cv2.VideoCapture('Tag0.mp4')
elif opt == 1:
	cap = cv2.VideoCapture('Tag1.mp4')
elif opt == 2:
	cap = cv2.VideoCapture('Tag2.mp4')
elif opt == 3:
	cap = cv2.VideoCapture('multipleTags.mp4')

lena_img = cv2.imread('Lena.png')
lena_resized = cv2.resize(lena_img,(200,200))


success = True
fps = cap.get(cv2.CAP_PROP_FPS)
fps = int(fps)
print("fps", fps)
count = 0

dim = 200
p1 = np.array([
		[0, 0],
		[dim - 1, 0],
		[dim - 1, dim - 1],
		[0, dim - 1]], dtype = "float32")

# Iterating through all the frames in the video
while (cap.isOpened()):
    success, frame = cap.read()
    if success == False:
        break
    img = cv2.resize(frame,(0,0), fx = 0.5, fy = 0.5)
    homography(img, p1)

cap.release()

