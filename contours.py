import cv2

img = cv2.imread("datasets/dog.jfif")
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, bw_otsu_img = cv2.threshold(grey_img, 0, 255, cv2.THRESH_OTSU)
ret, bw_bin_img = cv2.threshold(grey_img, 150, 255, cv2.THRESH_BINARY)


contour_img = img.copy()
contours, hierarchy = cv2.findContours(bw_otsu_img, method = cv2.CHAIN_APPROX_NONE, mode = cv2.RETR_TREE)
cv2.drawContours(contour_img, contours = contours, contourIdx = -1, color = (0, 255, 0), thickness = 2, lineType = cv2.LINE_AA)


cv2.imshow("Original: ", img)
cv2.imshow("Contours: ", contour_img)
"""cv2.imshow("GreyScale: ", grey_img)
cv2.imshow("bw_otsu_img: ", bw_otsu_img)
cv2.imshow("bw_bin_img: ", bw_bin_img)"""


cv2.waitKey(0)
cv2.destroyAllWindows()