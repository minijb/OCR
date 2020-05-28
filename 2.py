import numpy as np
import cv2
import argparse


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def four_point_transform(image, pts):
	# 获取输入坐标点
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 变换后对应坐标位置
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# 计算变换矩阵(先将二维转换为三维，再将三维变换为二维)
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回变换后结果
	return warped


image = cv2.imread("5.jpg")
# 坐标要随着热死则的变化而变化
ratio = image.shape[0] / 500.0  # 【0】是height

orig = image.copy()

image = resize(orig, height=500)
# cv_show("image",image)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转化灰度图
cv_show('gray',gray)
gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯滤波去除躁点
cv_show('gray',gray)
edged = cv2.Canny(gray, 75, 200)  # 边缘检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]  # 轮廓检测
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]  # 通过面积排序，找到面积最大的前5个
cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
cv_show('image',image)
# 遍历轮廓
for c in cnts:
    # 计算轮廓近似
    peri = cv2.arcLength(c, True)  # arcLength是计算周长的函数'
    # C表示输入的点集
    # epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
    # True表示封闭的
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 多边拟合函数,主要功能是把一个连续光滑曲线折线化，对图像轮廓点进行多边形拟合
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 2)
cv_show('image',image)
# 透视变换:将歪斜的东西变正，通过2组坐标可以实现
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
# orig，原始图像，screenCnt原始的点阵（4），新的四个坐标  注意原始图像已经变换，需要乘上比例
warped=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
ref=cv2.threshold(warped,100,255,cv2.THRESH_BINARY)[1]
cv2.imwrite('scan.jpg', ref)
cv2.imshow('orig',resize(orig,height=650))
cv2.imshow('ref',resize(ref,height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()
