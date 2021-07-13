import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from copy import deepcopy


def hisEqulColor(img):
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    channels = cv.split(ycrcb)
    cv.equalizeHist(channels[0], channels[0])
    cv.merge(channels, ycrcb)
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, img)
    return img


def GetPolePoints(queryImg, trainImg):  # qImg*M = tImg, 会将qImg转到tImg其所应在位置
    MIN_MATCH_COUNT = 4  # 设置最低匹配数量为4
    if len(queryImg.shape) == 3:
        queryImg = cv.cvtColor(queryImg, cv.COLOR_BGR2GRAY)
        trainImg = cv.cvtColor(trainImg, cv.COLOR_BGR2GRAY)

    # sift=cv.xfeatures2d.SIFT_create() #创建sift检测器
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(queryImg, None)
    # 返回值分别是一个Keypoint对象的列表和一个数组,这个数组包含的信息起到特征描述作用,即用从另一种角度更简洁描述图像
    kp2, des2 = sift.detectAndCompute(trainImg, None)
    # for i in kp1: #Keypoint对象属性有:pt--像素坐标系坐标, size--邻域搜索半径, angle--角度, response--响应强度
    #     print(i.pt)
    # 创建设置FLAAN匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    mathces = flann.knnMatch(des1, des2, k=2)

    # flann.knnMatch()返回两个最相近匹配点,格式为List[ list1[Dmatch_Object1 , Dmatch_Object2], list2[], ... ]
    # for index,matches_Listobject in enumerate(mathces): #enumerate()用以遍历对象,返回其索引和值.
    #     print(matches_Listobject[0].queryIdx) #Dmatch类型对象只有queryIdx，trainIdx，distance三个属性
    # queryIdx和trainIdx都是测试图像中的特征点的索引值,并非坐标值.分别代表的是具有同一特征的点
    # 在样本图像queryimg的特征点列表kp1的索引值,以及在测试图像trainimg的特征点列表kp2中的索引值
    # distance代表着测试图像中特征点的距离, 这个距离相对谁而言? 当然都是相对于样本图像特征点而言
    # 使用sift算法匹配出来的kp2即样本特征点的列表,所以有了索引号就能定位坐标以及距离
    # for mylist in mathces:
    #     print("queryIdx: ", mylist[0].queryIdx, "; queryIdx_distance: ", mylist[0].distance)

    good_feapoint = []  # 得到所有符合条件的最邻近特征点的索引值
    # 过滤不合格的匹配结果，大于0.4的都舍弃
    for m, n in mathces:  # Dmatch_Object1代表着最邻近,Dmatch_Object2代表次邻近,所以m是最邻近特征点,n是次邻近
        if m.distance < 0.4 * n.distance:
            good_feapoint.append(m)
            # print("m.queryIdx : ", m.queryIdx, "m.trainIdx : ", m.trainIdx)
            # print("left: ", kp1[1314].pt), print("right", kp2[305].pt)
    # 如果匹配结果大于10，则获取关键点的坐标，用于计算变换矩阵和掩膜
    if len(good_feapoint) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_feapoint]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_feapoint]).reshape(-1, 1, 2)
        # print(len(src_pts)), print(len(good_feapoint))
        # 计算变换矩阵和掩膜
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransacReprojThreshold=1.0, confidence=0.99)
        matchesMask = mask.ravel().tolist()  # mask是ndarray类型, ravel()将n维转成1维
        # 根据变换矩阵进行计算，找到小图像在大图像中的位置
        h, w = queryImg.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv.perspectiveTransform(pts, H)
        # cv.polylines(trainImg, [np.int32(dst)], True, 0, 5, cv.LINE_AA)
    else:
        print(" Not Enough matches are found")
        matchesMask = None

    # 画出特征匹配线
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
    # plt展示最终的结果
    img3 = cv.drawMatches(queryImg, kp1, trainImg, kp2, good_feapoint, None, **draw_params)
    plt.imshow(img3), plt.ion(), plt.pause(3), plt.close()
    return img3
