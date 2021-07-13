import cv2 as cv
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from drawMatchPointsLine import *


def fix_flags():
    flags = None
    flags = cv.CALIB_FIX_PRINCIPAL_POINT
    # flags = flags | cv.CALIB_FIX_FOCAL_LENGTH
    # flags = flags | cv.CALIB_FIX_K1
    # flags = flags | cv.CALIB_FIX_K2
    # flags = flags | cv.CALIB_FIX_K3
    # flags = flags | cv.CALIB_FIX_TANGENT_DIST
    return flags


def Stero_Calibrate_and_Save(letfImgroot, rightImgroot, Lsuffix, Rsuffix):
    board_size, scale = [11, 8], 20
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 30, 0.001)
    # 获取标定板角点的位置
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:(board_size[0] - 1) * scale:complex(0, board_size[0]),
                  0:(board_size[1] - 1) * scale:complex(0, board_size[1])].T.reshape(-1, 2)
    obj_points = []  # 存储3D点
    img_points_l = []  # 存储左侧相机2D点
    img_points_r = []  # 存储右侧相机2D点

    images_l = glob.glob(letfImgroot + "/*." + Lsuffix)
    images_r = glob.glob(rightImgroot + "/*." + Rsuffix)
    sizel, sizer = (), ()

    for index in range(len(images_l)):
        imgl = cv.imread(images_l[index])
        grayl = cv.cvtColor(imgl, cv.COLOR_BGR2GRAY)
        sizel = grayl.shape[::-1]
        imgr = cv.imread(images_r[index])
        grayr = cv.cvtColor(imgr, cv.COLOR_BGR2GRAY)
        sizer = grayr.shape[::-1]
        retl, cornersl = cv.findChessboardCorners(grayl, (board_size[0], board_size[1]))
        retr, cornersr = cv.findChessboardCorners(grayr, (board_size[0], board_size[1]))
        if retl and retr:
            obj_points.append(objp)
            corners2 = cv.cornerSubPix(grayl, cornersl, (3, 3), (-1, -1), criteria).reshape(-1, 2)  # 在原角点的基础上寻找亚像素角点
            # print(type(corners2))
            if corners2.any:
                img_points_l.append(corners2 / 1.0)
            else:
                img_points_l.append(cornersl / 1.0)
            cv.drawChessboardCorners(imgl, (board_size[0], board_size[1]), cornersl, retl)  # 记住，OpenCV的绘制函数一般无返回值

            corners2 = cv.cornerSubPix(grayr, cornersr, (3, 3), (-1, -1), criteria).reshape(-1, 2)  # 在原角点的基础上寻找亚像素角点
            if corners2.any:
                img_points_r.append(corners2 / 1.0)
            else:
                img_points_r.append(cornersr / 1.0)
            cv.drawChessboardCorners(imgr, (board_size[0], board_size[1]), cornersr, retr)  # 记住，OpenCV的绘制函数一般无返回值
            img_merge = np.hstack((imgl, imgr))
            size1 = img_merge.shape[::-1]
            img_merge = cv.resize(img_merge, (int(size1[1] / 2), int(size1[2] / 2)))
            # cv.imshow(images_l[index] + images_r[index], img_merge)
            # cv.waitKey(0)
    # 左侧相机内参标定
    rvecsl, mtxl, distl, rvecsl, tvecsl = cv.calibrateCamera(obj_points, img_points_l, sizel, None, None)
    print("左相机反向投影误差ret:", rvecsl)
    print("mtxl:\n", mtxl)  # 内参数矩阵
    print("distl:\n", distl)  # 畸变系数 distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    # print("rvecsl:\n", rvecsl) # 旋转向量 # 外参数
    # print("tvecsl:\n", tvecsl) # 平移向量 # 外参数
    print("------------------------------------------")

    # 右侧相机内参标定
    retr, mtxr, distr, rvecsr, tvecsr = cv.calibrateCamera(obj_points, img_points_r, sizer, None, None)
    print("右相机反向投影误差ret1:", retr)
    print("mtxr:\n", mtxr)  # 内参数矩阵
    print("distr:\n", distr)  # 畸变系数 distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    # print("rvecsr:\n", rvecsr) # 旋转向量 # 外参数
    # print("tvecsr:\n", tvecsr) # 平移向量 # 外参数
    print("------------------------------------------")

    # 双目立体矫正及左右相机内参进一步修正
    rms, C1, distr, C2, dist2, R, T, E, F = cv.stereoCalibrate(obj_points, img_points_l, img_points_r, mtxl, distl, mtxr,
                                                               distr, sizel,
                                                               flags=fix_flags())

    print("双目成像重投影误差rms:", rms)
    print("左相机内参数C1:\n", C1)  # 左相机内参数
    print("左相机畸变系数dist1:\n", distr)  # # 左相机畸变系数
    print("右相机内参数C2:\n", C2)  # 右相机内参数
    print("右相机畸变系数dist2:\n", dist2)  # # 右相机畸变系数
    print("右相机相对于左相机的旋转矩阵R:\n", R)  # 旋转矩阵 # 外参数
    print("右相机相对于左相机的平移矩阵T:\n", T)  # 平移向量 # 外参数

    fs = cv.FileStorage("./CameraParameter/cameraPara_limg_rimg.yml", cv.FILE_STORAGE_WRITE)
    fs.write('board_width', board_size[0])
    fs.write('board_height', board_size[1])
    fs.write('square_size', scale)
    fs.write('camera_matrixL', C1)
    fs.write('dist_coefsL', distr)
    fs.write('camera_matrixR', C2)
    fs.write('dist_coefsR', dist2)
    fs.write('R', R)
    fs.write('T', T)
    fs.write('sizel', np.array([sizel[0], sizel[1]]))
    fs.release()
    '''自己计算的双目之间的RT'''
    R, T = 0, 0
    for i in range(len(rvecsl)):
        R1, _ = cv.Rodrigues(rvecsl[i])
        R2, _ = cv.Rodrigues(rvecsr[i])
        R += (R2 @ np.linalg.inv(R1))
        T += tvecsr[i] - (R2 @ np.linalg.inv(R1)) @ tvecsl[i]
    R, T = R / len(rvecsl), T / len(tvecsr)
    print('自己计算的R\n:', R)
    print('自己计算的T\n:', T)


def GetstereoPara_and_DepthEstimate(leftImgroot, rightImgroot, Lsuffix, Rsuffix):
    def callbackFunc(e, x, y, f, p):
        if e == cv.EVENT_LBUTTONDOWN:
            print(threeD[y][x])

    fs = cv.FileStorage(cv.samples.findFile(r'./CameraParameter/cameraPara_C0_C1.yml'), cv.FILE_STORAGE_READ)
    board_width = int(fs.getNode('board_width').real())
    board_height = int(fs.getNode('board_height').real())
    square_size = fs.getNode('square_size').real()
    camera_matrixL = fs.getNode('camera_matrixL').mat()  # C1
    # print(camera_matrixL)
    dist_coefsL = fs.getNode('dist_coefsL').mat()
    camera_matrixR = fs.getNode('camera_matrixR').mat()
    dist_coefsR = fs.getNode('dist_coefsR').mat()
    R = fs.getNode('R').mat()
    T = fs.getNode('T').mat()
    temp = tuple(fs.getNode('sizel').mat().reshape(1, -1).tolist()[0])
    sizel = (int(temp[0]), int(temp[1]))
    fs.release()
    # print((sizel))

    C1, distr, C2, dist2 = camera_matrixL, dist_coefsL, camera_matrixR, dist_coefsR
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(C1, distr, C2, dist2, sizel, R, T, -1, (0, 0))
    left_map1, left_map2 = cv.initUndistortRectifyMap(C1, distr, R1, P1, sizel, cv.CV_16SC2)
    right_map1, right_map2 = cv.initUndistortRectifyMap(C2, dist2, R2, P2, sizel, cv.CV_16SC2)

    images_l = glob.glob(leftImgroot + "/*." + Lsuffix)
    images_r = glob.glob(rightImgroot + "/*." + Rsuffix)
    for index in range(min(len(images_l), int(len(images_r)))):
        frame1 = hisEqulColor(cv.imread(images_l[index]))
        frame2 = hisEqulColor(cv.imread(images_r[index]))
        # 在图中画出相匹配的方框, cv画图无返回值.
        cv.rectangle(frame1, (validPixROI1[0], validPixROI1[1]),
                     (validPixROI1[0] + validPixROI1[2], validPixROI1[1] + validPixROI1[3]), (0, 255, 0), 2)
        cv.rectangle(frame2, (validPixROI2[0], validPixROI2[1]),
                     (validPixROI2[0] + validPixROI2[2], validPixROI2[1] + validPixROI2[3]), (0, 255, 0), 2)

        Limg_rectified = cv.remap(frame1, left_map1, left_map2, cv.INTER_LINEAR)
        Rimg_rectified = cv.remap(frame2, right_map1, right_map2, cv.INTER_LINEAR)
        imgL = cv.cvtColor(Limg_rectified, cv.COLOR_BGR2GRAY)
        imgR = cv.cvtColor(Rimg_rectified, cv.COLOR_BGR2GRAY)
        GetPolePoints(imgL, imgR)

        # 深度图获取
        stereo = cv.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=3, P1=256, P2=864,
                                      speckleWindowSize=100, speckleRange=100, disp12MaxDiff=128, uniquenessRatio=2)
        disparity = stereo.compute(imgL, imgR)
        threeD = cv.reprojectImageTo3D(disparity.astype(np.float32) / 16., Q)  # 此三维坐标点的基准坐标系为左侧相机坐标系
        disp = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        plt.imshow(disp, cmap='gray'), plt.ion(), plt.pause(2), plt.close()

    cv.destroyAllWindows()


def StereoCalibration_function():
    import os

    # leftImgroot, rightImgroot = './C1_P0', './C0_P0'
    # leftImgroot, rightImgroot = './limg', './rimg'
    # Lsuffix, Rsuffix = os.listdir(leftImgroot)[0].split('.')[1], os.listdir(rightImgroot)[0].split('.')[1]
    # Stero_Calibrate_and_Save(leftImgroot, rightImgroot, Lsuffix, Rsuffix)

    leftImgroot, rightImgroot = './doorL', './doorR'
    Lsuffix, Rsuffix = os.listdir(leftImgroot)[0].split('.')[1], os.listdir(rightImgroot)[0].split('.')[1]
    GetstereoPara_and_DepthEstimate(leftImgroot, rightImgroot, Lsuffix, Rsuffix)


if __name__ == '__main__':
    StereoCalibration_function()
