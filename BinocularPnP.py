import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from glob import glob
from Optimized_byScipy import *
from LMoptimize import myLMoptimized
from scipy import linalg
import time


def points3D_transform(points1, points2):  # points3D_transform(Objpoints, P3d)
    '''points1 m*3 points2 n*3  m=n
        pattern_points棋盘格点,世界坐标系.
        P3d 交汇点,像机坐标系.
    '''

    center_points1 = np.mean(points1, 0)
    center_points2 = np.mean(points2, 0)
    stdpoint1, stdpoint2 = np.std(points1), np.std(points2)
    new_points1 = np.sqrt(2) * (points1 - center_points1) / stdpoint1
    new_points2 = np.sqrt(2) * (points2 - center_points2) / stdpoint2
    M = new_points2.T @ new_points1
    u, s, vt = np.linalg.svd(M)
    R = u @ vt
    if np.linalg.det(R) < 0:
        R = -R
    T = center_points2.T - R @ center_points1
    return R, T


def get_RMS(point1, point2):
    tot_error = 0
    for i in range(point1.shape[0]):
        tot_error += np.sum(np.abs(point1[i, :] - point2[i, :]) ** 2)
    mean_error_l = np.sqrt(tot_error / point1.shape[0])
    return mean_error_l


def getPara(io):
    def getOptimizedRT(io):
        fs = cv.FileStorage(cv.samples.findFile(io), cv.FILE_STORAGE_READ)
        R = fs.getNode('R').mat()
        T = fs.getNode('T').mat()
        fs.release()
        return R, T

    fs = cv.FileStorage(cv.samples.findFile(io), cv.FILE_STORAGE_READ)
    board_width = int(fs.getNode('board_width').real())
    board_height = int(fs.getNode('board_height').real())
    square_size = fs.getNode('square_size').real()
    camera_matrixL = fs.getNode('camera_matrixL').mat()
    print(camera_matrixL)
    dist_coefsL = fs.getNode('dist_coefsL').mat()
    camera_matrixR = fs.getNode('camera_matrixR').mat()
    dist_coefsR = fs.getNode('dist_coefsR').mat()
    R = fs.getNode('R').mat()
    T = fs.getNode('T').mat()
    fs.release()
    return board_width, board_height, square_size, camera_matrixL, dist_coefsL, camera_matrixR, dist_coefsR, R, T


def getImgpoints(images_ls,images_rs,camera_matrixL,camera_matrixR,R,T):
    criteria = (cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 30, 0.01)
    # left
    imgl = cv.imread(images_ls[index])
    grayl = cv.cvtColor(imgl, cv.COLOR_BGR2GRAY)
    size = grayl.shape[::-1]
    # print(size)
    imgr = cv.imread(images_rs[index])
    grayr = cv.cvtColor(imgr, cv.COLOR_BGR2GRAY)
    size = grayr.shape[::-1]
    retl, cornersl = cv.findChessboardCorners(grayl, (pattern_size[0], pattern_size[1]), None)
    retr, cornersr = cv.findChessboardCorners(grayr, (pattern_size[0], pattern_size[1]), None)
    # print(cornersl)

    if retl and retr:
        subcornersl = cv.cornerSubPix(grayl, cornersl, (3, 3), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        cv.drawChessboardCorners(imgl, pattern_size, subcornersl, retl)
        # 去畸变
        corners_undis_l = cv.undistortPoints(subcornersl, camera_matrixL, dist_coefsL)
        corners_undis_l = corners_undis_l.reshape(corners_undis_l.shape[0], 2)
        # print(corners_undis_l)

        subcornersr = cv.cornerSubPix(grayr, cornersr, (3, 3), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        cv.drawChessboardCorners(imgr, pattern_size, subcornersr, retr)
        # 去畸变
        corners_undis_r = cv.undistortPoints(subcornersr, camera_matrixR, dist_coefsR)
        corners_undis_r = corners_undis_r.reshape(corners_undis_r.shape[0], 2)

        img_merge = np.hstack((imgl, imgr))
        size1 = img_merge.shape[::-1]
        img_merge = cv.resize(img_merge, (int(size1[1] / 2), int(size1[2] / 2)))
        # cv.imshow(images_ls[index] + images_rs[index], img_merge)
        # cv.waitKey(1)
        # cv.destroyWindow(images_ls[index] + images_rs[index])

    # 对当前图像进行线线交会 Ax=b, DLT求解
    r0, r1, r2, Tx = R[0, 0], R[0, 1], R[0, 2], T[0, 0]
    r3, r4, r5, Ty = R[1, 0], R[1, 1], R[1, 2], T[1, 0]
    r6, r7, r8, Tz = R[2, 0], R[2, 1], R[2, 2], T[2, 0]

    P3d = []

    # Ax=b形式,归一化像点
    A = np.mat(np.zeros((4, 3)))
    b = np.mat(np.zeros((4, 1)))
    for i in range(corners_undis_l.shape[0]):
        u_left, v_left = corners_undis_l[i, 0], corners_undis_l[i, 1]
        u_right, v_right = corners_undis_r[i, 0], corners_undis_r[i, 1]
        # left
        A[:2, :] = np.array([[1, 0, -u_left], [0, 1, -v_left]])
        b[0, 0], b[1, 0] = 0, 0
        # right
        A[2:, :] = np.array([[u_right * r6 - r0, u_right * r7 - r1, u_right * r8 - r2],
                             [v_right * r6 - r3, v_right * r7 - r4, v_right * r8 - r5]])
        b[2, 0], b[3, 0] = -(u_right * Tz - Tx), -(v_right * Tz - Ty)
        # 3d point in left camera
        P = (A.T * A).I * A.T * b
        P = P.A
        P3d.append(P)
        # print("交汇点:\n", P)
        '''
        # Ax = 0形式, 归一化像点
        A = np.mat(np.zeros((4, 4)))
        for i in range(corners_undis_l.shape[0]):
            u_left, v_left = corners_undis_l[i, 0], corners_undis_l[i, 1]
            u_right, v_right = corners_undis_r[i, 0], corners_undis_r[i, 1]
            # left
            A[:2, :] = np.array([[1, 0, -u_left, 0], [0, 1, -v_left, 0]])
            # right
            A[2:, :] = np.array([[u_right * r6 - r0, u_right * r7 - r1, u_right * r8 - r2, u_right * Tz - Tx],
                                 [v_right * r6 - r3, v_right * r7 - r4, v_right * r8 - r5, v_right * Tz - Ty]])
            # 3d point in left camera
            U, D, VT = linalg.svd(A)
            P = VT[-1, :].reshape(1, 4)
            P = P / P[0, -1]
            P = P[0, :3]  # 是因为需要的是n*3的空间点,也就是非齐次形式才如此处理
            P3d.append(P)
            # print("交汇点:\n", P)
        P3d = np.array(P3d, dtype=float)
        P3d = P3d.reshape(P3d.shape[0], 3)
        '''

        '''   
        # Ax = 0形式, 非归一化像点
        Fxl, Fyl, Cxl, Cyl = camera_matrixL[0, 0], camera_matrixL[1, 1], camera_matrixL[0, 2], camera_matrixL[1, 2]
        Fxr, Fyr, Cxr, Cyr = camera_matrixR[0, 0], camera_matrixR[1, 1], camera_matrixR[0, 2], camera_matrixR[1, 2]
        # subcornersl, subcornersr = np.vstack(subcornersl), np.vstack(subcornersr)
        # subcornersl[:, 0], subcornersl[:, 1] = corners_undis_l[:, 0] * Fxl + Cxl, corners_undis_l[:, 1] * Fyl + Cyl
        # subcornersr[:, 0], subcornersr[:, 1] = corners_undis_r[:, 0] * Fxr + Cxr, corners_undis_r[:, 1] * Fyr + Cyr
        # subcornersl, subcornersr = subcornersl.reshape(subcornersl.shape[0], 2), subcornersr.reshape(subcornersr.shape[0], 2)
        # A = np.mat(np.zeros((4, 4)))
        # for i in range(subcornersl.shape[0]):
        #     u_left, v_left = subcornersl[i, 0], subcornersl[i, 1]
        #     u_right, v_right = subcornersr[i, 0], subcornersr[i, 1]
        #     # left
        #     A[:2, :] = np.array([[Fxl, 0, Cxl - u_left, 0], [0, Fyl, Cyl - v_left, 0]])
        #     # right
        #     A[2:, :] = np.array([[(u_right - Cxr) * r6 - Fxr * r0, (u_right - Cxr) * r7 - Fxr * r1,
        #                           (u_right - Cxr) * r8 - Fxr * r2, (u_right - Cxr) * Tz - Fxr * Tx],
        #                          [(v_right - Cyr) * r6 - Fyr * r3, (v_right - Cyr) * r7 - Fyr * r4,
        #                           (v_right - Cyr) * r8 - Fyr * r5, (v_right - Cyr) * Tz - Fyr * Ty]])
        #     # 3d point in left camera
        #     U, D, VT = linalg.svd(A)
        #     P = VT[-1, :].reshape(1, 4)
        #     P = P / P[0, -1]
        #     P = P[0, :3]  # 是因为需要的是n*3的空间点,也就是非齐次形式才如此处理
        #     P3d.append(P)
        #     # print("交汇点:\n", P)
        # P3d = np.array(P3d, dtype=float)
        # P3d = P3d.reshape(P3d.shape[0], 3)
        '''
    P3d = np.array(P3d, dtype=float)
    P3d = P3d.reshape(P3d.shape[0], 3)
    return P3d, corners_undis_l, corners_undis_r



if __name__ == '__main__':
    time_start = time.time()
    R_w2l_matslist, T_w2l_matslist, R_w2r_matslist, T_w2r_matslist = [], [], [], []  # 用以后面保存每张图平差优化后的位姿结果
    io = r'./CameraParameter/cameraPara_limg_rimg.yml'
    board_width, board_height, square_size, camera_matrixL, dist_coefsL, camera_matrixR, dist_coefsR, R, T = getPara(io)

    from StereoCalibration import StereoCalibration_function

    if camera_matrixL is None:
        StereoCalibration_function()
        board_width, board_height, square_size, camera_matrixL, dist_coefsL, camera_matrixR, dist_coefsR, R, T = getPara(io)
    #  获取角点像素坐标
    images_ls = glob("./limg/*.bmp")
    images_rs = glob("./rimg/*.bmp")
    pattern_size = (11, 8)
    Objpoints = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)  # ((6*9行,3列),数据类型)
    # 第一列填充为0,1,2...8,重复6次 ; 第二列则9个0,9个1,....9个6, 所以第一列存放的是该角点在横向角点中的索引,第二列则是该角点在第几行
    Objpoints[:, :2] = (np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)) * square_size

    h, w = cv.imread(images_ls[0], cv.IMREAD_GRAYSCALE).shape[:2]  # TODO: use imquery call to retrieve results
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    for index in range(0, len(images_ls)):
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Processing {index} Img~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        P3d, corners_undis_l, corners_undis_r = getImgpoints(images_ls, images_rs, camera_matrixL, camera_matrixR, R, T)

        # 绝对定向求解位姿
        R_w2l, T_w2l = points3D_transform(Objpoints, P3d)
        R_w2l = np.matrix(R_w2l)
        T_w2l = np.matrix(T_w2l).T
        corners_undis_l[:, 0] = corners_undis_l[:, 0] * camera_matrixL[0, 0] + camera_matrixL[0, 2]
        corners_undis_l[:, 1] = corners_undis_l[:, 1] * camera_matrixL[1, 1] + camera_matrixL[1, 2]
        corners_undis_l = corners_undis_l.reshape(corners_undis_l.shape[0], 2)
        # print(corners_undis_l)
        corners_undis_r[:, 0] = corners_undis_r[:, 0] * camera_matrixR[0, 0] + camera_matrixR[0, 2]
        corners_undis_r[:, 1] = corners_undis_r[:, 1] * camera_matrixR[1, 1] + camera_matrixR[1, 2]
        corners_undis_r = corners_undis_r.reshape(corners_undis_r.shape[0], 2)

        newcorners_rep_l = cv.projectPoints(Objpoints, R_w2l, T_w2l, camera_matrixL, np.zeros((5, 1)))
        newcorners_rep_l = newcorners_rep_l[0].reshape(newcorners_rep_l[0].shape[0], 2)
        # 计算重投影误差
        mean_error_l = get_RMS(corners_undis_l, newcorners_rep_l)
        # 重投影
        # 左相机坐标系转换到右相机
        R_w2r = R * R_w2l
        T_w2r = R * T_w2l + T
        newcorners_rep_r = cv.projectPoints(Objpoints, R_w2r, T_w2r, camera_matrixR, np.zeros((5, 1)))
        newcorners_rep_r = newcorners_rep_r[0].reshape(newcorners_rep_r[0].shape[0], 2)
        mean_error_r = get_RMS(corners_undis_r, newcorners_rep_r)
        print(f'重投影误差l1:{mean_error_l}')
        print(f'重投影误差r1:{mean_error_r}')
        plt.subplot(221)
        plt.scatter(corners_undis_l[:, 0], corners_undis_l[:, 1], marker='x', color='red', s=40, label='image points')
        #                   记号形状       颜色           点的大小    设置标签
        plt.scatter(newcorners_rep_l[:, 0], newcorners_rep_l[:, 1], marker='+', color='blue', s=40, label='reproject points')

        plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置

        plt.title(r'RMS=' + ("%.3f" % mean_error_l), fontsize=20)

        plt.ylabel('y/pixels')
        plt.grid(True)

        plt.subplot(222)
        plt.scatter(corners_undis_r[:, 0], corners_undis_r[:, 1], marker='x', color='red', s=40, label='image points')
        #                   记号形状       颜色           点的大小    设置标签
        plt.scatter(newcorners_rep_r[:, 0], newcorners_rep_r[:, 1], marker='+', color='blue', s=40, label='reproject points')

        plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置

        plt.title(r'RMS=' + ("%.3f" % mean_error_r), fontsize=20)
        plt.ylabel('y/pixels')
        plt.grid(True)

        optimizeflag = 0
        if optimizeflag:
            newcorners_rep_l, newcorners_rep_r, R_w2l, T_w2l, R_w2r, T_w2r = optimized_byScipy(Objpoints, corners_undis_l, corners_undis_r, R_w2l, T_w2l, R_w2r, T_w2r, camera_matrixL, camera_matrixR, R, T)
            mean_error_l = get_RMS(corners_undis_l, newcorners_rep_l)
            print(f'优化重投影误差l1:{mean_error_l}')
            mean_error_r = get_RMS(corners_undis_r, newcorners_rep_r)
            print(f'优化重投影误差r1:{mean_error_r}')
            plt.subplot(223)
            plt.scatter(corners_undis_l[:, 0], corners_undis_l[:, 1], marker='x', color='red', s=40, label='image points')
            #                   记号形状       颜色           点的大小    设置标签
            plt.scatter(newcorners_rep_l[:, 0], newcorners_rep_l[:, 1], marker='+', color='blue', s=40, label='reproject points')

            plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置

            plt.title(r'optimizedRMS=' + ("%.3f" % mean_error_l), fontsize=20)

            plt.xlabel('x/pixels')
            plt.ylabel('y/pixels')
            plt.grid(True)
            plt.subplot(224)
            plt.scatter(corners_undis_r[:, 0], corners_undis_r[:, 1], marker='x', color='red', s=40, label='image points')
            #                   记号形状       颜色           点的大小    设置标签
            plt.scatter(newcorners_rep_r[:, 0], newcorners_rep_r[:, 1], marker='+', color='blue', s=40, label='reproject points')

            plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置

            plt.title(r'optimizedRMS=' + ("%.3f" % mean_error_r), fontsize=20)

            plt.xlabel('x/pixels')
            plt.ylabel('y/pixels')
            plt.grid(True)
            plt.show()

        # 图像像素点x 分别是 corners_undis_l和corners_undis_r, 空间点坐标是Objpoints
        myoptimiflag = not optimizeflag
        if myoptimiflag:
            R_w2l, T_w2l, R_w2r, T_w2r, newcorners_rep_l,newcorners_rep_r, RMS_l, RMS_r = myLMoptimized(Objpoints, corners_undis_l, corners_undis_r, camera_matrixL, camera_matrixR, R_w2l, T_w2l, R_w2r, T_w2r)
            R_w2l_matslist.append(np.array(R_w2l)), T_w2l_matslist.append(np.array(T_w2l))  # 结束当前图片左像机的while, 储存该图片优化后的位姿
            R_w2r_matslist.append(np.array(R_w2r)), T_w2r_matslist.append(np.array(T_w2r))  # 结束当前图片右像机的while, 储存该图片优化后的位姿. 当前图片结束,进入for循环下一张图片处理
            plt.subplot(223)
            plt.scatter(corners_undis_l[:, 0], corners_undis_l[:, 1], marker='x', color='red', s=40, label='image points')
            #                   记号形状       颜色           点的大小    设置标签
            plt.scatter(newcorners_rep_l[:, 0], newcorners_rep_l[:, 1], marker='+', color='blue', s=40, label='reproject points')

            plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置

            plt.title(r'MyoptimizedRMS=' + ("%.3f" % RMS_l), fontsize=20)

            plt.xlabel('x/pixels')
            plt.ylabel('y/pixels')
            plt.grid(True)

            plt.subplot(224)
            plt.scatter(corners_undis_r[:, 0], corners_undis_r[:, 1], marker='x', color='red', s=40, label='image points')
            #                   记号形状       颜色           点的大小    设置标签
            plt.scatter(newcorners_rep_r[:, 0], newcorners_rep_r[:, 1], marker='+', color='blue', s=40, label='reproject points')

            plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置

            plt.title(r'MyoptimizedRMS=' + ("%.3f" % RMS_r), fontsize=20)

            plt.xlabel('x/pixels')
            plt.ylabel('y/pixels')
            plt.grid(True)
            # plt.ion(), plt.pause(3), plt.close()
            plt.show()
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{index} Img has been done~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

    # print('\nAll done\n')
    # print('初始双目之间的R:\n', R), print('初始双目之间的T:\n', T)
    # fs = cv.FileStorage("./StereoPositionGestureData/StereoPositionGestureData_C0_C1.yml", cv.FILE_STORAGE_WRITE)
    # fs.write('initialR', R)
    # fs.write('initialT', T)
    # fs.write('R_w2l_matslist', np.asarray(R_w2l_matslist))
    # fs.write('T_w2l_matslist', np.asarray(T_w2l_matslist))
    # fs.write('R_w2r_matslist', np.asarray(R_w2r_matslist))
    # fs.write('T_w2r_matslist', np.asarray(T_w2r_matslist))
    # fs.release()

