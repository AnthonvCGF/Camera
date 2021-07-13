import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from glob import glob
import os
from scipy import linalg


def getCamePara():
    fs = cv.FileStorage(cv.samples.findFile('./CameraParameter/cameraPara_C0_C1.yml'), cv.FILE_STORAGE_READ)
    # board_width = int(fs.getNode('board_width').real())
    # board_height = int(fs.getNode('board_height').real())
    square_size = fs.getNode('square_size').real()
    camera_matrix = fs.getNode('camera_matrix').mat()
    print(camera_matrix)
    extrinsics = fs.getNode('extrinsic_parameters').mat()
    dist_coefs = fs.getNode('dist_coefs').mat()
    fs.release()
    return square_size, camera_matrix, extrinsics, dist_coefs


# 求输入数据的归一化矩阵
def normalizing_input_data(coor_data):  # 输入数据 n*2
    x_avg = np.mean(coor_data[:, 0])
    y_avg = np.mean(coor_data[:, 1])
    sx = np.sqrt(2) / np.std(coor_data[:, 0])
    sy = np.sqrt(2) / np.std(coor_data[:, 1])

    norm_matrix = np.matrix([[sx, 0, -sx * x_avg],
                             [0, sy, -sy * y_avg],
                             [0, 0, 1]])
    return norm_matrix


def getH_by2D_3D(pic_coor, real_coor):  # 求的是单应矩阵, CSDN有收藏到详细推导的博文
    ''''''
    # 获得归一化矩阵
    pic_norm_mat = normalizing_input_data(pic_coor)
    real_norm_mat = normalizing_input_data(real_coor)
    M = []
    for i in range(len(pic_coor)):
        # 转换为齐次坐标
        single_pic_coor = np.array([pic_coor[i][0], pic_coor[i][1], 1])
        single_real_coor = np.array([real_coor[i][0], real_coor[i][1], 1])
        # 坐标归一化
        picxy = np.dot(pic_norm_mat, single_pic_coor)
        objXY = np.dot(real_norm_mat, single_real_coor)

        # 构造M矩阵
        M.append(np.array([-objXY.item(0), -objXY.item(1), -1, 0, 0, 0,
                           picxy.item(0) * objXY.item(0), picxy.item(0) * objXY.item(1),
                           picxy.item(0)]))

        M.append(np.array([0, 0, 0, -objXY.item(0), -objXY.item(1), -1,
                           picxy.item(1) * objXY.item(0), picxy.item(1) * objXY.item(1),
                           picxy.item(1)]))
    # 利用SVD求解M * h = 0中h的解
    U, S, VT = np.linalg.svd((np.array(M, dtype='float')).reshape((-1, 9)))
    # 最小的奇异值对应的奇异向量,S求出来按大小排列的，最后的最小
    H = VT[-1].reshape((3, 3))
    H = np.dot(np.dot(np.linalg.inv(pic_norm_mat), H), real_norm_mat)
    H /= H[-1, -1]
    return H.A


# 返回每一幅图的外参矩阵[R|t]
def get_Extrinsics(H, intrinsics_param):
    '''
    # H的类型需要时ndarray, 因为传matrix会得到h0是1*3或者3*1这都会被认为是1维
    传ndarray则会h0的shape是(3,),看成三维
    '''
    inv_intrinsics_param = np.linalg.inv(intrinsics_param)
    h0 = (H.reshape(3, 3))[:, 0]
    h1 = (H.reshape(3, 3))[:, 1]
    h2 = (H.reshape(3, 3))[:, 2]

    scale_factor = 1 / np.linalg.norm(np.dot(inv_intrinsics_param, h0))

    r0 = scale_factor * np.dot(inv_intrinsics_param, h0)
    r1 = scale_factor * np.dot(inv_intrinsics_param, h1)
    t = scale_factor * np.dot(inv_intrinsics_param, h2)
    r2 = np.cross(r0, r1)

    extrinsics_param = np.array([r0, r1, r2, t]).transpose()

    rvec, _ = cv.Rodrigues(extrinsics_param[:, :3])  # rvec接收的是第一个返回值,得到3*1的旋转向量, 适用于projectPoints函数
    tvec = t  # 用于projectPoints的tvecs也是3*1的平移向量
    return extrinsics_param, rvec, tvec


def get_RMS(point1, point2):
    tot_error = 0
    for i in range(point1.shape[0]):
        tot_error += np.sum(np.abs(point1[i, :] - point2[i, :]) ** 2)
    mean_error = np.sqrt(tot_error / point1.shape[0])
    return mean_error

def draw(corners_undis,corners_rep,mean_error,curImgname):
    plt.scatter(corners_undis[:, 0], corners_undis[:, 1], marker='x', color='red', s=40, label='image points')
    #                   记号形状       颜色           点的大小    设置标签
    plt.scatter(corners_rep[:, 0], corners_rep[:, 1], marker='+', color='blue', s=40, label='reproject points')

    plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置
    plt.title(r'RMS=' + ("%.3f" % mean_error), fontsize=20)
    plt.xlabel('x/pixels')
    plt.ylabel('y/pixels')
    plt.ion(), plt.pause(1), plt.close()
    brotherImgname = curImgname.split('\\')[-1]
    if mean_error > 0.2:
        if os.path.exists(curImgname):
            # os.remove(curImgname)
            # os.remove(brotherImgroot + brotherImgname)
            print(f'成功删除文件:{curImgname}和{brotherImgroot + brotherImgname}')
        else:
            print(f'未找到此文件:{curImgname}和{brotherImgroot + brotherImgname}')
    print(f"----------- {curImgname} has been done -----------")


def MonocularPnP(imgroot, brotherImgroot, square_size, camera_matrix, dist_coefs):
    img_names = glob(imgroot)
    pattern_size = (11, 8)
    pattern_points_3D = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points_3D[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points_3D *= square_size
    h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]  # TODO: use imquery call to retrieve results
    for curImgname in img_names:
        img = cv.imread(curImgname, 0)
        if img is None:
            print("Failed to load", curImgname)
            break
        found, corners = cv.findChessboardCorners(img, pattern_size)  # 初步寻找角点
        rgbimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        if found:
            print(f"----processing {curImgname} -----------")
            criteria = (cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 30, 0.01)  # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
            corners2 = cv.cornerSubPix(img, corners, (3, 3), (-1, -1), criteria)
            cv.drawChessboardCorners(rgbimg, pattern_size, corners2, found)  # 画出角点
            # 去畸变
            corners_undis = cv.undistortPoints(corners2, camera_matrix, dist_coefs)  # 得到的是归一化像点
            corners_undis = corners_undis.reshape(corners_undis.shape[0], 2)
            corners_undis[:, 0] = corners_undis[:, 0] * camera_matrix[0, 0] + camera_matrix[0, 2]  # 乘以Fx和加Cx得实际像点坐标
            corners_undis[:, 1] = corners_undis[:, 1] * camera_matrix[1, 1] + camera_matrix[1, 2]
            # PnP
            retval, rvec, tvec = cv.solvePnP(pattern_points_3D, corners2,
                                             camera_matrix,
                                             dist_coefs, flags=cv.SOLVEPNP_ITERATIVE)

            # 重投影
            corners_rep = cv.projectPoints(pattern_points_3D, rvec, tvec, camera_matrix, np.zeros((4, 1)))
            corners_undis = corners_undis.reshape(corners_undis.shape[0], 2)
            corners_rep = corners_rep[0].reshape(corners_rep[0].shape[0], 2)
            mean_error = get_RMS(corners_undis, corners_rep)

            plt.subplot(1,2,1)
            plt.scatter(corners_undis[:, 0], corners_undis[:, 1], marker='x', color='red', s=40, label='image points')
            #                   记号形状       颜色           点的大小    设置标签
            plt.scatter(corners_rep[:, 0], corners_rep[:, 1], marker='+', color='blue', s=40, label='reproject points')

            plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置
            plt.title(r'cv-EPnP RMS=' + ("%.3f" % mean_error), fontsize=20)
            plt.xlabel('x/pixels')
            plt.ylabel('y/pixels')

            '''--------------------------------用H做单目位姿估计-----------------------'''
            H = getH_by2D_3D(corners_undis, pattern_points_3D)
            extrinsics_param, rvec_myself, tvec_myself = get_Extrinsics(H, camera_matrix)
            corners_rep = cv.projectPoints(pattern_points_3D, rvec_myself, tvec_myself, camera_matrix, np.zeros((4, 1)))
            corners_undis = corners_undis.reshape(corners_undis.shape[0], 2)
            corners_rep = corners_rep[0].reshape(corners_rep[0].shape[0], 2)
            mean_error_H = get_RMS(corners_undis, corners_rep)
            plt.subplot(1, 2, 2)
            plt.scatter(corners_undis[:, 0], corners_undis[:, 1], marker='x', color='red', s=40, label='image points')
            #                   记号形状       颜色           点的大小    设置标签
            plt.scatter(corners_rep[:, 0], corners_rep[:, 1], marker='+', color='blue', s=40, label='reproject points')

            plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置
            plt.title(r'decomH RMS=' + ("%.3f" % mean_error_H), fontsize=20)
            plt.xlabel('x/pixels')
            plt.ylabel('y/pixels')

            # plt.ion(), plt.pause(2), plt.close()
            plt.show()
            brotherImgname = curImgname.split('\\')[-1]
            if mean_error > 0.2 and mean_error_H > 0.2:
                if os.path.exists(curImgname):
                    os.remove(curImgname)
                    os.remove(brotherImgroot + brotherImgname)
                    print(f'成功删除文件:{curImgname}和{brotherImgroot + brotherImgname}')
                else:
                    print(f'未找到此文件:{curImgname}和{brotherImgroot + brotherImgname}')
            print(f"----------- {curImgname} has been done -----------")


if __name__ == '__main__':
    square_size, camera_matrix, extrinsics, dist_coefs = getCamePara()
    if camera_matrix is None:
        from MonoCalibration import monoCalibrate
        pic_dir = r".\C0_P0\*.bmp"
        monoCalibrate(pic_dir)
        square_size, camera_matrix, extrinsics, dist_coefs = getCamePara()
    imgroot, brotherImgroot = './C0_P0/*.bmp', './C1_P0/'
    MonocularPnP(imgroot, brotherImgroot, square_size, camera_matrix, dist_coefs)
