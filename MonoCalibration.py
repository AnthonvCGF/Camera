import cv2 as cv
import numpy as np
import glob


def monoCalibrate(pic_dir):
    def fix_flags():
        flags = None
        flags = cv.CALIB_FIX_PRINCIPAL_POINT
        flags = flags | cv.CALIB_FIX_FOCAL_LENGTH
        # flags = flags | cv.CALIB_FIX_K1
        # flags = flags | cv.CALIB_FIX_K2
        # flags = flags | cv.CALIB_FIX_K3
        # flags = flags | cv.CALIB_FIX_TANGENT_DIST
        return flags

    def getRMS(obj_points, img_points, rvecs, tvecs, mtx, dist):
        total_error = 0
        for i in range(len(rvecs)):
            tot_error = 0
            total_points = 0
            corners_rep = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
            corners_rep = corners_rep[0]
            for j in range(obj_points[0].shape[0]):
                tot_error += np.sum(np.abs(img_points[i][j, :] - corners_rep[j, :]) ** 2)
            mean_error = np.sqrt(tot_error / img_points[i].shape[0])
            total_error += mean_error
            print(f'第{i + 1}张图的重投影误差:', mean_error)
        print('手动计算的rms:', total_error / len(rvecs))

    def SaveParameter(board_size, square_size, camera_matrix, dist_coefs):
        fs = cv.FileStorage("./cameraParameter/cameraPara_C0_C1.yml", cv.FILE_STORAGE_WRITE)
        fs.write('board_width', board_size[0])
        fs.write('board_height', board_size[1])
        fs.write('square_size', square_size)
        fs.write('camera_matrix', camera_matrix)
        fs.write('dist_coefs', dist_coefs)
        fs.release()

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 30, 0.001)
    # 获取标定板角点的位置
    board_size = [11, 8]  # 和你的棋盘图像有关.(横向角点数即列数,纵向角点数即行数),这里测试棋盘格横向9个角点,纵向9个角点
    square_size = 20  # mm
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)  # ((6*9行,3列),数据类型)
    # 第一列填充为0,1,2...8,重复6次 ; 第二列则9个0,9个1,....9个6, 所以第一列存放的是该角点在横向角点中的索引,第二列则是该角点在第几行
    objp[:, :2] = (np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2))*square_size

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    images = glob.glob(pic_dir)
    size = ()

    for index in range(len(images)):
        # left
        img = cv.imread(images[index])
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv.findChessboardCorners(gray, (board_size[0], board_size[1]), None)  # 提取角点的信息
        # print("寻找结果：",ret) #ret表示的是是否查询到，corners表示的是提取到的角点信息
        # print("角点信息：",corners.shape)

        if ret:
            obj_points.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
            cv.drawChessboardCorners(img, (board_size[0], board_size[1]), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
            # cv.imshow('img', img)  # 输出图像
            # cv.waitKey(100)  # 显示2秒
        cv.destroyAllWindows()

    # 相机标定
    rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, size, None, None)
    # #输出标定结果
    print("\nCV标定的RMS:", rms)
    print("-----------------------------------------------------")
    print("mtx（内参数矩阵）:\n", mtx)  # 内参数矩阵
    print("-----------------------------------------------------")
    print("dist（畸变系数）:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("-----------------------------------------------------")
    # print("rvecs（旋转向量）:\n", rvecs[0].shape)  # 旋转向量  # 外参数
    # print("-----------------------------------------------------")
    # print("tvecs:（平移向量）\n", tvecs[0].shape ) # 平移向量   # 外参数

    # 我们可以利用反向投影误差对我们找到的参数的准确性进行估计。
    # 得到的结果越接近 0 越好。有了内部参数，畸变参数和旋转变换矩阵，我们就可以使用 cv.projectPoints() 将对象点转换到图像点。
    # 然后就可以计算变换得到图像与角点检测算法的绝对差了。然后我们计算所有标定图像的误差平均值。
    getRMS(obj_points, img_points, rvecs, tvecs, mtx, dist)

    SaveParameter(board_size, square_size, mtx, dist)


if __name__ == '__main__':
    pic_dir = r".\C0_P0\*.bmp"
    monoCalibrate(pic_dir)
