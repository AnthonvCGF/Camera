import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from glob import glob
from scipy.optimize import least_squares
from scipy import linalg
import time

def optimize_params(imgpointsl, imgpointsr, objpoints, Paramater):  # n*2,n*3
    def func(initialData):
        fline = []  # Collinear constraint equation

        # 左像机
        r0_l, r1_l, r2_l, Tx_l = initialData[0], initialData[1], initialData[2], initialData[3]
        r3_l, r4_l, r5_l, Ty_l = initialData[4], initialData[5], initialData[6], initialData[7]
        r6_l, r7_l, r8_l, Tz_l = initialData[8], initialData[9], initialData[10], initialData[11]
        # 右像机
        r0_r, r1_r, r2_r, Tx_r = initialData[12], initialData[13], initialData[14], initialData[15]
        r3_r, r4_r, r5_r, Ty_r = initialData[16], initialData[17], initialData[18], initialData[19]
        r6_r, r7_r, r8_r, Tz_r = initialData[20], initialData[21], initialData[22], initialData[23]

        for i in range(imgpointsl.shape[0]):
            # 非线性方程组共线方程 , 形式一目了然, 计算量大.
            xl, yl = imgpointsl[i, 0], imgpointsl[i, 1]
            xr, yr = imgpointsr[i, 0], imgpointsr[i, 1]
            # 左像机
            x_re_l = ((r0_l * objpoints[i, 0] + r1_l * objpoints[i, 1] + r2_l * objpoints[i, 2] + Tx_l) / (
                    r6_l * objpoints[i, 0] + r7_l * objpoints[i, 1] + r8_l * objpoints[i, 2] + Tz_l)) * Fxl + Cxl
            y_re_l = ((r3_l * objpoints[i, 0] + r4_l * objpoints[i, 1] + r5_l * objpoints[i, 2] + Ty_l) / (
                    r6_l * objpoints[i, 0] + r7_l * objpoints[i, 1] + r8_l * objpoints[i, 2] + Tz_l)) * Fyl + Cyl
            fline.append(xl - x_re_l), fline.append(yl - y_re_l)
            # 右像机
            x_re_r = ((r0_r * objpoints[i, 0] + r1_r * objpoints[i, 1] + r2_r * objpoints[i, 2] + Tx_r) / (
                    r6_r * objpoints[i, 0] + r7_r * objpoints[i, 1] + r8_r * objpoints[i, 2] + Tz_r)) * Fxr + Cxr
            y_re_r = ((r3_r * objpoints[i, 0] + r4_r * objpoints[i, 1] + r5_r * objpoints[i, 2] + Ty_r) / (
                    r6_r * objpoints[i, 0] + r7_r * objpoints[i, 1] + r8_r * objpoints[i, 2] + Tz_r)) * Fyr + Cyr
            fline.append(xr - x_re_r), fline.append(yr - y_re_r)

        # 左像机正交约束
        fline.append(r0_l ** 2 + r3_l ** 2 + r6_l ** 2 - 1)
        fline.append(r1_l ** 2 + r4_l ** 2 + r7_l ** 2 - 1)
        fline.append(r2_l ** 2 + r5_l ** 2 + r8_l ** 2 - 1)
        fline.append(r0_l * r1_l + r3_l * r4_l + r6_l * r7_l)
        fline.append(r0_l * r2_l + r3_l * r5_l + r6_l * r8_l)
        fline.append(r1_l * r2_l + r4_l * r5_l + r7_l * r8_l)
        # 右像机正交约束
        fline.append(r0_r ** 2 + r3_r ** 2 + r6_r ** 2 - 1)
        fline.append(r1_r ** 2 + r4_r ** 2 + r7_r ** 2 - 1)
        fline.append(r2_r ** 2 + r5_r ** 2 + r8_r ** 2 - 1)
        fline.append(r0_r * r1_r + r3_r * r4_r + r6_r * r7_r)
        fline.append(r0_r * r2_r + r3_r * r5_r + r6_r * r8_r)
        fline.append(r1_r * r2_r + r4_r * r5_r + r7_r * r8_r)

        mat = np.asarray(fline, dtype=np.float64)
        return mat

    R_w2l, T_w2l, R_w2r, T_w2r, camera_matrixL, camera_matrixR, R, T = Paramater
    Fxl, Fyl, Cxl, Cyl = camera_matrixL[0, 0], camera_matrixL[1, 1], camera_matrixL[0, 2], camera_matrixL[1, 2]
    Fxr, Fyr, Cxr, Cyr = camera_matrixR[0, 0], camera_matrixR[1, 1], camera_matrixR[0, 2], camera_matrixR[1, 2]

    initialData = [
        R_w2l[0, 0], R_w2l[0, 1], R_w2l[0, 2], T_w2l[0, 0],
        R_w2l[1, 0], R_w2l[1, 1], R_w2l[1, 2], T_w2l[1, 0],
        R_w2l[2, 0], R_w2l[2, 1], R_w2l[2, 2], T_w2l[2, 0],
        R_w2r[0, 0], R_w2r[0, 1], R_w2r[0, 2], T_w2r[0, 0],
        R_w2r[1, 0], R_w2r[1, 1], R_w2r[1, 2], T_w2r[1, 0],
        R_w2r[2, 0], R_w2r[2, 1], R_w2r[2, 2], T_w2r[2, 0],
    ]
    mat = func(initialData)
    root = least_squares(func, initialData, method='lm')
    print("Status : ", root.status)
    result = root.x
    R_w2l[0, 0], R_w2l[0, 1], R_w2l[0, 2], T_w2l[0, 0] = result[0], result[1], result[2], result[3]
    R_w2l[1, 0], R_w2l[1, 1], R_w2l[1, 2], T_w2l[1, 0] = result[4], result[5], result[6], result[7]
    R_w2l[2, 0], R_w2l[2, 1], R_w2l[2, 2], T_w2l[2, 0] = result[8], result[9], result[10], result[11]
    R_w2r[0, 0], R_w2r[0, 1], R_w2r[0, 2], T_w2r[0, 0] = result[12], result[13], result[14], result[15]
    R_w2r[1, 0], R_w2r[1, 1], R_w2r[1, 2], T_w2r[1, 0] = result[16], result[17], result[18], result[19]
    R_w2r[2, 0], R_w2r[2, 1], R_w2r[2, 2], T_w2r[2, 0] = result[20], result[21], result[22], result[23]
    T_w2l, T_w2r = T_w2l.reshape(3, 1), T_w2r.reshape(3, 1)
    return R_w2l, T_w2l, R_w2r, T_w2r


def optimized_byScipy(Objpoints, corners_undis_l, corners_undis_r, R_w2l, T_w2l, R_w2r, T_w2r, camera_matrixL, camera_matrixR, R, T):
    print('开启scipy优化')
    Paramater = [R_w2l, T_w2l, R_w2r, T_w2r, camera_matrixL, camera_matrixR, R, T]
    R_w2l, T_w2l, R_w2r, T_w2r = optimize_params(corners_undis_l, corners_undis_r, Objpoints, Paramater)
    newcorners_rep_l = cv.projectPoints(Objpoints, R_w2l, T_w2l, camera_matrixL, np.zeros((5, 1)))
    newcorners_rep_l = newcorners_rep_l[0].reshape(newcorners_rep_l[0].shape[0], 2)
    newcorners_rep_r = cv.projectPoints(Objpoints, R_w2r, T_w2r, camera_matrixR, np.zeros((5, 1)))
    newcorners_rep_r = newcorners_rep_r[0].reshape(newcorners_rep_r[0].shape[0], 2)
    return newcorners_rep_l, newcorners_rep_r, R_w2l, T_w2l, R_w2r, T_w2r