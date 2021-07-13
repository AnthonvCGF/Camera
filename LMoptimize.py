import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from glob import glob
from Optimized_byScipy import *
from scipy import linalg
import time


def myLMoptimized(Objpoints, corners_undis_l, corners_undis_r, camera_matrixL, camera_matrixR, R_w2l, T_w2l, R_w2r, T_w2r):
    Fxl, Fyl, Cxl, Cyl = camera_matrixL[0, 0], camera_matrixL[1, 1], camera_matrixL[0, 2], camera_matrixL[1, 2]
    Fxr, Fyr, Cxr, Cyr = camera_matrixR[0, 0], camera_matrixR[1, 1], camera_matrixR[0, 2], camera_matrixR[1, 2]
    last_RMS_l = 0
    iteration_l, step_l, iteration_num_l = 0, 1000, 1000

    while iteration_l < iteration_num_l:
        r0_l, r1_l, r2_l, Tx_l = R_w2l[0, 0], R_w2l[0, 1], R_w2l[0, 2], T_w2l[0, 0]
        r3_l, r4_l, r5_l, Ty_l = R_w2l[1, 0], R_w2l[1, 1], R_w2l[1, 2], T_w2l[1, 0]
        r6_l, r7_l, r8_l, Tz_l = R_w2l[2, 0], R_w2l[2, 1], R_w2l[2, 2], T_w2l[2, 0]
        RMS_l = 0  # RMS其实就是迭代优化里的cost, 同时,RMS也是优化过程中的待优化函数f(x)
        # error_l_x, error_l_y = 0, 0

        for i in range(Objpoints.shape[0]):
            x_l, y_l = corners_undis_l[i, 0], corners_undis_l[i, 1]
            Xc_l = r0_l * Objpoints[i, 0] + r1_l * Objpoints[i, 1] + r2_l * Objpoints[i, 2] + Tx_l
            Yc_l = r3_l * Objpoints[i, 0] + r4_l * Objpoints[i, 1] + r5_l * Objpoints[i, 2] + Ty_l
            Zc_l = r6_l * Objpoints[i, 0] + r7_l * Objpoints[i, 1] + r8_l * Objpoints[i, 2] + Tz_l
            x_l_, y_l_ = Fxl * Xc_l / Zc_l + Cxl, Fyl * Yc_l / Zc_l + Cyl
            RMS_l += (x_l_ - x_l) ** 2 + (y_l_ - y_l) ** 2
            # error_l_x += (x_l_ - x_l)
            # error_l_y += (y_l_ - y_l)

        RMS_l = np.sqrt(RMS_l / Objpoints.shape[0])
        if iteration_l == 0:
            print(f'-----------左像机初始值RMS_l:{RMS_l}-----------')
            # print('error_l_x:', error_l_x)
            # print('error_l_y:', error_l_y)

        #  假设更新了位姿, 计算更新后的误差函数----------------------------------------------这是假设!!
        adjustnum_l, doneflag_l = 0, 0
        while 1:  # 该循环第一次的结果将与初始值RMS一模一样,所以得跑一次循环才是加上了增量才能看增量后的RMS
            tempR_w2l, tempT_w2l = R_w2l, T_w2l
            tempr0_l, tempr1_l, tempr2_l, tempTx_l = tempR_w2l[0, 0], tempR_w2l[0, 1], tempR_w2l[0, 2], tempT_w2l[0, 0]
            tempr3_l, tempr4_l, tempr5_l, tempTy_l = tempR_w2l[1, 0], tempR_w2l[1, 1], tempR_w2l[1, 2], tempT_w2l[1, 0]
            tempr6_l, tempr7_l, tempr8_l, tempTz_l = tempR_w2l[2, 0], tempR_w2l[2, 1], tempR_w2l[2, 2], tempT_w2l[2, 0]
            tempgrad_l = np.zeros([2, 6], dtype=np.float64)
            tempH_l = np.zeros([6, 6], dtype=np.float64)
            tempb_l = np.zeros([6, 1], dtype=np.float64)
            tempcovmat_l = np.zeros([6, 6], dtype=np.float64)
            tempRMS_l = 0
            temperror_l_x, temperror_l_y = 0, 0
            for i in range(Objpoints.shape[0]):
                tempx_l, tempy_l = corners_undis_l[i, 0], corners_undis_l[i, 1]
                tempXc_l = tempr0_l * Objpoints[i, 0] + tempr1_l * Objpoints[i, 1] + tempr2_l * Objpoints[i, 2] + tempTx_l
                tempYc_l = tempr3_l * Objpoints[i, 0] + tempr4_l * Objpoints[i, 1] + tempr5_l * Objpoints[i, 2] + tempTy_l
                tempZc_l = tempr6_l * Objpoints[i, 0] + tempr7_l * Objpoints[i, 1] + tempr8_l * Objpoints[i, 2] + tempTz_l
                tempx_l_, tempy_l_ = Fxl * tempXc_l / tempZc_l + Cxl, Fyl * tempYc_l / tempZc_l + Cyl
                tempe = np.array([tempx_l_ - tempx_l, tempy_l_ - tempy_l], dtype=np.float64).reshape(2, 1)
                tempRMS_l += (tempx_l_ - tempx_l) ** 2 + (tempy_l_ - tempy_l) ** 2
                tempJ_l = np.zeros([2, 6], dtype=np.float64).reshape(2, 6)  # 前三列是对平移偏导,后三列是对旋转偏导
                tempJ_l[0, 0], tempJ_l[0, 1], tempJ_l[0, 2], tempJ_l[0, 3], tempJ_l[0, 4], tempJ_l[0, 5] = Fxl / tempZc_l, 0, - Fxl * tempXc_l / (
                        tempZc_l ** 2), \
                                                                                                           -Fxl * tempXc_l * tempYc_l / (tempZc_l ** 2), \
                                                                                                           Fxl * (1 + (tempXc_l ** 2) / (
                                                                                                                   tempZc_l ** 2)), -Fxl * tempYc_l / tempZc_l
                tempJ_l[1, 0], tempJ_l[1, 1], tempJ_l[1, 2], tempJ_l[1, 3], tempJ_l[1, 4], tempJ_l[1, 5] = 0, Fyl / tempZc_l, - Fyl * tempYc_l / (
                        tempZc_l ** 2), \
                                                                                                           -Fyl * (1 + (tempYc_l ** 2) / (
                                                                                                                   tempZc_l ** 2)), \
                                                                                                           Fyl * tempXc_l * tempYc_l / (
                                                                                                                   tempZc_l ** 2), Fyl * tempXc_l / tempZc_l
                temperror_l_x += (tempx_l_ - tempx_l)
                temperror_l_y += (tempy_l_ - tempy_l)
                tempJ_l = np.mat(tempJ_l)
                tempgrad_l += tempJ_l
                tempH_l += tempJ_l.T * tempJ_l
                tempb_l += -tempJ_l.T * tempe
            tempRMS_l = np.sqrt(tempRMS_l / Objpoints.shape[0])

            if tempRMS_l < RMS_l and adjustnum_l > 0:
                # print(f'找到当前step{step_l},可使得RMS下降')
                RMS_l = tempRMS_l
                dx_l = tempdx_l
                linear_increment_l = -tempgrad_l @ dx_l.T
                R_w2l = tempR_w2l  # 进行真正的位姿更新------------------------------------这才是真正的
                T_w2l = tempT_w2l
                rou_l = linalg.norm(np.array([linear_increment_l[0, 0] / temperror_l_x, linear_increment_l[1, 0] / temperror_l_y])) / np.sqrt(2)
                if abs(1 - rou_l) < 1e-1:
                    step_l *= 3 / 4
                else:
                    step_l *= 1 / 8
                break

            for k in range(tempH_l.shape[0]):
                tempcovmat_l[k, k] = tempH_l[k, k] / Objpoints.shape[0]
            tempdx_l = linalg.solve(tempH_l + step_l * tempcovmat_l, tempb_l).T  # todo :1*6
            temp_deltaR_w2l, _ = cv.Rodrigues(tempdx_l[0, 3:])
            tempR_w2l *= temp_deltaR_w2l
            tempT_w2l[0, 0], tempT_w2l[1, 0], tempT_w2l[2, 0] = tempT_w2l[0, 0] + tempdx_l[0, 0], tempT_w2l[1, 0] + tempdx_l[0, 1], tempT_w2l[2, 0] + \
                                                                tempdx_l[0, 2]
            if adjustnum_l == 0:
                adjustnum_l += 1
                continue

            if tempRMS_l > RMS_l:
                if step_l < 1e+4:
                    step_l = step_l * (1 + 0.1 * np.log2(adjustnum_l)) if adjustnum_l > 2 else step_l * (1 + 0.1 * np.exp(adjustnum_l))
                else:
                    step_l *= (1 + 0.1 * np.exp(adjustnum_l))
                adjustnum_l += 1
                # print(f'当前重调次数为{adjustnum_l},为使RMS下降,增大step为{step_l}')

            if step_l > 1e+9:  # 结束step寻找条件
                doneflag_l = 1
                break

        '''旋转矩阵换个数学说法可以说成是属于李群中的一个子集,所以所有的旋转矩阵的集合构成了一个李群,因为旋转矩阵的正交性所以称之为特殊正交群,记作SO(3) '''
        '''旋转向量就是李代数了,特殊正交群的李代数记为so(3)。SO(3)如果对应一个球面,那么so(3)就对应着球面某点的上的切面'''
        '''SO(3) = exp(so(3)), 即李代数指数映射后变为李群,所以李代数做加法等于李群做乘法, 所以想更新R矩阵为R',那么是R'=exp(R的李代数)*R '''
        '''补充, 罗德里格斯公式是指数映射即李代数变李群也即是旋转向量变旋转矩阵, 但是cv.Rodrigues函数同时包含了李代数到李群以及李群到李代数的转换'''

        if iteration_l > 0:
            # print(f'=======================第{iteration_l}次迭代开始=======================')
            # print('linear_increment_l:', linear_increment_l)
            # print('error_l_x:', error_l_x)
            # print('error_l_y:', error_l_y)
            # if RMS_l - last_RMS_l < 0:  # 只是怕有漏网之鱼
            # print('RMS_l < last_RMS_l,迭代正确,相差:', RMS_l - last_RMS_l)
            # print('rou_l:', rou_l)
            # print('RMS_l:', RMS_l)

            #  迭代结束条件
            if abs(RMS_l - last_RMS_l) < 1e-12:
                print(f'step_l最后更新为:{step_l}')
                print('RMS_l最后为:', RMS_l)
                print(f'相邻两次RMS小于1e-12,退出迭代,该图片在左像机的处理结束')
                break

            elif iteration_l == iteration_num_l - 2:
                # print(f'============step最后更新为:{step_l},该图片处理结束=============')
                print(f'达到最大迭代次数{iteration_l},当前RMS与上一次RMS之差为{RMS_l - last_RMS_l}')
                print('RMS_l最后为:', RMS_l)
                break
            elif doneflag_l:
                print('RMS_l最后为:', RMS_l)
                print(f'============本身就很棒了优化不动,该图片处理结束=============')
                break

            # print(f'============step更新为:{step_l},本次迭代结束=============\n')
        #  误差函数值更新
        last_RMS_l = RMS_l
        iteration_l += 1


    last_RMS_r = 0
    iteration_r, step_r, iteration_num_r = 0, 1000, 1000
    while iteration_r < iteration_num_r:
        r0_r, r1_r, r2_r, Tx_r = R_w2r[0, 0], R_w2r[0, 1], R_w2r[0, 2], T_w2r[0, 0]
        r3_r, r4_r, r5_r, Ty_r = R_w2r[1, 0], R_w2r[1, 1], R_w2r[1, 2], T_w2r[1, 0]
        r6_r, r7_r, r8_r, Tz_r = R_w2r[2, 0], R_w2r[2, 1], R_w2r[2, 2], T_w2r[2, 0]
        RMS_r = 0  # RMS其实就是迭代优化里的cost, 同时,RMS也是优化过程中的待优化函数f(x)
        # error_r_x, error_r_y = 0, 0

        for i in range(Objpoints.shape[0]):
            x_r, y_r = corners_undis_r[i, 0], corners_undis_r[i, 1]
            Xc_r = r0_r * Objpoints[i, 0] + r1_r * Objpoints[i, 1] + r2_r * Objpoints[i, 2] + Tx_r
            Yc_r = r3_r * Objpoints[i, 0] + r4_r * Objpoints[i, 1] + r5_r * Objpoints[i, 2] + Ty_r
            Zc_r = r6_r * Objpoints[i, 0] + r7_r * Objpoints[i, 1] + r8_r * Objpoints[i, 2] + Tz_r
            x_r_, y_r_ = Fxr * Xc_r / Zc_r + Cxr, Fyr * Yc_r / Zc_r + Cyr
            RMS_r += (x_r_ - x_r) ** 2 + (y_r_ - y_r) ** 2
            # error_r_x += (x_r_ - x_r)
            # error_r_y += (y_r_ - y_r)

        RMS_r = np.sqrt(RMS_r / Objpoints.shape[0])
        if iteration_r == 0:
            print(f'-----------右像机初始值RMS_r:{RMS_r}-----------')
            # print('error_r_x:', error_r_x)
            # print('error_r_y:', error_r_y)

        #  假设更新了位姿, 计算更新后的误差函数----------------------------------------------这是假设!!
        adjustnum_r, doneflag_r = 0, 0
        while 1:  # 该循环第一次的结果将与初始值RMS一模一样,所以得跑一次循环才是加上了增量才能看增量后的RMS
            tempR_w2r, tempT_w2r = R_w2r, T_w2r
            tempr0_r, tempr1_r, tempr2_r, tempTx_r = tempR_w2r[0, 0], tempR_w2r[0, 1], tempR_w2r[0, 2], tempT_w2r[0, 0]
            tempr3_r, tempr4_r, tempr5_r, tempTy_r = tempR_w2r[1, 0], tempR_w2r[1, 1], tempR_w2r[1, 2], tempT_w2r[1, 0]
            tempr6_r, tempr7_r, tempr8_r, tempTz_r = tempR_w2r[2, 0], tempR_w2r[2, 1], tempR_w2r[2, 2], tempT_w2r[2, 0]
            tempgrad_r = np.zeros([2, 6], dtype=np.float64)
            tempH_r = np.zeros([6, 6], dtype=np.float64)
            tempb_r = np.zeros([6, 1], dtype=np.float64)
            tempcovmat_r = np.zeros([6, 6], dtype=np.float64)
            tempRMS_r = 0
            temperror_r_x, temperror_r_y = 0, 0
            for i in range(Objpoints.shape[0]):
                tempx_r, tempy_r = corners_undis_r[i, 0], corners_undis_r[i, 1]
                tempXc_r = tempr0_r * Objpoints[i, 0] + tempr1_r * Objpoints[i, 1] + tempr2_r * Objpoints[i, 2] + tempTx_r
                tempYc_r = tempr3_r * Objpoints[i, 0] + tempr4_r * Objpoints[i, 1] + tempr5_r * Objpoints[i, 2] + tempTy_r
                tempZc_r = tempr6_r * Objpoints[i, 0] + tempr7_r * Objpoints[i, 1] + tempr8_r * Objpoints[i, 2] + tempTz_r
                tempx_r_, tempy_r_ = Fxr * tempXc_r / tempZc_r + Cxr, Fyr * tempYc_r / tempZc_r + Cyr
                tempe_r = np.array([tempx_r_ - tempx_r, tempy_r_ - tempy_r], dtype=np.float64).reshape(2, 1)
                tempRMS_r += (tempx_r_ - tempx_r) ** 2 + (tempy_r_ - tempy_r) ** 2
                tempJ_r = np.zeros([2, 6], dtype=np.float64).reshape(2, 6)  # 前三列是对平移偏导,后三列是对旋转偏导
                tempJ_r[0, 0], tempJ_r[0, 1], tempJ_r[0, 2], tempJ_r[0, 3], tempJ_r[0, 4], tempJ_r[0, 5] = Fxr / tempZc_r, 0, - Fxr * tempXc_r / (
                        tempZc_r ** 2), \
                                                                                                           -Fxr * tempXc_r * tempYc_r / (tempZc_r ** 2), \
                                                                                                           Fxr * (1 + (tempXc_r ** 2) / (
                                                                                                                   tempZc_r ** 2)), -Fxr * tempYc_r / tempZc_r
                tempJ_r[1, 0], tempJ_r[1, 1], tempJ_r[1, 2], tempJ_r[1, 3], tempJ_r[1, 4], tempJ_r[1, 5] = 0, Fyr / tempZc_r, - Fyr * tempYc_r / (
                        tempZc_r ** 2), \
                                                                                                           -Fyr * (1 + (tempYc_r ** 2) / (
                                                                                                                   tempZc_r ** 2)), \
                                                                                                           Fyr * tempXc_r * tempYc_r / (
                                                                                                                   tempZc_r ** 2), Fyr * tempXc_r / tempZc_r
                temperror_r_x += (tempx_r_ - tempx_r)
                temperror_r_y += (tempy_r_ - tempy_r)
                tempJ_r = np.mat(tempJ_r)
                tempgrad_r += tempJ_r
                tempH_r += tempJ_r.T * tempJ_r
                tempb_r += -tempJ_r.T * tempe_r
            tempRMS_r = np.sqrt(tempRMS_r / Objpoints.shape[0])

            if tempRMS_r < RMS_r and adjustnum_r > 0:
                # print(f'找到当前step{step_r},可使得RMS下降')
                RMS_r = tempRMS_r
                dx_r = tempdx_r
                linear_increment_r = -tempgrad_r @ dx_r.T
                R_w2r = tempR_w2r  # 进行真正的位姿更新------------------------------------这才是真正的
                T_w2r = tempT_w2r
                rou_r = linalg.norm(np.array([linear_increment_r[0, 0] / temperror_r_x, linear_increment_r[1, 0] / temperror_r_y])) / np.sqrt(2)
                if abs(1 - rou_r) < 1e-1:
                    step_r *= 3 / 4
                else:
                    step_r *= 1 / 8
                break

            for k in range(tempH_r.shape[0]):
                tempcovmat_r[k, k] = tempH_r[k, k] / Objpoints.shape[0]
            tempdx_r = linalg.solve(tempH_r + step_r * tempcovmat_r, tempb_r).T  # todo :1*6
            temp_deltaR_w2r, _ = cv.Rodrigues(tempdx_r[0, 3:])
            tempR_w2r *= temp_deltaR_w2r
            tempT_w2r[0, 0], tempT_w2r[1, 0], tempT_w2r[2, 0] = tempT_w2r[0, 0] + tempdx_r[0, 0], tempT_w2r[1, 0] + tempdx_r[0, 1], tempT_w2r[2, 0] + \
                                                                tempdx_r[0, 2]
            if adjustnum_r == 0:
                adjustnum_r += 1
                continue

            if tempRMS_r > RMS_r:
                if step_r < 1e+4:
                    step_r = step_r * (1 + 0.1 * np.log2(adjustnum_r)) if adjustnum_r > 2 else step_r * (1 + 0.1 * np.exp(adjustnum_r))
                else:
                    step_r *= (1 + 0.1 * np.exp(adjustnum_r))
                adjustnum_r += 1
                # print(f'当前重调次数为{adjustnum_r},为使RMS下降,增大step为{step_r}')

            if step_r > 1e+9:  # 结束step寻找条件
                doneflag_r = 1
                break

        '''旋转矩阵换个数学说法可以说成是属于李群中的一个子集,所以所有的旋转矩阵的集合构成了一个李群,因为旋转矩阵的正交性所以称之为特殊正交群,记作SO(3) '''
        '''旋转向量就是李代数了,特殊正交群的李代数记为so(3)。SO(3)如果对应一个球面,那么so(3)就对应着球面某点的上的切面'''
        '''SO(3) = exp(so(3)), 即李代数指数映射后变为李群,所以李代数做加法等于李群做乘法, 所以想更新R矩阵为R',那么是R'=exp(R的李代数)*R '''
        '''补充, 罗德里格斯公式是指数映射即李代数变李群也即是旋转向量变旋转矩阵, 但是cv.Rodrigues函数同时包含了李代数到李群以及李群到李代数的转换'''

        if iteration_r > 0:
            # print(f'=======================第{iteration_r}次迭代开始=======================')
            # print('linear_increment_r:', linear_increment_r)
            # print('error_r_x:', error_r_x)
            # print('error_r_y:', error_r_y)
            # if RMS_r - last_RMS_r < 0:  # 只是怕有漏网之鱼
            # print('RMS_r < last_RMS_r,迭代正确,相差:', RMS_r - last_RMS_r)
            # print('rou_r:', rou_r)
            # print('RMS_r:', RMS_r)

            #  迭代结束条件
            if abs(RMS_r - last_RMS_r) < 1e-12:
                print(f'step_r最后更新为:{step_r}')
                print('RMS_r最后为:', RMS_r)
                print(f'相邻两次RMS小于1e-12,退出迭代,该图片在右像机的处理结束')
                break

            elif iteration_r == iteration_num_r - 2:
                # print(f'============step最后更新为:{step_r},该图片处理结束=============')
                print(f'达到最大迭代次数{iteration_r},当前RMS与上一次RMS之差为{RMS_r - last_RMS_r}')
                print('RMS_r最后为:', RMS_r)
                break
            elif doneflag_r:
                print('RMS_r最后为:', RMS_r)
                print(f'============本身就很棒了优化不动,该图片处理结束=============')
                break

            # print(f'============step更新为:{step_r},本次迭代结束=============\n')
        #  误差函数值更新
        last_RMS_r = RMS_r
        iteration_r += 1
    newcorners_rep_l = cv.projectPoints(Objpoints, R_w2l, T_w2l, camera_matrixL, np.zeros((5, 1)))
    newcorners_rep_l = newcorners_rep_l[0].reshape(newcorners_rep_l[0].shape[0], 2)
    newcorners_rep_r = cv.projectPoints(Objpoints, R_w2r, T_w2r, camera_matrixR, np.zeros((5, 1)))
    newcorners_rep_r = newcorners_rep_r[0].reshape(newcorners_rep_r[0].shape[0], 2)
    return R_w2l, T_w2l, R_w2r, T_w2r,newcorners_rep_l,newcorners_rep_r, RMS_l, RMS_r
