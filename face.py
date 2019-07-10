import dlib
import cv2
import numpy as np
import math


predictor_path = "D:\\shape_predictor_68_face_landmarks.dat"

# 使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def landmark_dec_dlib_fun(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    land_marks = []

    rects = detector(img_gray, 0)

    for i in range(len(rects)):
        land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[i]).parts()])
        land_marks.append(land_marks_node)
    return land_marks


'''
方法： Interactive Image Warping 局部平移算法
'''


def localTranslationWarp(srcImg, startX, startY, endX, endY, radius):
    '''
    :param srcImg: 所输入的图片
    :param startX: 未移动时的点所在的x坐标
    :param startY: 未移动时的点所在的y坐标
    :param endX: 移动后的点所在的x坐标
    :param endY: 移动后的点所在的y坐标
    :param radius:半径
    :return:修改后的图片
    '''
    ddradius = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()

    # 计算公式中的|m-c|^2
    ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
    H, W, C = srcImg.shape
    for i in range(W):
        for j in range(H):
            # 计算该点是否在形变圆的范围之内
            # 优化，第一步，直接判断是会在（startX,startY)的矩阵框中
            if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                continue

            distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)

            if (distance < ddradius):
                # 计算出（i,j）坐标的原坐标
                # 计算公式中右边平方号里的部分
                ratio = (ddradius - distance) / (ddradius - distance + ddmc)
                ratio = ratio * ratio

                # 映射原位置
                UX = i - ratio * (endX - startX)
                UY = j - ratio * (endY - startY)

                # 根据双线性插值法得到UX，UY的值
                value = BilinearInsert(srcImg, UX, UY)
                # 改变当前 i ，j的值
                copyImg[j, i] = value

    return copyImg


# 双线性插值法
def BilinearInsert(src, ux, uy):
    w, h, c = src.shape
    if c == 3:
        x1 = int(ux)
        x2 = x1 + 1
        y1 = int(uy)
        y2 = y1 + 1

        part1 = src[y1, x1].astype(np.float) * (float(x2) - ux) * (float(y2) - uy)
        part2 = src[y1, x2].astype(np.float) * (ux - float(x1)) * (float(y2) - uy)
        part3 = src[y2, x1].astype(np.float) * (float(x2) - ux) * (uy - float(y1))
        part4 = src[y2, x2].astype(np.float) * (ux - float(x1)) * (uy - float(y1))

        insertValue = part1 + part2 + part3 + part4

        return insertValue.astype(np.int8)


def face_thin_auto(src):
    landmarks = landmark_dec_dlib_fun(src)

    # 如果未检测到人脸关键点，就不进行瘦脸
    if len(landmarks) == 0:
        return

    thin_image = src
    landmarks_node = landmarks[0]
    endPt = landmarks_node[16]
    print('landmarks_node[37]\n ',landmarks_node[37])
    print('landmarks_node[40]\n ', landmarks_node[40])
    print('landmarks_node[43]\n ', landmarks_node[43])
    print('landmarks_node[46]\n ', landmarks_node[46])
    print('landmarks_node[34]\n ', landmarks_node[34])
    for index in range(3, 14, 2):
        start_landmark = landmarks_node[index]
        end_landmark = landmarks_node[index + 2]
        r = math.sqrt((start_landmark[0, 0] - end_landmark[0, 0]) * (start_landmark[0, 0] - end_landmark[0, 0]) +
                      (start_landmark[0, 1] - end_landmark[0, 1]) * (start_landmark[0, 1] - end_landmark[0, 1]))
        thin_image = localTranslationWarp(thin_image, start_landmark[0, 0], start_landmark[0, 1], endPt[0, 0], endPt[0, 1], 1.1*r)
    # 显示
    cv2.imshow('thin', thin_image)
    cv2.imwrite("thin.jpg", thin_image)

def main():
    src = cv2.imread("shou.jpg")
    # cv2.imshow('src', src)
    face_thin_auto(src)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
