import cv2
import numpy as np

def Bigeye(image,PointX,PointY,Radius,Strength):
     copyImg = np.zeros(image.shape, np.uint8)
     copyImg = image.copy()
     height = image.shape[0]
     width = image.shape[1]
     Left = 0 if PointX - Radius < 0 else PointX - Radius
     Top = 0 if PointY - Radius < 0 else PointY - Radius
     Right = width-1 if PointX + Radius >= width else PointX + Radius
     Bottom = height - 1 if PointY + Radius >= height else PointY + Radius
     PowRadius = Radius * Radius
     for Y in range(Top,Bottom):
         OffsetY = Y - PointY
         for X in range(Left, Right):
             OffsetX = X - PointX
             XY = OffsetX * OffsetX + OffsetY * OffsetY
             if XY <= PowRadius:
                 ScaleFactor = 1 - XY / PowRadius
                 ScaleFactor = 1 - Strength / 100 * ScaleFactor
                 UX = OffsetX * ScaleFactor + PointX
                 UY = OffsetY * ScaleFactor + PointY
                 PosX = 0 if UX < 0 else UX
                 PosX = width-1 if UX >=width else UX
                 PosY = 0 if UY < 0 else UY
                 PosY = height - 1 if UY >= height else UY
                 # 根据双线性插值法得到UX，UY的值
                 value = BilinearInsert(image, PosX, PosY)
                 # 改变当前 i ，j的值
                 copyImg[Y, X] = value

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

def main():
    src = cv2.imread("shou.jpg")
    eye_image = Bigeye(src,221, 177, 10, 30)
    eye_image = Bigeye(eye_image,291, 150, 10, 30)
    cv2.imshow('eye', eye_image)
    cv2.imwrite("eye.jpg", eye_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
