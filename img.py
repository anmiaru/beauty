import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    #image = cv2.imread("monalisa.jpg")
    image = cv2.imread(r'd:\shou.jpg')
    #image = plt.imread(os.path.join(sys.path[0], "monalisa.jpg"))
    #plt.tight_layout(w_pad=1.0, h_pad=1.0)
    #transformed_image.save("monalisa_2.jpg")
    #plt.imsave('monalisa_2.jpg', image)
    '''
    points_list1 = [(186, 140), (295, 135), (208, 181), (261, 181), (184, 203), (304, 202),
                   (213, 225), (243, 225), (211, 244), (253, 244), (195, 254), (232, 281), (285, 252)]
    points_list2 = [(186, 140), (295, 135), (208, 181), (261, 181), (190, 203), (300, 202),
                   (213, 225), (243, 225), (213, 244), (251, 244), (202, 252), (232, 281), (276, 250)]
    '''
    #points_list1 = [(207, 248), (240, 276), (286, 285), (325, 270), (349, 238), (353, 197),(345, 157)]
    points_list1 = [(186, 214), (196, 232), (207, 248),(222, 264), (240, 276), (262, 283), (286, 285),
                   (307, 280), (325, 270), (338, 255), (349, 238), (354, 218), (353, 197), (349, 177),
                   (345, 157)]
    points_list2 = [(186, 214), (198, 232), (212, 245), (228, 260), (244, 268), (264, 275), (285, 274),
                    (301, 269), (320, 258), (332, 246), (340, 233), (346, 215), (349, 200), (349, 177),
                    (345, 157)]
    point_size = 2
    point_size1 = 2
    point_color = (0, 0, 255)
    thickness = 2
    for point in points_list1:
        cv2.circle(image, point, point_size, point_color, thickness)
    point_color1 = (0, 255, 0)
    for point in points_list2:
        cv2.circle(image, point, point_size1, point_color1, thickness)
    cv2.namedWindow("image")
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()