#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares

@author: Jarvis ZHANG
@date: 2017/8/8
@editor: VS Code
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from img_utils import (mls_affine_deformation, mls_affine_deformation_inv,
                       mls_similarity_deformation, mls_similarity_deformation_inv,
                       mls_rigid_deformation, mls_rigid_deformation_inv)


def show_example():
    img = plt.imread(os.path.join(sys.path[0], "double.jpg"))
    plt.imshow(img)
    plt.show()

def demo(fun, fun_inv, name):
    '''
    p = np.array([
        [30, 155], [125, 155], [225, 155],
        [100, 235], [160, 235], [85, 295], [180, 293]
    ])
    q = np.array([
        [42, 211], [125, 155], [235, 100],
        [80, 235], [140, 235], [85, 295], [180, 295]
    ])
    '''
    p = np.array([
        [186, 214], [196, 232], [207, 248],[222, 264], [240, 276], [262, 283], [286, 285],
        [307, 280], [325, 270], [338, 255], [349, 238], [354, 218], [353, 197], [349, 177],
        [345, 157], [216, 172], [230, 179], [287, 147], [303, 152], [286, 211]
    ])
    q = np.array([
        [186, 214], [198, 232], [212, 245], [228, 260], [244, 268], [264, 275], [285, 274],
        [301, 269], [320, 258], [332, 246], [340, 233], [346, 215], [349, 200], [349, 177],
        [345, 157], [216, 172], [230, 179], [287, 147], [303, 152], [286, 211]
    ])
    image = plt.imread(os.path.join(sys.path[0], "shou.jpg"))

    plt.figure(figsize=(8, 6))
    plt.subplot(131)
    plt.axis('off')
    plt.imshow(image)
    plt.title("Original Image")
    if fun is not None:
        transformed_image = fun(image, p, q, alpha=1, density=1)
        plt.subplot(133)
        plt.axis('off')
        plt.imshow(transformed_image)
        plt.title(" %s Deformation " % name)
        '''
        transformed_image = fun(image, p, q, alpha=1, density=0.7)
        plt.subplot(235)
        plt.axis('off')
        plt.imshow(transformed_image)
        plt.title("%s Deformation \n Sampling density 0.7"%name)
        '''
    if fun_inv is not None:
        transformed_image = fun_inv(image, p, q, alpha=1, density=1)
        plt.subplot(132)
        plt.axis('off')
        plt.imshow(transformed_image)
        plt.title("Inverse %s Deformation "%name)
        '''
        transformed_image = fun_inv(image, p, q, alpha=1, density=0.7)
        plt.subplot(236)
        plt.axis('off')
        plt.imshow(transformed_image)
        plt.title("Inverse %s Deformation \n Sampling density  0.7"%name)
        '''
    plt.tight_layout(w_pad=0.1)
    plt.show()

def demo2(fun):
    ''' 
        Smiled Monalisa  
    '''

    p = np.array([
        [186, 140], [295, 135], [208, 181], [261, 181], [184, 203], [304, 202], [213, 225], 
        [243, 225], [211, 244], [253, 244], [195, 254], [232, 281], [285, 252]
    ])
    q = np.array([
        [186, 140], [295, 135], [208, 181], [261, 181], [184, 203], [304, 202], [213, 225], 
        [243, 225], [211, 244], [253, 244], [202, 252], [232, 281], [276, 250]
    ])
    image = plt.imread(os.path.join(sys.path[0], "monalisa.jpg"))
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(image)
    transformed_image = fun(image, p, q, alpha=1, density=1)
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(transformed_image)
    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    #transformed_image.save("monalisa_2.jpg")
    #plt.savefig("monalisa_2.jpg")
    plt.imsave('monalisa_2.jpg', transformed_image)
    plt.show()


if __name__ == "__main__":
    #affine deformation
    demo(mls_affine_deformation, mls_affine_deformation_inv, "Affine")
    #demo2(mls_affine_deformation_inv)

    #similarity deformation
    #demo(mls_similarity_deformation, mls_similarity_deformation_inv, "Similarity")
    #demo2(mls_similarity_deformation_inv)

    #rigid deformation
    #demo(mls_rigid_deformation, mls_rigid_deformation_inv, "Rigid")
    #demo2(mls_rigid_deformation_inv)
    #demo2(mls_similarity_deformation)
