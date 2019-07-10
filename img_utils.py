
import numpy as np
from skimage.transform import rescale

np.seterr(divide='ignore', invalid='ignore')

def mls_affine_deformation_1pt(p, q, v, alpha=1):
    ''' Calculate the affine deformation of one point.   
    This function is used to test the algorithm.
    '''
    '''计算一点的仿射变形。
      该函数用于测试算法。
    '''
    ctrls = p.shape[0]
    np.seterr(divide='ignore')
    w = 1.0 / np.sum((p - v) ** 2, axis=1) ** alpha
    w[w == np.inf] = 2**31-1
    pstar = np.sum(p.T * w, axis=1) / np.sum(w)
    qstar = np.sum(q.T * w, axis=1) / np.sum(w)
    phat = p - pstar
    qhat = q - qstar
    reshaped_phat1 = phat.reshape(ctrls, 2, 1)
    reshaped_phat2 = phat.reshape(ctrls, 1, 2)
    reshaped_w = w.reshape(ctrls, 1, 1)
    pTwp = np.sum(reshaped_phat1 * reshaped_w * reshaped_phat2, axis=0)
    try:
        inv_pTwp = np.linalg.inv(pTwp)
    except np.linalg.linalg.LinAlgError:
        if np.linalg.det(pTwp) < 1e-8:
            new_v = v + qstar - pstar
            return new_v
        else:
            raise
    mul_left = v - pstar
    mul_right = np.sum(reshaped_phat1 * reshaped_w * qhat[:, np.newaxis, :], axis=0)
    new_v = np.dot(np.dot(mul_left, inv_pTwp), mul_right) + qstar
    return new_v

def mls_affine_deformation(image, p, q, alpha=1.0, density=1.0):
    ''' Affine deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    '''仿射变形
       ###  参数：
             * image -  ndarray：原始图像
             * p  -  ndarray：一个大小为[n，2]的数组，原始控制点
             * q  -  ndarray：一个大小为[n，2]的数组，最终控制点
             * alpha  -  float：权重使用的参数
             * density  -  float：网格的密度
         ###返回：
             变形的图像。
    '''
    height = image.shape[0]
    width = image.shape[1]
    #将（x，y）更改为（row，col）
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]
    # 在原始图像上制作网格
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # 网格行
    gcol = vx.shape[1]  # 网格列
    ctrls = p.shape[0]  # 控制点

    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    #print('the reshaped_v is\n ', reshaped_v)
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha   #计算每一个点到控制点p的距离    # [ctrls, grow, gcol]
    #print('the w is\n ', w)
    w[w == np.inf] = 2**31 - 1 #如果该点为控制顶点，那么需要把该控制顶点的权重设置为无穷大
    reshaped_pp= reshaped_p.transpose(1, 0, 2, 3)
    #print('the reshaped_pp is\n ', reshaped_pp)  # 关于p的四维数组
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    #print('the pstar is\n ', pstar)
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 2, 1, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    pTwp = np.sum(reshaped_phat1 * reshaped_w * reshaped_phat2, axis=0)                 # [2, 2, grow, gcol]
    #print('the pTwp is\n ', pTwp)
    inv_pTwp = np.linalg.inv(pTwp.transpose(2, 3, 0, 1))
    mul_left = reshaped_v - pstar                                                       # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)        # [grow, gcol, 1, 2]
    mul_right = reshaped_w * reshaped_phat1                                             # [ctrls, 2, 1, grow, gcol]
    reshaped_mul_right =mul_right.transpose(0, 3, 4, 1, 2)                              # [ctrls, grow, gcol, 2, 1]
    A = np.matmul(np.matmul(reshaped_mul_left, inv_pTwp), reshaped_mul_right)           # [ctrls, grow, gcol, 1, 1]
    reshaped_A = A.reshape(ctrls, 1, grow, gcol)                                        # [ctrls, 1, grow, gcol]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]

    # Get final image transfomer -- 3-D array获得最终图像变换器 -  3D阵列
    transformers = np.sum(reshaped_A * qhat, axis=0) + qstar                            # [2, grow, gcol]

    # Removed the points outside the border 删除了边界外的点
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image 映射原始图像
    transformed_image = np.ones_like(image) * 255 #三原色都为255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16), 
                                        (np.arange(grow) / density).astype(np.int16))
    transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]    # [grow, gcol]
    return transformed_image

def mls_affine_deformation_inv(image, p, q, alpha=1.0, density=1.0):
    '''仿射逆变形
         参数：###
             * image -  ndarray：原始图像
             * p  -  ndarray：一个大小为[n，2]的数组，原始控制点
             * q  -  ndarray：一个大小为[n，2]的数组，最终控制点
             * alpha  -  float：权重使用的参数
             * density - 浮点数：网格的密度
         ###返回：
             变形的图像。
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    w[w == np.inf] = 2**31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]

    reshaped_phat = phat.reshape(ctrls, 2, 1, grow, gcol)                               # [ctrls, 2, 1, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    pTwq = np.sum(reshaped_phat * reshaped_w * reshaped_qhat, axis=0)                   # [2, 2, grow, gcol]
    inv_pTwq = np.linalg.inv(pTwq.transpose(2, 3, 0, 1))                            # [grow, gcol, 2, 2]
    mul_left = reshaped_v - qstar                                                       # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)        # [grow, gcol, 1, 2]
    mul_right = np.sum(reshaped_phat * reshaped_w * reshaped_phat2, axis=0)             # [2, 2, grow, gcol]
    reshaped_mul_right =mul_right.transpose(2, 3, 0, 1)                                 # [grow, gcol, 2, 2]
    temp = np.matmul(np.matmul(reshaped_mul_left, inv_pTwq), reshaped_mul_right)        # [grow, gcol, 1, 2]
    reshaped_temp = temp.reshape(grow, gcol, 2).transpose(2, 0, 1)                      # [2, grow, gcol]

    # Get final image transfomer -- 3-D array
    transformers = reshaped_temp + pstar                                                # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = image[tuple(transformers.astype(np.int16))]    # [grow, gcol]

    # Rescale image
    transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')

    return transformed_image



def mls_similarity_deformation(image, p, q, alpha=1.0, density=1.0):
    '''相似变形
         参数：###
             *图像 -  ndarray：原始图像
             * p  -  ndarray：一个大小为[n，2]的数组，原始控制点
             * q  -  ndarray：一个大小为[n，2]的数组，最终控制点
             * alpha  -  float：权重使用的参数
             *密度 - 浮点数：网格的密度
         ###返回：
             变形的图像。
     '''

    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    print('the w is\n', w)
    sum_w = np.sum(w, axis=0)                                                           # [grow, gcol]
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / sum_w                # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 1, 2, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 2, 1, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    print('the reshaped_w is\n', reshaped_w)
    mu = np.sum(np.matmul(reshaped_w.transpose(0, 3, 4, 1, 2) *
                          reshaped_phat1.transpose(0, 3, 4, 1, 2), 
                          reshaped_phat2.transpose(0, 3, 4, 1, 2)), axis=0)             # [grow, gcol, 1, 1] us的值
    reshaped_mu = mu.reshape(1, grow, gcol)                                             # [1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]                                
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
    mul_left = np.concatenate((reshaped_phat1, reshaped_neg_phat_verti), axis=1)        # [ctrls, 2, 2, grow, gcol]
    vpstar = reshaped_v - pstar                                                         # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)                                  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0],...]                                               # [2, grow, gcol]
    neg_vpstar_verti[1,...] = -neg_vpstar_verti[1,...]                                  
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)              # [2, 1, grow, gcol]
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)    # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.reshape(1, 2, 2, grow, gcol)                         # [1, 2, 2, grow, gcol]
    A = np.matmul((reshaped_w * mul_left).transpose(0, 3, 4, 1, 2), 
                       reshaped_mul_right.transpose(0, 3, 4, 1, 2))                     # [ctrls, grow, gcol, 2, 2]

     # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol).transpose(0, 3, 4, 1, 2)      # [ctrls, grow, gcol, 1, 2]

    # Get final image transfomer -- 3-D array
    temp = np.sum(np.matmul(reshaped_qhat, A), axis=0).transpose(2, 3, 0, 1)            # [1, 2, grow, gcol]
    reshaped_temp = temp.reshape(2, grow, gcol)                                         # [2, grow, gcol]
    transformers = reshaped_temp / reshaped_mu  + qstar                                 # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = np.ones_like(image) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16), 
                                        (np.arange(grow) / density).astype(np.int16))
    transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]    # [grow, gcol]

    return transformed_image


def mls_similarity_deformation_inv(image, p, q, alpha=1.0, density=1.0):
    ''' Similarity inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    w[w == np.inf] = 2**31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 1, 2, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 2, 1, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]

    mu = np.sum(np.matmul(reshaped_w.transpose(0, 3, 4, 1, 2) * 
                          reshaped_phat1.transpose(0, 3, 4, 1, 2), 
                          reshaped_phat2.transpose(0, 3, 4, 1, 2)), axis=0)             # [grow, gcol, 1, 1]
    reshaped_mu = mu.reshape(1, grow, gcol)                                             # [1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]                                
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
    mul_right = np.concatenate((reshaped_phat1, reshaped_neg_phat_verti), axis=1)       # [ctrls, 2, 2, grow, gcol]
    mul_left = reshaped_qhat * reshaped_w                                               # [ctrls, 1, 2, grow, gcol]
    Delta = np.sum(np.matmul(mul_left.transpose(0, 3, 4, 1, 2), 
                             mul_right.transpose(0, 3, 4, 1, 2)), 
                   axis=0).transpose(0, 1, 3, 2)                                        # [grow, gcol, 2, 1]
    Delta_verti = Delta[...,[1, 0],:]                                                   # [grow, gcol, 2, 1]
    Delta_verti[...,0,:] = -Delta_verti[...,0,:]
    B = np.concatenate((Delta, Delta_verti), axis=3)                                    # [grow, gcol, 2, 2]
    inv_B = np.linalg.inv(B)                                                        # [grow, gcol, 2, 2]


    v_minus_qstar_mul_mu = (reshaped_v - qstar) * reshaped_mu                           # [2, grow, gcol]
    
    # Get final image transfomer -- 3-D array
    reshaped_v_minus_qstar_mul_mu = v_minus_qstar_mul_mu.reshape(1, 2, grow, gcol)      # [1, 2, grow, gcol]
    transformers = np.matmul(reshaped_v_minus_qstar_mul_mu.transpose(2, 3, 0, 1),
                            inv_B).reshape(grow, gcol, 2).transpose(2, 0, 1) + pstar    # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = image[tuple(transformers.astype(np.int16))]    # [grow, gcol]

    # Rescale image
    transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')

    return transformed_image


def mls_rigid_deformation(image, p, q, alpha=1.0, density=1.0):
    '''
    刚性变形
         参数：###
             * image -  ndarray：原始图像
             * p  -  ndarray：一个大小为[n，2]的数组，原始控制点
             * q  -  ndarray：一个大小为[n，2]的数组，最终控制点
             * alpha  -  float：权重使用的参数
             * density - float：网格的密度
         ###返回：
             变形的图像。
     '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    sum_w = np.sum(w, axis=0)                                                           # [grow, gcol]
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / sum_w                # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat = phat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]                                
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
    mul_left = np.concatenate((reshaped_phat, reshaped_neg_phat_verti), axis=1)         # [ctrls, 2, 2, grow, gcol]
    vpstar = reshaped_v - pstar                                                         # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)                                  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0],...]                                               # [2, grow, gcol]
    neg_vpstar_verti[1,...] = -neg_vpstar_verti[1,...]                                  
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)              # [2, 1, grow, gcol]
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)    # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.reshape(1, 2, 2, grow, gcol)                         # [1, 2, 2, grow, gcol]
    A = np.matmul((reshaped_w * mul_left).transpose(0, 3, 4, 1, 2), 
                       reshaped_mul_right.transpose(0, 3, 4, 1, 2))                     # [ctrls, grow, gcol, 2, 2]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [2, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol).transpose(0, 3, 4, 1, 2)      # [ctrls, grow, gcol, 1, 2]

    # Get final image transfomer -- 3-D array
    temp = np.sum(np.matmul(reshaped_qhat, A), axis=0).transpose(2, 3, 0, 1)            # [1, 2, grow, gcol]
    reshaped_temp = temp.reshape(2, grow, gcol)                                         # [2, grow, gcol]
    norm_reshaped_temp = np.linalg.norm(reshaped_temp, axis=0, keepdims=True)           # [1, grow, gcol]
    norm_vpstar = np.linalg.norm(vpstar, axis=0, keepdims=True)                         # [1, grow, gcol]
    transformers = reshaped_temp / norm_reshaped_temp * norm_vpstar  + qstar            # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0
    
    # Mapping original image
    transformed_image = np.ones_like(image) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16), 
                                        (np.arange(grow) / density).astype(np.int16))
    transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]    # [grow, gcol]
    
    return transformed_image

def mls_rigid_deformation_inv(image, p, q, alpha=1.0, density=1.0):
    ''' Rigid inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    '''刚性逆变形
         参数：###
             * image -  ndarray：原始图像
             * p  -  ndarray：一个大小为[n，2]的数组，原始控制点
             * q  -  ndarray：一个大小为[n，2]的数组，最终控制点
             * alpha  -  float：权重使用的参数
             * destity - 浮点数：网格的密度
         ###返回：
             变形的图像。
    '''
    height = image.shape[0]
    #print('the height is\n ', height)
    width = image.shape[1]
    #print('the width  is\n ', width )
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
   # print('the q is\n ', q)
    p = p[:, [1, 0]]
   # print('the p is\n ', p)

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows .shape读取网格行数字
    #print('the grow is\n ', grow)
    gcol = vx.shape[1]  # grid cols #.shape读取网格列数字
    #print('the gcol is\n ', gcol)
    ctrls = p.shape[0]  # control points #.shape读取控制点数量
   # print('the ctrls is\n ', ctrls)

    # Compute #计算
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1] #变成四维数组
    print('the reshaped_p is\n ', reshaped_p)
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    print('the reshaped_v is\n ', reshaped_v)
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    print('the w is\n ', w)
    w[w == np.inf] = 2**31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 1, 2, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 2, 1, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]

    neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]                                
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
    mul_right = np.concatenate((reshaped_phat1, reshaped_neg_phat_verti), axis=1)       # [ctrls, 2, 2, grow, gcol]
    mul_left = reshaped_qhat * reshaped_w                                               # [ctrls, 1, 2, grow, gcol]
    Delta = np.sum(np.matmul(mul_left.transpose(0, 3, 4, 1, 2), 
                             mul_right.transpose(0, 3, 4, 1, 2)), 
                   axis=0).transpose(0, 1, 3, 2)                                        # [grow, gcol, 2, 1]
    Delta_verti = Delta[...,[1, 0],:]                                                   # [grow, gcol, 2, 1]
    Delta_verti[...,0,:] = -Delta_verti[...,0,:]
    B = np.concatenate((Delta, Delta_verti), axis=3)                                    # [grow, gcol, 2, 2]
    inv_B = np.linalg.inv(B)                                                        # [grow, gcol, 2, 2]
    vqstar = reshaped_v - qstar                                                         # [2, grow, gcol]
    reshaped_vqstar = vqstar.reshape(1, 2, grow, gcol)                                  # [1, 2, grow, gcol]

    # Get final image transfomer -- 3-D array
    temp = np.matmul(reshaped_vqstar.transpose(2, 3, 0, 1),
                     inv_B).reshape(grow, gcol, 2).transpose(2, 0, 1)                   # [2, grow, gcol]
    norm_temp = np.linalg.norm(temp, axis=0, keepdims=True)                             # [1, grow, gcol]
    norm_vqstar = np.linalg.norm(vqstar, axis=0, keepdims=True)                         # [1, grow, gcol]
    transformers = temp / norm_temp * norm_vqstar + pstar                               # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = image[tuple(transformers.astype(np.int16))]    # [grow, gcol]

    # Rescale image
    transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')

    return transformed_image
