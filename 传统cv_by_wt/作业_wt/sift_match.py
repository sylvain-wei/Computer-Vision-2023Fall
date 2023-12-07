import numpy as np
import os
import cv2
import argparse
import matplotlib.pyplot as plt 

parse = argparse.ArgumentParser(description='大作业-图像SIFT匹配')
parse.add_argument('--img_paths', nargs='?', default=['./left.png', './right.png'], help='paths of left/right images')
parse.add_argument('-k', type=int, default=2, help='kNN: k neighbours')
parse.add_argument('--ratio', type=float, default=0.75, help='ratio for match comparison')
parse.add_argument('--reprojThresh', type=float, default=0.75, help='reprojThresh')

args = parse.parse_args()


# load图像
def load_image(img_path):
    img = cv2.imread(img_path)
    return img

# 通过cv2封装好的SIFT特征提取器，提取特征关键点位置和特征向量
def detect_sift(sift, img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 转为灰度图
    keyPointer, descriptor = sift.detectAndCompute(gray_img, None)
    keyPointer = np.float32([kp.pt for kp in keyPointer])
    return keyPointer, descriptor

# 获得匹配出来的点坐标、坐标变换矩阵等
def match(kps1, kps2, des1, des2, k=2, ratio=0.75, reprojThresh=4.0):
    bf = cv2.BFMatcher() # crossCheck为True时匹配条件更严格
    raw_matches = bf.knnMatch(des1, des2, k)
    matches = []
    for m in raw_matches:
        # 找到符合最近邻匹配要求的所有匹配对，并把他们的索引分别提取出来
        # 遍历raw_matches，如果第一对点的欧氏距离小于ratio倍第二对点的欧氏距离则说明第一对点匹配可靠，将这对点的索引号追加到matches
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            matches.append((m[0].queryIdx, m[0].trainIdx))
    kps1 = np.float32([kps1[m[0]] for m in matches])    # 提取出对应点的keypointer（图片坐标）
    kps2 = np.float32([kps2[m[1]] for m in matches])    # 同上
    # 求解转换矩阵
    # 求解方法：cv2.RANSAC
    # reprojThresh是将点对视为内点的最大允许重投影错误阈值（仅用于RANSAC和RHO方法）
    # 返回的M为变换矩阵
    # status能够指示是否该点有匹配解
    (M, status) = cv2.findHomography(kps1, kps2, cv2.RANSAC, reprojThresh)  
    return (M, matches, status)

def stich(img1, img2, M):
    result = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0: img1.shape[0], 0: img2.shape[1]] = img2
    return result


if __name__ == '__main__':
    # 导入图像
    image_left = load_image(args.img_paths[0])
    image_right = load_image(args.img_paths[1])

    # 创建SIFT特征提取对象
    sift = cv2.SIFT_create()    

    # 获取关键点坐标、特征描述
    keyPointer_left, descriptor_left = detect_sift(sift, image_left)
    keyPointer_right, descriptor_right = detect_sift(sift, image_right)


    # 特征点匹配
    (M, matches, status) = match(keyPointer_right, keyPointer_left, descriptor_right, descriptor_left,  args.k, args.ratio, args.reprojThresh)

    # 拼接
    result = stich(image_right, image_left, M)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # 最后显示效果
    plt.imshow(result)
    plt.show()