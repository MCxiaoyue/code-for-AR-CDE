import cv2
from scipy.stats import pearsonr
import numpy as np
import os

def check_color_space(image, expected_color_space='BGR'):
    """检查图像的颜色空间是否符合预期。"""
    channels = len(image.shape)
    if channels == 3:
        if image.shape[2] == 3 and expected_color_space.lower() == 'bgr':
            return True
        elif image.shape[2] == 4 and expected_color_space.lower() == 'bgra':
            return True
    elif channels == 2 and expected_color_space.lower() == 'gray':
        return True
    else:
        return False

def calculate_pcc(image1, image2):
    """计算两幅图像的 Pearson 相关系数。"""
    flat_img1 = image1.flatten()
    flat_img2 = image2.flatten()
    pcc, _ = pearsonr(flat_img1, flat_img2)
    return pcc

def adjust_brightness(src_img, target_img):
    """根据目标图片调整源图片的亮度"""
    # 将图像转换为浮点类型以避免溢出
    src_img_float = src_img.astype(np.float32)
    target_img_float = target_img.astype(np.float32)

    src_mean = cv2.mean(src_img_float)[0]
    target_mean = cv2.mean(target_img_float)[0]
    diff = target_mean - src_mean

    # 调整亮度
    dst_img_float = cv2.add(src_img_float, np.ones_like(src_img_float) * diff)

    # 将结果转换回原来的类型
    dst_img = dst_img_float.clip(0, 255).astype(np.uint8)

    return dst_img

def duibi(img_path1, img_path2, i):

    # 从路径加载图像
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # img1 = cv2.resize(img1, (256, 256))
    # 在调用duibi函数之前先对img2进行亮度调整
    img2 = adjust_brightness(img2, img1)

    # 确保图像被正确读取
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both of the image files could not be found.")

    # 检查图像是否为BGR格式
    if not (check_color_space(img1, 'BGR') and check_color_space(img2, 'BGR')):
        raise ValueError("One or both images are not in BGR format.")

    # 图像预处理
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    pcc_score = calculate_pcc(gray1, gray2)
    print(pcc_score)

    print("Final------")
    print(i)

    return pcc_score


flag = 0
sum = 0

# 存储所有PCC值的列表
pcc_values = []

subject = '19'

for i in range(0, 10):
    # 示例用法
    ssim_score = duibi(
        "./"+subject+"/Recon_"+str(i)+".png",
        "./"+subject+"/Recon_"+str(i)+"_.png",
        i)
    pcc_values.append(ssim_score)
    flag += ssim_score

average_pcc = flag/10
print(f"Average PCC Value: {average_pcc}")
# 使用numpy计算标准差
std_dev = np.std(pcc_values)
print(f"Standard Deviation of PCC Values: {std_dev}")