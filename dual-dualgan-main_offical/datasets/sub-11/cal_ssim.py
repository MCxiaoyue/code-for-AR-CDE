import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
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

def duibi(img_path1, img_path2, i, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 从路径加载图像
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # img1 = cv2.resize(img1, (256, 256))

    print(img1)
    print(img2)

    # 确保图像被正确读取
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both of the image files could not be found.")

        # 检查图像是否为BGR格式
    if not (check_color_space(img1, 'BGR') and check_color_space(img2, 'BGR')):
        raise ValueError("One or both images are not in BGR format.")

    # 图像预处理（如果需要与原代码功能一致的话）
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    ssim_score = ssim(gray1, gray2, data_range=gray2.max() - gray2.min())

    # 设置图像显示
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.title('Original')

    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.title('Predicted')

    # 计算像素差并显示
    pixel_diff = cv2.absdiff(img1, img2)
    plt.subplot(1, 3, 3)
    plt.imshow(pixel_diff, cmap='gray')
    plt.title(f'Pixel Difference\nSSIM Score: {ssim_score:.5f}')

    # 保存比较结果
    plt.savefig(os.path.join(save_path, f"{i}.jpg"))

    print(ssim_score)

    print("Final------")



    return ssim_score


    # 可根据需要取消注释以显示图像
    # plt.show()

flag = 0
sum = 0

# 存储所有MCD值的列表
ssim_values = []

for i in range(1, 80, 10):
    # 示例用法
    ssim_score = duibi(
        "E:\\second_paper\\dual-dualgan-main_offical\\datasets\\sub-11\\test\\C\\C_"+str(i)+".png",
        "E:\\second_paper\\dual-dualgan-main_offical\\test\\sub-11-img_sz_256-fltr_dim_64-L1-lambda_BC_1000.0_1000.0\\B\\B_" + str(
            i) + "_B2C.jpg",
        i,
        "./comparison_epoch50_results")
    ssim_values.append(ssim_score)
    sum += 1
    flag += ssim_score

average_mcd = flag/sum
print(f"Average SSIM Value: {average_mcd}")
# 使用numpy计算标准差
std_dev = np.std(ssim_values)
print(f"Standard Deviation of SSIM Values: {std_dev}")





