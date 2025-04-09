import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os


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
    # print(img_path1)
    # print(img_path2)
    # 在调用duibi函数之前先对img2进行亮度调整
    img2 = adjust_brightness(img2, img1)
    # 如果提供了保存路径和文件名，则保存调整后的图像

    if save_path and img_path2:
        save_file_path = os.path.join(save_path, img_path2.split('/')[2].split('.')[0]+str("__")+".png")
        print(save_file_path)
        cv2.imwrite(save_file_path, img2)

    # img1 = cv2.resize(img1, (256, 256))

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

    print(i)

    return ssim_score



flag = 0

# 存储所有MCD值的列表
ssim_values = []

for i in range(0, 6):
    # 示例用法
    ssim_score = duibi(
        "./results/Recon_"+str(i)+".png",
        "./results/Recon_"+str(i)+"_.png",
        i,
        "./results/comparison_results")
    ssim_values.append(ssim_score)
    flag += ssim_score

average_mcd = flag/6
print(f"Average SSIM Value: {average_mcd}")
# 使用numpy计算标准差
std_dev = np.std(ssim_values)
print(f"Standard Deviation of SSIM Values: {std_dev}")




