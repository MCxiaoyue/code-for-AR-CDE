import cv2
import numpy as np
import os
import re


def crop_and_concat_images(image1_path, image2_path, output_path, ratio=3 / 5):
    # 读取两张输入图像
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    img1 = cv2.resize(img1, (256, 256))

    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both images could not be read. Please check the paths.")

    height, width, _ = img1.shape

    # 计算裁剪高度
    crop_height1 = int(height * ratio)
    crop_height2 = height - crop_height1

    # 裁剪图像
    cropped_img1 = img1[:crop_height1, :]
    cropped_img2 = img2[-crop_height2:, :]

    # 创建一个与原图等宽、高为两图裁剪高度之和的空白图像
    concatenated_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 将裁剪后的图像拼接到空白图像上
    concatenated_image[:crop_height1, :] = cropped_img1
    concatenated_image[crop_height1:, :] = cropped_img2

    # 保存拼接后的图像
    cv2.imwrite(output_path, concatenated_image)


# 设置相对路径
image_folder_A = "./A"
image_folder_C = "./C"
output_folder_B = "./B"

# 确保输出文件夹存在
if not os.path.exists(output_folder_B):
    os.makedirs(output_folder_B)

# 使用循环处理所有图像
for filename in os.listdir(image_folder_A):
    if filename.endswith(".png"):  # 假设我们只处理PNG文件
        match = re.search(r'A_(\d+)\.png$', filename)
        if match:
            number = match.group(1)
            corresponding_filename_C = f"C_{number}.png"
            corresponding_filepath_C = os.path.join(image_folder_C, corresponding_filename_C)

            if os.path.exists(corresponding_filepath_C):  # 检查对应文件是否存在
                # 构建完整的文件路径
                image1_path = os.path.join(image_folder_A, filename)
                image2_path = corresponding_filepath_C
                output_image_path = os.path.join(output_folder_B, f"B_{number}.png")

                # 调用函数进行图像裁剪和拼接
                crop_and_concat_images(image1_path, image2_path, output_image_path)
            else:
                print(f"Warning: No matching file found for {filename} in folder C.")