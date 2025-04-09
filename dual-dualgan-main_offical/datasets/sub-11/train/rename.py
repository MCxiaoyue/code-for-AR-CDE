import os

# 设置工作目录为当前目录下的B文件夹
work_dir = './B'

# 检查目录是否存在
if not os.path.exists(work_dir):
    print(f"目录 {work_dir} 不存在")
else:
    # 切换到工作目录
    os.chdir(work_dir)

    # 遍历工作目录中的文件
    for filename in os.listdir('./'):
        # 检查文件是否是以'B'开头并且是'.png'结尾
        if filename.startswith("B") and filename.endswith(".png"):
            # 创建新的文件名
            new_filename = "C" + filename[1:]
            # 重命名文件
            os.rename(filename, new_filename)
    print("重命名完成")