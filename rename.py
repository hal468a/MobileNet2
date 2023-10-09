import os

# 指定你想要更改文件名的目录
directory = "Recycle"

i = 1
# 遍历目录中的每个文件
for filename in os.listdir(directory):
    # 确保只处理.txt文件
    if filename.endswith(".jpg"):
        # 创建新的文件名
        new_filename = f"recycle{i}.jpg"
        # 使用os.rename()方法重命名文件
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

        i = i + 1