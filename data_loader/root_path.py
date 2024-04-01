import os


# 返回元组[文件所在文件夹路径，文件名]
def root_spliter(mode: str, root: str) -> list[str, str, str]:
    folder_file: tuple[str, str] = os.path.split(root)
    folder = "NULL"
    if mode == 's':
        folder = R'Images/Single/'
    elif mode == 'b':
        folder = R'Images/Batch/raw/'
    file: list[str, str] = (folder_file[1]).split('.')
    filename: str = file[0]
    fileend: str = '.' + file[1]
    return list((folder, filename, fileend))


# 返回文件夹下所有文件名
def traversal(filepath) -> list[str]:
    return os.listdir(filepath)
