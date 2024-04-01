import tkinter as tk
from tkinter import filedialog


def select_folder_path():
    selected_folder_path = None
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory()
    if folder_path:
        selected_folder_path = folder_path
        root.withdraw()
        root.destroy()

    return selected_folder_path


def select_file_path():
    selected_file_path = None
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    if file_path:
        selected_file_path = file_path
        root.destroy()

    return selected_file_path


if __name__ == '__main__':
    path = select_folder_path()
    print(path)
    path = select_file_path()
    print(path)
