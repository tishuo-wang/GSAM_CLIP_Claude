"""
控制台版本

"""
import os
import pyperclip
import sys
import ctypes

inputWord = ""
DIC = {}  # 数字编号 int ：文件名 str


def m1(arr, string) -> bool:
    """
    检测输入的关键词列表是否和 目标字符串匹配
    对于arr中的每一个小字符串，检测他们是否都是string的子串
    """
    return all(content in string for content in arr)


def refreshDic():
    """刷新操作，此操作影响全局变量dic"""
    # 初始化 dic
    DIC.clear()
    for i, fileName in enumerate(os.listdir("myData")):
        DIC[i] = fileName


def showList():
    """展示列表"""
    for i, fileName in enumerate(os.listdir("myData")):
        print(str(i).zfill(3), fileName)


def getContentToClipboard(fileName):
    """
    根据文件名，把文件名复制到粘贴板
    fileName: "xxx.txt"
    """
    with open(f"myData/{fileName}", encoding="utf-8") as f:
        content = f.read()
    pyperclip.copy(content)


def changeColor(color: int):
    """更改打印颜色"""
    std_out_handle = ctypes.windll.kernel32.GetStdHandle(-11)
    return ctypes.windll.kernel32.SetConsoleTextAttribute(std_out_handle, color)


def main():
    global inputWord
    refreshDic()
    showList()
    print("======")
    while True:
        # os.system("cls")
        print("请输入搜索关键字，空格隔开，如果只输入数字表示选定")
        inputWord = input(">>>")
        if inputWord.isdigit():
            # 纯数字，表示选定了
            inputNum = int(inputWord)
            if inputNum in DIC:
                getContentToClipboard(DIC[inputNum])
                print("内容已经进入您的粘贴板！")
            else:
                print("您输入的数字超出范围")
        elif inputWord == "open":
            os.system(f'start {os.path.abspath("myData")}')
        elif inputWord == "refresh":
            refreshDic()
            changeColor(10)
            sys.stdout.write("已经刷新\n")
            changeColor(15)
        elif inputWord == "help":
            print("open  打开存放数据的文件夹")
            print("help  查看帮助")
            print("数字  选定某个标号的代码片段，复制到粘贴板")
            print("refresh  刷新数据文件，当你更改了数据文件夹里的文件时候用")
        else:
            # 是正常的检索
            isFind = False
            for i, fileName in enumerate(os.listdir("myData")):
                if m1(inputWord.split(), fileName):
                    isFind = True
                    changeColor(10)
                    sys.stdout.write(str(i).zfill(3) + " " + fileName + '\n')
            changeColor(15)
            if not isFind:
                changeColor(12)
                sys.stdout.write("没有搜索到相关内容" + '\n')
                changeColor(15)


if __name__ == "__main__":
    main()
