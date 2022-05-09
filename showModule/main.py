import cv2
import matplotlib.pyplot as plt
import os
import random

def get_defect_names(Dir_path, Images_Dir='IMAGES'):
    """
    return a list, as following,
        ['crazing',
         'inclusion',
         'patches',
         'pitted_surface',
         'rolled-in_scale',
         'scratches' ]
    """
    DEF_NAMES = []
    piclist = os.listdir(os.path.join(Dir_path, Images_Dir))
    for pic in piclist:
        picname = pic[0:pic.rfind("_")]
        if picname not in DEF_NAMES:
            DEF_NAMES.append(picname)

    return DEF_NAMES


def show_labels_img(img_id, img_Dir='datas/NEU-DET/IMAGES'):
    """imgname是输入图像的名称"""
    img = cv2.imread(img_Dir + '/' + img_id + ".jpg")  # 读取图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像格式
    h, w = img.shape[:2]  # 获取图像高宽
    print(w, h)
    label = []
    with open(f_Dir + "/labels/" + img_id + ".txt", 'r') as flabel:  # 读取labels文件中的真实框信息
        for label in flabel:
            label = label.split(' ')
            label = [float(x.strip()) for x in label]  # 读取后的信息为[11,0.332,0.588,0.12,0.09866666666666667] 都是归一化后的数值
            print(DEF_NAMES[int(label[0])])  # 获取类别信息
            pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))  # 根据x,y,w,h，转换得到真实框的左上角点
            pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))  # 根据x,y,w,h，转换得到真实框的右下角点
            cv2.putText(img, DEF_NAMES[int(label[0])], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
    return img


f_Dir = 'datas/NEU-DET'
DEF_NAMES = get_defect_names(f_Dir)

if __name__ == "__main__":

    flag = input("请输入显示模式：\n 0 : 原始数据访问 \n 1 : 结果访问 \n")
    if flag not in ("0", "1"):
        print("参数错误")
        exit(-1)
    fig = plt.figure()
    plt.ion()
    plt.show()
    dirList = os.listdir(f_Dir + "/IMAGES")
    random.shuffle(dirList)
    for dir in dirList:
        print(dir.split("."))
        if flag == "0":
            img = cv2.imread(f_Dir + "/IMAGES/" + dir)
        else:
            img = show_labels_img(dir.split(".")[0])
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        plt.imshow(img)
        plt.draw()
        plt.pause(1)
