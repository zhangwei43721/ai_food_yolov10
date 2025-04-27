import os
from lxml import etree
import shutil

def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f'文件夹: {folder_path} 所有内容已删除!')
    except Exception as e:
        print(f'删除文件夹及其所有内容报错: {e}')

def xml_to_yolotxt(source_path, label_path, cls=''):
    if not os.path.exists(label_path):
        os.mkdir(label_path)
        # 获取xml文件名称列表
    files = os.listdir(source_path)
    classes = get_classes(files, source_path)
    print('---获取所有xml文件中的cls=', classes)
    # classes = ['fire']
    # 生成分类字典，列如{'apple':0,'banana':1}
    class_dict = dict(zip(classes,range(len(classes))))
    print('------生成分类字典=', class_dict)
    # class_dict = {'fire': 1}
    # class_dict = {cls: 0}
    return
    print('---------------begin convert----')
    count = 0
    if 1:
        for file in files:
            count = count + 1
            # print('------------', file)
            convert_xml2txt(file, source_path, label_path, class_dict, norm=True)
    print('---------------finish convert----,count=', count)


def convert_xml2txt(file_name, source_path, label_path, class_dict, norm=False):
    # 创建txt文件，并打开、写入
    new_name = file_name.split('.')[0] + '.txt'
    f = open(label_path+'/'+new_name,'w')
    with open(source_path+file_name,'rb') as fb:
        # 开始解析xml文件，获取图像尺寸
        xml = etree.HTML(fb.read())
        width = int(xml.xpath('//size/width/text()')[0])
        height = int(xml.xpath('//size/height/text()')[0])
        # 获取对象标签
        labels = xml.xpath('//object') # 单张图片中的目标数量 len(labels)
        for label in labels:
            name = label.xpath('./name/text()')[0]
            label_class = class_dict[name]
            xmin = int(label.xpath('./bndbox/xmin/text()')[0])
            xmax = int(label.xpath('./bndbox/xmax/text()')[0])
            ymin = int(label.xpath('./bndbox/ymin/text()')[0])
            ymax = int(label.xpath('./bndbox/ymax/text()')[0])
 
            # xyxy-->xywh,且归一化
            if norm :
                dw = 1 / width
                dh = 1 / height
                x_center = (xmin + xmax) / 2
                y_center = (ymax + ymin) / 2
                w = (xmax - xmin)
                h = (ymax - ymin)
                x, y, w, h = x_center * dw, y_center * dh, w * dw, h * dh
                f.write(str(label_class)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+' '+'\n')
    #关闭文件
    f.close()
    
# 获取分类名称列表
def get_classes(files, source_path):
    class_set = set([])
    for file in files:
        with open(source_path+file,'rb') as fb:
            #解析xml文件
            xml = etree.HTML(fb.read())
            labels = xml.xpath('//object')
            for label in labels:
                name = label.xpath('./name/text()')[0] 
                class_set.add(name)
    return list(class_set)


if __name__ == '__main__':
    source_path = 'C:/myself/hw_ai/ai_datasets/菜品目标检测数据集2600+/images/xmls/'
    label_path = 'C:/myself/hw_ai/ai_datasets/菜品目标检测数据集2600+/images/labels/'
    xml_to_yolotxt(source_path, label_path)