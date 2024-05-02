import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

def main():
    root = os.getcwd()
    data_dir = 'data/GolfBall/Golf_Ball/Detection/Annotations'
    data_csv_dir = 'data/GolfBall/Golf_Ball/Detection/AnnotationsCsv'
    if not os.path.exists(data_csv_dir):
    # Nếu thư mục chưa tồn tại, tạo nó
        os.makedirs(data_csv_dir)
        print(f"Thư mục '{data_csv_dir}' đã được tạo.")

    video_names = []
    for i in os.listdir(data_dir):
        if i.endswith("xml"):
            video_names.append(i.split("_")[0])
    video_names = np.unique(video_names)
    print(video_names)
    for video_name in video_names:
        print("********")
        csv = f"{video_name}.csv"
        xmls = []
        Frame = []
        Visibility = []
        X = []
        Y = []
        for xml in os.listdir(data_dir):
            if not xml.endswith("xml"):
                continue
            if xml.startswith(video_name):
                xmls.append(xml)
        sorted_xmls = sorted(xmls)

        for i, xml in enumerate(sorted_xmls):
            Frame.append(i)
            tree = ET.parse(os.path.join(data_dir, xml))
            root = tree.getroot()
            if root.findall('object'):
                for object in root.findall('object'):
                    bndbox = object.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                Visibility.append(1)
                X.append(x_center)
                Y.append(y_center)
            else:
                Visibility.append(0)
                X.append(0)
                Y.append(0)
        data = {
                'Frame': Frame,
                'Visibility': Visibility,
                'X': X,
                'Y' : Y
            }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(data_csv_dir, csv), index=False)
    return 
if __name__ == "__main__":
    main()