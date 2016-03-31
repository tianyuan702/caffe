import os
import cv2
from PIL import Image

proj_dir = "/mnt_data/data/yuan_tian/train_bird"
dataset_dir = "/mnt_data/data/yuan_tian/data/CUB_200_2011"
image_dir = dataset_dir + "/images"
image_names_file = dataset_dir + "/images.txt"
split_file = dataset_dir + "/train_test_split.txt"
train_file = proj_dir + "/train_list.txt"
test_file = proj_dir + "/test_list.txt"
crop_dir = proj_dir + "/crop_img"

if not os.path.isdir(crop_dir):
        os.makedirs(crop_dir)

image_names = open(image_names_file, "r")
split = open(split_file, "r")
train_list = open(train_file, "w")
test_list = open(test_file, "w")

image_names_list = []
for item in image_names.readlines():
        info = item.strip().split(" ")
        image_names_list.append(info[1])

for item_idx, item in enumerate(split.readlines()):
        info = item.strip().split(" ")
        image_id = info[0]
        image_status = int(info[1])
        image_name = image_names_list[item_idx]
        image_label = int(image_name.split(".")[0])

        # crop image
        image = cv2.imread(image_dir + "/" + image_name)
        image_height = image.shape[0]
        image_width = image.shape[1]
        start_x = 0
        start_y = 0
        if image_height > image_width:
                image_height = int(256. * image_height / image_width)
                image_width = 256
                start_y = (image_height - 256) / 2
        else:
                image_width = int(256. * image_width / image_height)
                image_height = 256
                start_x = (image_width - 256) / 2
        image_sub = image_name.split("/")[0]
        if not os.path.isdir(crop_dir + "/" + image_sub):
                os.makedirs(crop_dir + "/" + image_sub)
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
        image = image[start_y : start_y + 256, start_x : start_x + 256]
        cv2.imwrite(crop_dir + "/" + image_name, image)

        save_info = "/" + image_name + " " + str(image_label) + "\n"
        if image_status == 1:
                train_list.writelines(save_info)
        else:   
                test_list.writelines(save_info)

image_names.close()
split.close()
train_list.close()
test_list.close()
