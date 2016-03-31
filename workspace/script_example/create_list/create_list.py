proj_dir = "workspace/train_bird"
dataset_dir = "/media/htc/work/dataset/CUB_200_2011/CUB_200_2011"
image_dir = dataset_dir + "/images"
image_names_file = dataset_dir + "/images.txt"
split_file = dataset_dir + "/train_test_split.txt"
train_file = proj_dir + "/train_list.txt"
test_file = proj_dir + "/test_list.txt"

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
	save_info = "/" + image_name + " " + str(image_label) + "\n"
	if image_status == 1:
		train_list.writelines(save_info)
	else:
		test_list.writelines(save_info)

image_names.close()
split.close()
train_list.close()
test_list.close()
