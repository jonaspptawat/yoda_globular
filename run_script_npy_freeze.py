import os

imgsize = 1024
epochs = 500
batch_size = 2
data_ = "../running_data/npy_dataset/data.yaml"
# data_ = './data/data_npy.yaml'
hpy_aug = "./data/hyps/hyp.with-mosaic.yaml"
hpy_no_aug = "./data/hyps/hyp.no-mosaic.yaml"
device = 0


'''
No Augmentations
'''

project_name = "./runs/train/freeze/no_augment"

base_path = "./runs/train/base/no_augment/"

pt_5n_3_noaug = os.path.join(base_path, './5n_3/weights/best.pt')
pt_5s_3_noaug = os.path.join(base_path, './5s_3/weights/best.pt')

# 3x3 and 1x1
os.system("python train_freeze.py --img {} \
--epochs {} --batch-size {} \
--data {} \
--hyp {} --cfg yolov5n.yaml \
--device {} --exist-ok --patience 0 \
--name 5n_3 --project {} --weights {}".format(imgsize, epochs, batch_size, data_, hpy_no_aug, device, project_name, pt_5n_3_noaug))

os.system("python train_freeze.py --img {} \
--epochs {} --batch-size {} \
--data {} \
--hyp {} --cfg yolov5s.yaml \
--device {} --exist-ok --patience 0 \
--name 5s_3 --project {} --weights {}".format(imgsize, epochs, batch_size, data_, hpy_no_aug, device, project_name, pt_5s_3_noaug))




imgsize = 1024
epochs = 300
batch_size = 2
data_ = "../running_data/npy_dataset/data.yaml"
# data_ = './data/data_npy.yaml'
hpy_aug = "./data/hyps/hyp.with-mosaic.yaml"
hpy_no_aug = "./data/hyps/hyp.no-mosaic.yaml"
device = 0

'''
With Augmentations
'''

project_name = "./runs/train/freeze/augment"

base_path = "./runs/train/base/augment/"

pt_5n_3_aug = os.path.join(base_path, './5n_3/weights/best.pt')
pt_5s_3_aug = os.path.join(base_path, './5s_3/weights/best.pt')

# 3x3 and 1x1
os.system("python train_freeze.py --img {} \
--epochs {} --batch-size {} \
--data {} \
--hyp {} --cfg yolov5n.yaml \
--device {} --exist-ok --patience 0 \
--name 5n_3 --project {} --weights {}".format(imgsize, epochs, batch_size, data_, hpy_aug, device, project_name, pt_5n_3_aug))

os.system("python train_freeze.py --img {} \
--epochs {} --batch-size {} \
--data {} \
--hyp {} --cfg yolov5s.yaml \
--device {} --exist-ok --patience 0 \
--name 5s_3 --project {} --weights {}".format(imgsize, epochs, batch_size, data_, hpy_aug, device, project_name, pt_5s_3_aug))

