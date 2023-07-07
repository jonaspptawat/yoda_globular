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

project_name = "./runs/train/base/no_augment"

# 3x3 and 1x1
os.system("python train.py --img {} \
--epochs {} --batch-size {} \
--data {} \
--weights '' --hyp {} --cfg yolov5n.yaml \
--device {} --exist-ok --patience 100 \
--name 5n_3 --project {}".format(imgsize, epochs, batch_size, data_, hpy_no_aug, device, project_name))

# S and M
os.system("python train.py --img {} \
--epochs {} --batch-size {} \
--data {} \
--weights '' --hyp {} --cfg yolov5s.yaml \
--device {} --exist-ok --patience 100 \
--name 5s_3 --project {}".format(imgsize, epochs, batch_size, data_, hpy_no_aug, device, project_name))




imgsize = 1024
epochs = 700
batch_size = 2
data_ = "../running_data/npy_dataset/data.yaml"
# data_ = './data/data_npy.yaml'
hpy_aug = "./data/hyps/hyp.with-mosaic.yaml"
hpy_no_aug = "./data/hyps/hyp.no-mosaic.yaml"
device = 0


'''
With Augmentations
'''

project_name = "./runs/train/base/augment"

# 3x3 and 1x1
os.system("python train.py --img {} \
--epochs {} --batch-size {} \
--data {} \
--weights '' --hyp {} --cfg yolov5n.yaml \
--device {} --exist-ok --patience 200 \
--name 5n_3 --project {}".format(imgsize, epochs, batch_size, data_, hpy_aug, device, project_name))

# S and 
os.system("python train.py --img {} \
--epochs {} --batch-size {} \
--data {} \
--weights '' --hyp {} --cfg yolov5s.yaml \
--device {} --exist-ok --patience 200 \
--name 5s_3 --project {}".format(imgsize, epochs, batch_size, data_, hpy_aug, device, project_name))


'''
Freeze YODA Training
'''
os.system("python run_script_npy_freeze.py")
