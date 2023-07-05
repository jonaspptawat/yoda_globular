import os

imgsize = 1024
epochs = 1000
batch_size = 2
data_ = "../running_data/npy_dataset/data.yaml"
# data_ = './data/data_npy.yaml'
hpy_aug = "./data/hyps/hyp.with-mosaic.yaml"
hpy_no_aug = "./data/hyps/hyp.no-mosaic.yaml"
device = 0

fe_models_list = ['FE_Add', 'FE_AddNS', 'FE_Conv', 'FE_ConvNS', 'FE_OneR']

'''
No Augmentations
'''

project_name = "./runs/train/no_augment"

# N
for fe_model in fe_models_list:
    cur_project_path = os.path.join(project_name, fe_model) # Get project name according to fe models
    
    os.system("python train.py --img {} \
    --epochs {} --batch-size {} \
    --data {} \
    --weights '' --hyp {} --cfg yolov5n.yaml \
    --device {} --exist-ok --patience 0 \
    --name scratch_5n --project {} --fe_models {}".format(imgsize, epochs, batch_size, data_, hpy_no_aug, device, cur_project_path, fe_model))

# S
for fe_model in fe_models_list:
    cur_project_path = os.path.join(project_name, fe_model) # Get project name according to fe models
    
    os.system("python train.py --img {} \
    --epochs {} --batch-size {} \
    --data {} \
    --weights '' --hyp {} --cfg yolov5s.yaml \
    --device {} --exist-ok --patience 0 \
    --name scratch_5s --project {} --fe_models {}".format(imgsize, epochs, batch_size, data_, hpy_no_aug, device, cur_project_path, fe_model))


'''
With Augmentations
'''

project_name = "./runs/train/augment"

# N
for fe_model in fe_models_list:
    cur_project_path = os.path.join(project_name, fe_model) # Get project name according to fe models
    
    os.system("python train.py --img {} \
    --epochs {} --batch-size {} \
    --data {} \
    --weights '' --hyp {} --cfg yolov5n.yaml \
    --device {} --exist-ok --patience 0 \
    --name scratch_5n --project {} --fe_models {}".format(imgsize, epochs, batch_size, data_, hpy_aug, device, cur_project_path, fe_model))

# S
for fe_model in fe_models_list:
    cur_project_path = os.path.join(project_name, fe_model) # Get project name according to fe models
    
    os.system("python train.py --img {} \
    --epochs {} --batch-size {} \
    --data {} \
    --weights '' --hyp {} --cfg yolov5s.yaml \
    --device {} --exist-ok --patience 0 \
    --name scratch_5s --project {} --fe_models {}".format(imgsize, epochs, batch_size, data_, hpy_aug, device, cur_project_path, fe_model))


'''
Size M Both Non-aug and Aug
'''

project_name = "./runs/train/no_augment"

# M
for fe_model in fe_models_list:
    cur_project_path = os.path.join(project_name, fe_model) # Get project name according to fe models

    os.system("python train.py --img {} \
    --epochs {} --batch-size {} \
    --data {} \
    --weights '' --hyp {} --cfg yolov5m.yaml \
    --device {} --exist-ok --patience 0 \
    --name scratch_5m --project {} --fe_models {}".format(imgsize, epochs, batch_size, data_, hpy_no_aug, device, cur_project_path, fe_model))


project_name = "./runs/train/augment"

# M
for fe_model in fe_models_list:
    cur_project_path = os.path.join(project_name, fe_model) # Get project name according to fe models

    os.system("python train.py --img {} \
    --epochs {} --batch-size {} \
    --data {} \
    --weights '' --hyp {} --cfg yolov5m.yaml \
    --device {} --exist-ok --patience 0 \
    --name scratch_5m --project {} --fe_models {}".format(imgsize, epochs, batch_size, data_, hpy_aug, device, cur_project_path, fe_model))
