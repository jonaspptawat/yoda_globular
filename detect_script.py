import os

imgsize = 1024
source = "../running_data/fits/"
project_name = "./runs/detect"
conf_thres = 0.25
iou_thres = 0.45

# Full image output filter
out_filter = 'Linear'

weight_ = "./yoda/weights/yolo_best_s.pt"
yoda_weight = './yoda/weights/FE_stacked_convns_s_3.pt'

os.system("python detect_modify.py --imgsz {} \
--source {} --conf-thres {} --iou-thres {}  \
--weights {} --save-conf --save-txt \
--device cpu --exist-ok --raw-type 0 \
--name testing --project {} --yoda_weights {} --out_filter {}".format(imgsize, source, conf_thres, iou_thres, weight_, project_name, yoda_weight, out_filter))