# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import numpy as np
import glob
import time
import re
import matplotlib.pyplot as plt
from astropy.io import fits


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yoda.stacked_convns_models import YODA

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

##################################################################################################
""" Dir processing"""
# create folder function
def create_folder(folder):
    if not os.path.exists(folder):
            os.makedirs(folder)

# remove folder function
def remove_folder(folder):
    try:
        os.system(f"rm -r {folder}")
        print(f"Removed temp files in {colorstr('bold', folder)}")
    except:
        print(f"Failed to remove temp files in {colorstr('bold', folder)}")

""" Post process plotting """
def plot_bbox(img, list_bbox, target_path):
    for bbox in list_bbox:
        center_x, center_y, width, height = bbox[1:5]
        x_min = int(center_x - width/2)
        y_min = int(center_y - height/2)
        x_max = int(center_x + width/2)
        y_max = int(center_y + height/2)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 10)
    # save image
    plt.imsave(target_path, img, cmap='gray')

""" Preprocess for FITS file """
def load_raw_data(data_path, raw_type, flip=False):
    if raw_type == 0: # for fits
        print(f"Processing {colorstr('bold', 'FITS')} file...")
        hdu_list = fits.open(data_path)
        data = hdu_list['SCI'].data.astype(np.float32)
    if raw_type == 1: # for npy
        print(f"Processing {colorstr('bold', 'npy')} file...")
        data = np.load(data_path)
    if flip:
        print(f"{colorstr('bold', 'Flipping')} data...")
        data = np.flipud(data)
        data = np.flip(data, axis=0)
    return data

""" Cropping function """
def crop_subregions(image, subregion_size, output_folder):
    original_height, original_width = image.shape
    subregion_height, subregion_width = subregion_size
    print(f"original shape: {colorstr('bold', image.shape)}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for y in range(0, original_height, subregion_height): # Step every 1024
        for x in range(0, original_width, subregion_width): # Step every 1024
            # if subregion hight or width is not out of bound, then crop with normal case
            
            # If y, x coord is not larger than ori_y,x
            if y + subregion_height <= original_height and x + subregion_width <= original_width:
                subregion = image[y:y+subregion_height, x:x+subregion_width]
            
            # If y coord is not larger but x, then grab all x till end
            elif y + subregion_height <= original_height and x + subregion_width > original_width:
                subregion = image[y:y+subregion_height, x::]
            
            # If x coord is not larger but y, then grab all y till end
            elif y + subregion_height > original_height and x + subregion_width <= original_width:
                subregion = image[y::, x:x+subregion_width]
            
            # If y,x coord is larger, then grab the rest(remaining)
            elif y + subregion_height > original_height and x + subregion_width > original_width:
                subregion = image[y::, x::]
                
            else:
                print('Warning: subregion is out of bound')
                
            filename = os.path.join(output_folder, f'subregion_{y}_{x}.npy')
            np.save(filename, subregion)
            
            print('shape of subregion: ', subregion.shape)

def combine_subregions(subregion_folder, detected_labels_folder, image_size, cropped_size):
    image_height, image_width = image_size
    subregion_width, subregion_height = cropped_size[0], cropped_size[1]

    # load whole labels
    labels_list = glob.glob(os.path.join(detected_labels_folder, '*'))
    labels_list = [i.split('/')[-1].split('.')[0] for i in labels_list]

    image = np.zeros((image_height, image_width), dtype=np.float32)
    print(f"shape of merging image: {colorstr('bold', image.shape)}")

    # # make save dir (TESTING)
    # save_dir_test = '/'.join(str(subregion_folder).split('/')[:-1])
    # save_dir_temp = os.path.join(save_dir_test, 'test_im')
    # os.mkdir(save_dir_temp)

    labels_for_merged_image = []
    for y in range(0, image_height, subregion_height):
        for x in range(0, image_width, subregion_width):
            filename = os.path.join(subregion_folder, f'subregion_{y}_{x}.npy')
            subregion = np.load(filename)
            
            # Save subregion to test dir (TESTING)
            # print(subregion.shape)
            # im_to_save = subregion[0, 0, :, :]
            # plt.imsave(os.path.join(save_dir_temp, f'subregion_{y}_{x}.png'), im_to_save, cmap='gray')
            
            
            filename_x = filename.split('/')[-1].split('.')[0]
            if filename_x in labels_list:
                # this below code to load labels then convert from subregion coordinate to original image coordinate
                # load labels which are same name with subregion
                with open(os.path.join(detected_labels_folder, f'{filename_x}.txt'), 'r') as f:
                    labels = f.readlines()
                # the example of labels
                # class x_center y_center width height
                # 0 0.29541 0.258789 0.196289 0.195312 0.906503
                # 0 0.35498 0.889648 0.194336 0.195312 0.918074
                # here we need to convert yolo format to pixel coordinate (0-1 to pixel coordinate)
                # then convert subregion coordinate to original image coordinate (change the coordinate from subregion to original image)
                # final result is center x, center y, width, height in pixel unit
                # convert YOLO format to pixel coordinate (0-1 to pixel coordinate)
                labels = [label.split() for label in labels]
                for label in labels:
                    class_index = int(label[0])
                    x_center = float(label[1])
                    y_center = float(label[2])
                    width = float(label[3])
                    height = float(label[4])

                    # Convert from YOLO format to pixel coordinates
                    x_center = x_center * subregion_width + x
                    y_center = y_center * subregion_height + y
                    width = width * subregion_width
                    height = height * subregion_height

                    # Convert from subregion coordinate to original image coordinate
                    x_center = min(max(x_center, 0), image_width)
                    y_center = min(max(y_center, 0), image_height)
                    width = min(width, image_width - x_center)
                    height = min(height, image_height - y_center)

                    # Append the converted label to the list
                    labels_for_merged_image.append([class_index, x_center, y_center, width, height])
                    
            # if shape of subregion is greater than 2 dimensions, then convert to 2 dimensions
            # by taking only last 2 dimensions
            if len(subregion.shape) > 2:
                subregion = subregion[0, 0, :, :]

            # if subregion hight or width is not out of bound, then fill
            if y + subregion_height <= image_height and x + subregion_width <= image_width:
                image[y:y+subregion_height, x:x+subregion_width] = subregion
            elif y + subregion_height <= image_height and x + subregion_width > image_width:
                image[y:y+subregion_height, x::] = subregion[:, 0:image_width-x]
            elif y + subregion_height > image_height and x + subregion_width <= image_width:
                image[y::, x:x+subregion_width] = subregion[0:image_height-y, :]
            elif y + subregion_height > image_height and x + subregion_width > image_width:
                image[y::, x::] = subregion[0:image_height-y, 0:image_width-x]
            # otherwise, fill with only remaining part in the image
            else:
                print('Warning: subregion is out of bound')

    return image, labels_for_merged_image
##################################################################################################


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        ##########################################
        raw_type=None, # 0 means FITS, 1 means npy
        ##########################################
        yoda_weights='./best_weights/FE_stacked_convns_s_3.pt',
        out_filter='Linear',
):
    ##################################################################################################
    """ 
        After load full npy files, we need to separate parts of them to 1024 x 1024 pixels.
        Then we will loop through all the parts and do the inference.
    """
    filter_list = ['Linear', 'Log', 'Power', 'Sqrt', 'Squared', 'ASINH', 'SINH']
    if (out_filter in filter_list):
        out_filter_idx = filter_list.index(out_filter)
    else:
        assert False, f'Cant find filter: {out_filter}'
    
    npy_path = str(source)
    # get all the npy files
    if raw_type == 0:
        npy_path = glob.glob(npy_path + '/*.fits')
        if npy_path == [] or npy_path == None:
            assert False, 'No FITS files in the folder'
    elif raw_type == 1:
        npy_path = glob.glob(npy_path + '/*.npy')
        if npy_path == [] or npy_path == None:
            assert False, 'No npy files in the folder'

    for index in range(len(npy_path)): # loop through whole npy files (full FITS files)
        # initialize the timer
        t0 = time.time()

        # create a temp folder to save the parts of processing npy files
        temp_sub_folder_i = os.path.join(project, 'img_{}'.format(index))
        create_folder(temp_sub_folder_i)

        # create label temp folder to save the parts
        temp_label_sub_i = os.path.join(temp_sub_folder_i, 'labels')
        create_folder(temp_label_sub_i)

        # create temp folder to save grad-cam's results
        temp_grad_sub_i = os.path.join(temp_sub_folder_i, 'grads')
        create_folder(temp_grad_sub_i)

        temp_folder = os.path.join(temp_sub_folder_i, 'temp_files')
        create_folder(temp_folder)

        # Load the original image data
        original_image = load_raw_data(npy_path[index], raw_type, flip=False)

        # Define the size of the sub-regions
        subregion_size = (imgsz[0], imgsz[1])
        # Specify the output folder
        output_folder = temp_folder
        # Crop the sub-regions and save them into the folder
        crop_subregions(original_image, subregion_size, output_folder)

        source = temp_folder
        image_name = os.path.basename(npy_path[index]).split('.')[0]
    ##################################################################################################
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        
        yoda = YODA(channels=7, kernels=(1, 3))
        yoda.load_state_dict(torch.load(yoda_weights))
        yoda.float()
        try: 
            yoda.to(device)
        except:
            yoda.cpu()
        yoda.eval() # Need to be eval to make output consistent (make sure disable BN)

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        # model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        ##############################################################################
        try:
            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        except:
            model.warmup(imgsz=(1 if pt or model.triton else bs, 1, *imgsz))  # warmup
        ##############################################################################
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                # im = torch.from_numpy(im).to(model.device)
                ###############################################
                im_copy = im.copy()
                
                # im_to_save = im.copy().reshape((im_copy.shape[1], im_copy.shape[2]))
                # to_save_path = os.path.join(temp_grad_sub_i, f"{str(path).split('/')[-1].split('.')[0]}.png")
                # plt.imsave(to_save_path, im_to_save, cmap='gray')
                # print(im_to_save.shape)
                
                
                im = torch.from_numpy(im_copy).to(model.device)
                ###############################################
                im = im.float()  # uint8 to fp16/32
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                # pred = model(im, augment=augment, visualize=visualize)
                ######################################################################################################
                
                # Transformed data using YODA
                im, stacked_im = yoda(im)
                
                pred, train_out = model(im, augment=augment, visualize=visualize)
                temp_sub_processed = os.path.join(temp_sub_folder_i, 'temp_sub_processed')
                create_folder(temp_sub_processed)

                export_path = path.split('/')[-1].split('.')[0]
                img_i_path = os.path.join(temp_sub_processed, export_path)

                # save sub .npy that is processed by YoDa
                np.save(img_i_path, stacked_im[:, out_filter_idx:out_filter_idx+1, :, :].detach().cpu().numpy())

                # save only first channel of im.detach().cpu().numpy()
                # np.save(img_i_path, im[0, 1, :, :].detach().cpu().numpy())
                ######################################################################################################

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                
                
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                
                ###################################################### 
                # For label to be the same dir
                txt_path = temp_label_sub_i + f'/{str(p.stem)}'
                ######################################################
                
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            ######################################################################################################
                            # save .npy instead of .jpg
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.npy', BGR=True)
                            ######################################################################################################

                # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            ##################################################################
            # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            ##################################################################
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

        #################################################################################################
        # mearge sub .npy files into one .npy file
        # create a temp folder to save sub-image processed npy and jpg files
        export_full_image_path = os.path.join(temp_sub_folder_i, 'full_image/')
        create_folder(export_full_image_path)

        # output_image, coor_and_label = combine_subregions(temp_sub_processed, str(save_dir) + '/labels', original_image.shape, subregion_size)
        output_image, coor_and_label = combine_subregions(temp_sub_processed, temp_label_sub_i, original_image.shape, subregion_size)
        # save coor_and_label with txt format
        np.savetxt(export_full_image_path + image_name + '.txt', coor_and_label, fmt='%s')
        
        # save the combined image with .npy and .jpg format
        np.save(export_full_image_path + image_name + '.npy', output_image)
        # export with matplotlib
        plt.imsave(export_full_image_path + image_name + '.jpg', output_image, cmap='gray')

        # plot bbox on the combined image
        image_as_jpg = cv2.imread(export_full_image_path + image_name + '.jpg')
        image_as_jpg = cv2.cvtColor(image_as_jpg, cv2.COLOR_BGR2GRAY)
        image_as_jpg = cv2.cvtColor(image_as_jpg, cv2.COLOR_BGR2RGB)
        plot_bbox(image_as_jpg, coor_and_label, export_full_image_path + image_name + '_detected' + '.jpg')
        
        # # remove temp files in temp_folder
        # remove_folder(temp_folder)
        # # remove temp_sub_processed
        # remove_folder(temp_sub_processed)
        # # remove labels folder in testing after end of each image
        # remove_folder(temp_label_sub_i)

        # remove labels folder in testing after end of each image
        remove_folder(save_dir)
        
        config_np = np.array([original_image.shape, subregion_size])
        np.save(temp_sub_folder_i+'/config.npy', config_np)

        print(f'Finished at {time.time() - t0:.3f}s | {index + 1} / {len(npy_path)} of images processed')
        #################################################################################################

    os.system('python grad_cam.py') # Run grad cam for every directory

def parse_opt():
    parser = argparse.ArgumentParser()
    ###########################################################################################
    parser.add_argument('--raw-type', type=int, default=None, help='0 means FITS, 1 means npy')
    ###########################################################################################
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')

    parser.add_argument('--yoda_weights', type=str, default='./best_weights/FE_stacked_convns_s_3.pt', help='yoda model path or triton URL')
    parser.add_argument('--out_filter', type=str, default='Linear', help='Output filter of detection') 
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)