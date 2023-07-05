from explainer.explainer import run
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# model weight path
yolo_w = "./yoda/weights/yolo_best_s.pt"
yoda_w = './yoda/weights/FE_stacked_convns_s_3.pt'

# detection path
det_path = './runs/detect/'
all_det_dir = os.listdir(det_path)


""" Post process plotting """
def plot_bbox(img, list_bbox, target_path):
    for bbox in list_bbox:
        center_x, center_y, width, height = bbox[1:5]
        x_min = int(center_x - width/2)
        y_min = int(center_y - height/2)
        x_max = int(center_x + width/2)
        y_max = int(center_y + height/2)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 10)
    # save image
    cv2.imwrite(target_path, img,)


def run_grad_cam(weights,
                 source,
                 w_yoda,
                 project,
                 backprop_array=['confidence'],
                 preprocess=False,
                 name='sub_grad'):
    return run(
        weights=weights,
        source=source,
        w_yoda=w_yoda,
        project=project,
        backprop_array=backprop_array,
        preprocess=preprocess,
        name=name,
    )


def combine_subregions_heatmap(subregion_folder, image_size, cropped_size):
    image_height, image_width = image_size
    subregion_width, subregion_height = cropped_size[0], cropped_size[1]


    image = np.zeros((image_height, image_width), dtype=np.uint8)
    print(f"shape of heating image: {image.shape}")


    for y in range(0, image_height, subregion_height):
        for x in range(0, image_width, subregion_width):
            filename = os.path.join(subregion_folder, f'subregion_{y}_{x}_heat_.png')
            
            if not os.path.exists(filename):
                # Check if there is a heatmap
                continue
            
            subregion = cv2.imread(filename, 0)

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

    return image


for dir in all_det_dir:
    current_dir = os.path.join(det_path, dir)
    sub_dir_list = os.listdir(current_dir)
    
    if not ('labels' in sub_dir_list) or not('temp_files' in sub_dir_list) or not ('temp_sub_processed' in sub_dir_list) or not ('full_image' in sub_dir_list):
        print(f'{current_dir}: do not have enough elements')
        continue
    
    
    source_im = os.path.join(current_dir, 'temp_sub_processed')
    project_path = os.path.join(current_dir, 'grads')
    
    assert os.path.exists(source_im), f'{source_im} does not exist'
    
    if not os.path.exists(project_path):
        os.mkdir(project_path)
        assert os.path.exists(project_path), f'{project_path} does not exist'
    
    # Run gradcam
    name_sub_grad = 'sub_grad'
    run_grad_cam(weights=yolo_w,
                 source=source_im,
                 w_yoda=yoda_w,
                 project=project_path,
                 name=name_sub_grad,)
    
    config_np = np.load(current_dir + '/config.npy')
    
    out_grad_im = os.path.join(project_path, name_sub_grad)
    heat_map_im = combine_subregions_heatmap(out_grad_im, image_size=config_np[0], cropped_size=config_np[1])
    cv2.imwrite(os.path.join(out_grad_im, 'final_heatmap.png'), heat_map_im)
    
    # Load full im and plot heatmap
    path_full_im = os.path.join(current_dir, 'full_image')
    full_im_candidate = [im for im in os.listdir(path_full_im)
                         if im.split('.')[-1] in ['jpg', 'png']]
    
    # Denied all detection jpg
    full_im_candidate = [im for im in full_im_candidate
                         if im.split('_')[-1].split('.')[0] == 'drc']
    
    full_im = cv2.imread(path_full_im + f'/{full_im_candidate[0]}')
    
    # Clip value to 0-1
    heat_map_im = heat_map_im / 255.0
    heat_map_im = heat_map_im.reshape((*config_np[0], 1))
    full_im = full_im / 255.0
    
    cam_im_full = show_cam_on_image(full_im, heat_map_im, use_rgb=True, image_weight=0.5)
    cam_im_full = cv2.cvtColor(cam_im_full, cv2.COLOR_BGR2RGB)
    
    # Grad-cam and detectiona
    label_path = os.path.join(path_full_im, f"{full_im_candidate[0].split('.')[0]}.txt")
    labels = np.loadtxt(label_path)
    cam_im_full_det = cam_im_full.copy()
    plot_bbox(cam_im_full_det, labels, os.path.join(path_full_im, 'grad_im_full_detected.png'))
    
    cv2.imwrite(os.path.join(path_full_im, 'grad_im_full.png'), cam_im_full) # Plot only grad
    


if __name__ == '__main__':
    print()