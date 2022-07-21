import os
import cv2
import time
import argparse
import numpy as np
import torch
import model.detector
import utils.utils
from PIL import Image
import torch.backends.cudnn as cudnn
import albumentations as A
import archs
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import math
from utils2 import AverageMeter,str2bool
import yaml
import xlsxwriter as xw
import pandas as pd
import openpyxl as op

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

from tqdm import tqdm
from metrics import iou_score
from utils2 import AverageMeter
from glob import glob


threshold=0.25
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
k_size=41
mid_k_size=int(k_size/2)+1
out_tipx = []
out_tipy = []
out_angle = []
out_length = []
mask_tipx = []
mask_tipy = []
mask_angle = []
mask_length = []

print(mid_k_size)
kernel_default=np.zeros([k_size,k_size],dtype=np.uint8)*255
img_line=cv2.line(kernel_default,(0,mid_k_size),(k_size,mid_k_size),(1,1,1),2).astype(np.uint8)
print(img_line)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='train_crop_aug3_RAUnet1_woDS',
                        help='model name')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--data', type=str, default='D:/YoloFastestV2_unet2/data/coco.data',
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='weights/coco-290-epoch-0.987358ap-model.pth',
                        help='The path of the .pth model to be transformed')

    args = parser.parse_args()

    return args


def prepare_image2 (img):
    # convert the color from BGR to RGB then convert to PIL array
    cvt_image =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cvt_image=cvt_image/255
    im_pil = Image.fromarray(cvt_image)
    img_array = img_to_array(im_pil)
    image_array_expanded = img_array
    return image_array_expanded


# This function calculates the normal angle and length values
def angle_distance_calculate(min_x, min_y, max_x, max_y):
    xlength = max_x - min_x
    ylength = max_y - min_y
    if (xlength == 0):
        Gradient = 0
    else:
        Gradient = ylength / xlength  # opp/adjacent

    length = math.sqrt(math.pow(ylength, 2) + math.pow(xlength, 2))
    angle = math.atan2(ylength, xlength) * 180 / 3.14
    if (angle < 0):  # Please find better way of not getting negative angles Masi
        angle = angle + 180
    return length, angle


def line_of_best_fit(array_bestx, array_besty,change=False):
    # print("array_bestx",array_bestx)
    # print("array_besty",array_besty)
    unique_x = len(np.unique(array_bestx))
    unique_y = len(np.unique(array_besty))
    new_arry = []
    new_arrx = []
    if (np.size(array_bestx) == 0):
        bestfit_length = 0
        bestfit_angle = 0
        x_cordinates = [0, 0]
        y_cordinates = [0, 0]
        return x_cordinates, y_cordinates, bestfit_angle, bestfit_length
    if (unique_x >= unique_y):
        # print('first method')
        model_fit = np.polyfit(array_bestx, array_besty, 1)
        predict = np.poly1d(model_fit)
        pred_array = predict(array_bestx)
        max_x = len(array_bestx) - 1
        max_y = len(pred_array) - 1
        yy1 = pred_array[0]
        xx1 = array_bestx[0]
        yy2 = pred_array[max_y]
        xx2 = array_bestx[max_x]
    else:
        # print('second method')
        model_fit = np.polyfit(array_besty, array_bestx, 1)
        predict = np.poly1d(model_fit)
        pred_array = predict(array_besty)
        max_y = len(array_besty) - 1
        max_x = len(pred_array) - 1
        yy1 = array_besty[0]
        xx1 = pred_array[0]
        yy2 = array_besty[max_y]
        xx2 = pred_array[max_x]

    if change==True:
        xx2  = xx2 + 11
        yy2  = yy2 - 6.5
        yy1  = yy1 + 2

    x_cordinates = [xx1, xx2]
    y_cordinates = [yy1, yy2]


    #print("x values", x_cordinates)
    #print("y values", y_cordinates)

    bestfit_length, bestfit_angle = angle_distance_calculate(xx1, yy1, xx2, yy2)

    return x_cordinates, y_cordinates, bestfit_angle, bestfit_length


def detect_angle_length(im, img_resized, image_name):
    # global lineofbestfit_time
    im2 = 1 - im
    # BW = im2#(im2 >0.5).astype(np.uint8)
    BW = im2  # remove_small_objects(im2, min_size=100, connectivity=50)
    '''
    BW = cv2.morphologyEx(im2, cv2.MORPH_OPEN, None)
    BW = cv2.dilate(BW, None, iterations=1)
    '''
    im3 = BW  # thin(BW)
    im3 = np.uint8(im3)
    #cv2.imwrite('D:/YoloFastestV2_unet2/images_saved/'+image_name+'b.png',im3*255)

    array_besty, array_bestx = (im3 == 1).nonzero()
    # timenow=time.time()
    x_cordinates, y_cordinates, bestfit_angle, bestfit_length = line_of_best_fit(array_bestx, array_besty)
    '''
    global img_line, k_size, mid_k_size
    img_line = cv2.line(kernel_default, (0, mid_k_size), (k_size, mid_k_size), (1, 1, 1), 2).astype(np.uint8)
    M = cv2.getRotationMatrix2D((mid_k_size, mid_k_size), 180 - bestfit_angle, 1)
    img_line2 = cv2.warpAffine(img_line, M, (k_size, k_size))
    # print(img_line2)
    img_line=cv2.merge((img_line,img_line,img_line))
    #cv2.imwrite('D:/YoloFastestV2_unet2/images_saved/DOGPL.png',img_line)
    im3 = cv2.dilate(im3, img_line2, iterations=1)
    # im3=cv2.erode(im3, img_line2, iterations=2)
    if img_resized[0]:
        im3 = cv2.resize(im3, (img_resized[1], img_resized[2]), interpolation=cv2.INTER_LINEAR)
    # im3=thin(im3).astype(np.uint8)

    array_besty, array_bestx = (im3 == 1).nonzero()
    x_cordinates, y_cordinates, bestfit_angle, bestfit_length = line_of_best_fit(array_bestx, array_besty,change=True)
    '''
    im3_w, im3_h = im3.shape
    line_crop = np.zeros([im3_w, im3_h], dtype=np.uint8)
    line_crop = cv2.line(line_crop, (int(x_cordinates[0]), int(y_cordinates[0])),
                         (int(x_cordinates[1]), int(y_cordinates[1])), (1, 1, 1), 2).astype(np.uint8)
    #cv2.imwrite('D:/YoloFastestV2_unet2/images_saved/'+image_name+'L.png',line_crop)
    im3 = cv2.multiply(im3, line_crop)

    return im3, bestfit_angle, bestfit_length, x_cordinates, y_cordinates

def maskdetect_angle_length(im):
    # global lineofbestfit_time
    '''
    im2 = 1 - im
    # BW = im2#(im2 >0.5).astype(np.uint8)
    BW = im2  # remove_small_objects(im2, min_size=100, connectivity=50)
    BW = cv2.morphologyEx(im2, cv2.MORPH_OPEN, None)
    BW = cv2.dilate(BW, None, iterations=1)
    im3 = BW  # thin(BW)
    im3 = np.uint8(im3)
    #cv2.imwrite('D:/YoloFastestV2_unet2/images_saved/'+image_name+'b.png',im3*255)

    array_besty, array_bestx = (im3 == 1).nonzero()
    # timenow=time.time()
    x_cordinates, y_cordinates, bestfit_angle, bestfit_length = line_of_best_fit(array_bestx, array_besty)
    global img_line, k_size, mid_k_size
    img_line = cv2.line(kernel_default, (0, mid_k_size), (k_size, mid_k_size), (1, 1, 1), 2).astype(np.uint8)
    M = cv2.getRotationMatrix2D((mid_k_size, mid_k_size), 180 - bestfit_angle, 1)
    img_line2 = cv2.warpAffine(img_line, M, (k_size, k_size))
    # print(img_line2)
    img_line=cv2.merge((img_line,img_line,img_line))
    #cv2.imwrite('D:/YoloFastestV2_unet2/images_saved/DOGPL.png',img_line)
    im3 = cv2.dilate(im3, img_line2, iterations=1)
    # im3=cv2.erode(im3, img_line2, iterations=2)
    ##if img_resized[0]:
    ##    im3 = cv2.resize(im3, (img_resized[1], img_resized[2]), interpolation=cv2.INTER_LINEAR)
    # im3=thin(im3).astype(np.uint8)

    array_besty, array_bestx = (im3 == 1).nonzero()
    x_cordinates, y_cordinates, bestfit_angle, bestfit_length = line_of_best_fit(array_bestx, array_besty, change=True)

    im3_w, im3_h = im3.shape
    line_crop = np.zeros([im3_w, im3_h], dtype=np.uint8)
    line_crop = cv2.line(line_crop, (int(x_cordinates[0]), int(y_cordinates[0])),
                         (int(x_cordinates[1]), int(y_cordinates[1])), (1, 1, 1), 2).astype(np.uint8)
    #cv2.imwrite('D:/YoloFastestV2_unet2/images_saved/'+image_name+'L.png',line_crop)
    im3 = cv2.multiply(im3, line_crop)
    '''
    im2 = 1 - im
    array_besty, array_bestx = (im2 == 1).nonzero()
    x_cordinates, y_cordinates, bestfit_angle, bestfit_length = line_of_best_fit(array_bestx, array_besty)


    return bestfit_angle, bestfit_length, x_cordinates, y_cordinates

def image_output(w, y, img_resized, image_name):
    im3, bestfit_angle, bestfit_length, x_cordinates, y_cordinates = detect_angle_length(y, img_resized, image_name)
    # im3=ndimage.binary_dilation(im3)

    h = int(x_cordinates[1] + img_resized[5])
    k = int(y_cordinates[1] + img_resized[3])
    #global out_tipx,out_tipy,out_angle,out_length
    out_tipx.append(h)
    out_tipy.append(k)
    out_angle.append(bestfit_angle)
    out_length.append(bestfit_length)
    #cordinate_tip_x = h
    #cordinate_tip_y = k
    print('tip:', (h, k))
    print("angle",bestfit_angle)
    print("length",bestfit_length)


    L = 4
    p1 = [h + L, k + L]
    p3 = [h - L, k - L]

    im3 = im3 * 255
    im3 = cv2.resize(im3, (img_resized[4] - img_resized[3], img_resized[6] - img_resized[5]), interpolation=cv2.INTER_LINEAR)
    im3_complement = 1 - (im3 / 255).astype(np.uint8)
    b_img, g_img, r_img = cv2.split(w)

    r_img[img_resized[3]:img_resized[4], img_resized[5]:img_resized[6]] = cv2.add(
        r_img[img_resized[3]:img_resized[4], img_resized[5]:img_resized[6]], im3)
    b_img[img_resized[3]:img_resized[4], img_resized[5]:img_resized[6]] = cv2.multiply(
        b_img[img_resized[3]:img_resized[4], img_resized[5]:img_resized[6]], im3_complement)
    g_img[img_resized[3]:img_resized[4], img_resized[5]:img_resized[6]] = cv2.multiply(
        g_img[img_resized[3]:img_resized[4], img_resized[5]:img_resized[6]], im3_complement)

    imgclr2 = cv2.merge((b_img, g_img, r_img))
    return imgclr2, p1, p3


def plot_sample(real_image, ori_img, binary_preds, img_resized, image_name):
    binary = (np.squeeze(binary_preds))
    w = real_image

    # cv2.imwrite('../images_saved/'+image_name+'.png',w)
    # img_resized=[True,prev_width,prev_height,start_row,end_row,start_col,end_col,actual_height,actual_width]
    # Predicted image
    imgclr2, p1, p3 = image_output(w, binary, img_resized, image_name)
    cv2.rectangle(imgclr2, (p1[0], p1[1]), (p3[0], p3[1]), (0, 255, 255))
    #cv2.rectangle(imgclr2, (img_resized[5], img_resized[3]), (img_resized[6], img_resized[4]), (0, 255, 0),thickness=2)
    cv2.rectangle(w, (img_resized[5], img_resized[3]), (img_resized[6], img_resized[4]), (0, 255, 0),thickness=2)
    cv2.imwrite('D:/YoloFastestV2_unet2/images_saved/' + image_name + 'P.png', imgclr2)
    cv2.imwrite('D:/YoloFastestV2_unet2/images_saved/' + image_name + 'w.png', w)
    #cv2.imwrite('D:/YoloFastestV2_unet2/images_saved/' + image_name + 'w.png', w)
    #cv2.rectangle(w, (p1[0], p1[1]), (p3[0], p3[1]), (0, 255, 255))
    #cv2.imwrite('D:/YoloFastestV2_unet2/images_saved/' + image_name + 'w.png', w)

if __name__ == '__main__':
    #指定训练配置文件
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model_seg = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model_seg = model_seg.cuda()

    # Data loading code
    path = os.path.join('inputs', config['dataset'], 'img', '*' + config['img_ext'])
    img_ids = glob(os.path.join('inputs', 'test_images', 'img', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    val_img_ids = img_ids
    model_seg.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model_seg.eval()
    val_transform = Compose([
        A.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    cfg = utils.utils.load_datafile(args.data)


    #模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_detect = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model_detect.load_state_dict(torch.load(args.weights, map_location=device))

    #sets the module in eval node
    model_detect.eval()

    avg_meter = AverageMeter()
    dirname_input_image = "D:\\YoloFastestV2_unet2\\inputs\\test_images\\img"
    image_pathnames = sorted(glob(dirname_input_image + "/*.jpg"))
    dirname_input_mask = "D:\\YoloFastestV2_unet2\\inputs\\test_images\\mask"
    mask_pathnames = sorted(glob(dirname_input_mask + "/*.jpg"))
    # image_pathnames=image_pathnames[15:20]
    num_images = len(image_pathnames)
    segmentation_time = 0
    post_processing_time = 0
    detection_time = 0
    interfacer_time = 0

    start_time = time.time()
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('output', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for image_pathname in image_pathnames:
            meta = os.path.splitext(image_pathname.split('\\')[-1])[0]
            real_img = cv2.imread(image_pathname)
            image = Image.open(image_pathname)
            actual_height, actual_width, actual_channels = real_img.shape

            timenow = time.time()
            #start_row, start_col, end_row, end_col = yolo.detect_image(image)
            res_img = cv2.resize(real_img, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
            img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
            img = torch.from_numpy(img.transpose(0, 3, 1, 2))
            img = img.to(device).float() / 255.0
            preds = model_detect(img)
            # 模型推理
            # 特征图后处理
            output = utils.utils.handel_preds(preds, cfg, device)
            output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)
            # 加载label names
            LABEL_NAMES = []
            with open(cfg["names"], 'r') as f:
                for line in f.readlines():
                    LABEL_NAMES.append(line.strip())

            h0, w0, _ = real_img.shape
            scale_h, scale_w = h0 / cfg["height"], w0 / cfg["width"]

            # 绘制预测框

            box = output_boxes[0][0].tolist()

            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]

            start_col,start_row = int(box[0] * scale_w), int(box[1] * scale_h)
            end_col,end_row = int(box[2] * scale_w), int(box[3] * scale_h)
            box_center_x = (start_col + end_col)/2
            box_center_y = (start_row + end_row)/2
            img_center_x = w0/2
            img_center_y = h0/2
            detection_time += (time.time() - timenow)
            timenow = time.time()

            w = end_col - start_col
            h = end_row - start_row
            if (w <= 256 and h <= 256):
                end_col = start_col + 256
                end_row = start_row + 256
            else:
                if (w < h):
                    end_col = start_col + h
                else:
                    end_row = start_row + w

                # shift if new bounding box exceeds image boundaries
            if (start_col < 0):
                end_col = end_col - start_col
                start_col = 0

            if (start_row < 0):
                end_row = end_row - start_row
                start_row = 0

            if (end_col > actual_width):
                start_col = start_col - (end_col - actual_width)
                end_col = actual_width

            if (end_row > actual_height):
                start_row = start_row - (end_row - actual_height)
                end_row = actual_height

            cropped = real_img[start_row:end_row, start_col:end_col]
            if ((end_col - start_col) > 256):
                prev_width, prev_height, prev_channel = cropped.shape
                img_resized = [True, prev_width, prev_height, start_row, end_row, start_col, end_col, actual_height,
                               actual_width]
                cropped = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_LINEAR)
            else:
                img_resized = [True, 256, 256, start_row, end_row, start_col, end_col, actual_height, actual_width]

            augmented = val_transform(image=cropped)  # 这个包比较方便，能把mask也一并做掉
            img = augmented['image']
            img = img.astype('float32') / 255
            img = img.transpose(2, 0, 1)
            interfacer_time += (time.time() - timenow)
            input = torch.from_numpy(img)
            input = input.unsqueeze(0)
            input = input.cuda()
            Flag = config['deep_supervision']
            # compute output
            timenow = time.time()
            if config['deep_supervision']:
                output = model_seg(input)[-1]
            else:
                output = model_seg(input)
            output = torch.sigmoid(output).cpu().numpy()
            segmentation_time += (time.time() - timenow)
            for i in range(len(output)):
                for c in range(config['num_classes']):
                    predict_binary = (output[i, c] > 0.5).astype(np.uint8)

                    # predict_binary = cv2.resize(predict_binary, (actual_width, actual_height), interpolation=cv2.INTER_NEAREST)
                    out_put = (predict_binary * 255).astype('uint8')
                    # out_put = cv2.resize(out_put, (actual_width, actual_height), interpolation=cv2.INTER_NEAREST)
                    prev_width = img_resized[1]
                    prev_height = img_resized[2]
                    start_row = img_resized[3]
                    end_row = img_resized[4]
                    start_col = img_resized[5]
                    end_col = img_resized[6]
                    actual_height = img_resized[7]
                    actual_width = img_resized[8]
                    real_img = real_img.astype('uint8')
                    # predict_binary = predict_binary[start_row:end_row,start_col: end_col]
                    crop = cropped.astype('uint8')
                    cv2.imwrite('D:/YoloFastestV2_unet2/images_saved/' + meta + '_crop.png', crop)
                    crop = prepare_image2(crop)
                    x = crop.squeeze()
                    img_resize = [True, prev_width, prev_height, start_row, end_row, start_col, end_col,
                                  actual_height, actual_width]
                    cv2.imwrite(os.path.join('output', config['name'], str(c), meta + '.jpg'),
                                out_put)
                    timenow = time.time()
                    plot_sample(real_img, x, predict_binary, img_resize, meta)
                    post_processing_time += (time.time() - timenow)

    n = num_images
    # print('IoU: %.4f' % avg_meter.avg)
    full_time = (time.time() - start_time) / n
    print("Time per image", (full_time))
    print("FPS= ", 1 / ((time.time() - start_time) / n))
    print("Detection", detection_time / n)
    print("Interfacer", interfacer_time / n)
    print("Segmentation", segmentation_time / n)
    print("Post Processing", post_processing_time / n)
    #global mask_tipx, mask_tipy, mask_angle, mask_length

    for mask_pathname in mask_pathnames:
        meta = os.path.splitext(mask_pathname.split('\\')[-1])[0]
        mask = cv2.imread(mask_pathname)
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_binary = mask_gray.astype('float32') / 255
        mask_binary = (mask_binary > 0.5).astype(np.uint8)
        bestfit_angle, bestfit_length, x_cordinates, y_cordinates = maskdetect_angle_length(mask_binary)
        mask_tipx.append(round(x_cordinates[1]))
        mask_tipy.append(round(y_cordinates[1]))
        mask_angle.append(bestfit_angle)
        mask_length.append(bestfit_length)

        print('tip:', (round(x_cordinates[1]), round(y_cordinates[1])))
        print('angle:', bestfit_angle)
        print('length', bestfit_length)
    #   plot_examples(input, target, model,num_examples=3)
    torch.cuda.empty_cache()
    #数据预处理
    #math.sqrt(math.pow(ylength, 2) + math.pow(xlength, 2))
    tip_error = []
    angle_error = []
    length_error = []
    for i in range(len(mask_tipx)):
        tip_error.append( math.sqrt(math.pow((mask_tipx[i] - out_tipx[i]),2)+math.pow((mask_tipy[i] - out_tipy[i]),2)) )
        angle_error.append( math.fabs(mask_angle[i] - out_angle[i]) )
        length_error.append( math.fabs(mask_length[i] - out_length[i]) )

    testData = [mask_tipx,mask_tipy,mask_angle,mask_length,out_tipx,out_tipy,out_angle,out_length,tip_error,angle_error ,length_error,tip_error,angle_error,length_error]
    filename = 'redult.xlsx'
    


    def pd_toexcel(data, filename):  # pandas库储存数据到excel
        dfData = {  # 用字典设置DataFrame所需数据
            'mask_tipx': data[0],
            'mask_tipy': data[1],
            'mask_angle': data[2],
            'mask_length':data[3],
            'out_tipx': data[4],
            'out_tipy': data[5],
            'out_angle': data[6],
            'out_length': data[7],
            'tip_error':data[8],
            'angle_error':data[9],
            'length_error':data[10]
        }
        df = pd.DataFrame(dfData)  # 创建DataFrame
        print(df)
        df.to_excel(filename, index=False)  # 存表，去除原始索引列(0,1,2...)

    
    pd_toexcel(testData, filename)

    

