import os
import cv2
import math
import time
import json
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import shape, box
import shapely

SLIDE_FORMAT = ["sdpc", "svs", "ndpi", "tiff", "tif", "dcm", "svslide", "bif", "vms", "vmu", "mrxs", "scn"]
ANNOTATION_FORMAT = ["sdpl", "json"]

# 配置 OpenSlide 路径
OPENSLIDE_PATH = r'E:\Anaconda\envs\DL\Library\openslide-win64-20231011\bin'
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

parser = argparse.ArgumentParser(description='Code to tile WSI')
# necessary params.
parser.add_argument('--data_dir', type=str,
                    default=r'G:\前列腺\data_test',
                    help='dir of slide files')
parser.add_argument('--save_dir', type=str,
                    default=r'G:\前列腺\save_test',
                    help='dir of patch saving')
parser.add_argument('--annotation_dir', type=str,
                    default=r'G:\前列腺\annotation_test',
                    help='dir of annotation files (optional)')

# general params. (we set the layer at the highest magnification as 40x if no magnification is provided in slide properties)
parser.add_argument('--which2cut', type=str, default="magnification", choices=["magnification", "resolution"], 
                    help='use magnification or resolution to cut patches')
parser.add_argument('--magnification', type=float, default=20, help='magnification to cut patches: 5x, 20x, 40x, ...')
parser.add_argument('--resolution', type=float, default=0.4, help='resolution to cut patches: 0.103795, ... (um/pixel)')
parser.add_argument('--patch_w', type=int, default=1024, help='width of patch')
parser.add_argument('--patch_h', type=int, default=1024, help='height of patch')
parser.add_argument('--overlap_w', type=int, default=0, help='overlap width of patch')
parser.add_argument('--overlap_h', type=int, default=0, help='overlap height of patch')

parser.add_argument('--thumbnail_level', type=int, default=2, choices=[1, 2, 3, 4],
                    help='top level to catch WSI thumbnail images (larger is higher resolution)')
parser.add_argument('--use_otsu', action='store_false', help='use the Otsu algorithm to accelerate tiling patches or not')
parser.add_argument('--blank_rate_th', type=float, default=0.95, help='cut patches with a blank rate lower than this threshold')
parser.add_argument('--null_th', type=int, default=10, help='threshold to drop null patches (larger to drop more): 5, 10, 15, 20, ...')


def get_bg_mask(thumbnail, kernel_size=5):
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    _, th1 = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_OTSU)
    close_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(th1), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    _image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    image_open = (_image_open / 255.0).astype(np.uint8)
    return image_open

def isWhitePatch(patch, satThresh=5):
    patch = np.array(patch)
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:, :, 1]) < satThresh else False

def isNullPatch(patch, rgbThresh=10, null_rate=0.9):
    r, g, b = patch.split()
    r_arr = np.array(r, dtype=int)
    g_arr = np.array(g, dtype=int)
    b_arr = np.array(b, dtype=int)
    rgb_arr = (np.abs(r_arr - g_arr) + np.abs(r_arr - b_arr) + np.abs(g_arr - b_arr)) / 3
    rgb_sum = np.sum(rgb_arr < rgbThresh) / patch.size[0] / patch.size[1]
    return True if rgb_sum > null_rate else False

def mag_transfer(slide, sdpc_path, magnification, resolution, patch_w, patch_h, 
                 overlap_w, overlap_h, which2cut="magnification"):
    if which2cut == "magnification":
        if sdpc_path.endswith(".sdpc"):
            scan_mag = slide.readSdpc(sdpc_path).contents.picHead.contents.rate
        else:
            if "aperio.AppMag" in slide.properties.keys():
                scan_mag = float(slide.properties["aperio.AppMag"])
            else:
                scan_mag = 40.0
        zoomscale = magnification / scan_mag
    else:
        if sdpc_path.endswith(".sdpc"):
            scan_res = slide.readSdpc(sdpc_path).contents.picHead.contents.ruler
        else:
            scan_res = float(slide.properties["openslide.mpp-x"])
        zoomscale = scan_res / resolution
    try:
        scan_mag_step = slide.level_downsamples[1] / slide.level_downsamples[0]
        WSI_level = math.floor(math.log(1 / zoomscale, scan_mag_step))
        zoomrate = slide.level_downsamples[WSI_level]
    except:
        scan_mag_step = slide.level_downsample[1] / slide.level_downsample[0]
        WSI_level = math.floor(math.log(1 / zoomscale, scan_mag_step))
        zoomrate = slide.level_downsample[WSI_level]
    x_size = int(patch_w / zoomscale)
    y_size = int(patch_h / zoomscale)
    x_overlap = int(overlap_w / zoomscale)
    y_overlap = int(overlap_h / zoomscale)
    
    x_step, y_step = x_size - x_overlap, y_size - y_overlap
    x_offset = int(x_size / zoomrate)
    y_offset = int(y_size / zoomrate)
    return x_size, y_size, x_step, y_step, x_offset, y_offset, WSI_level

def img_detect(save_dir, slide, coord, bg_mask, marked_img, marked_img_red, thumbnail_mask, WSI_level, slide_x, slide_y, patch_w, patch_h,
               x_size, y_size, x_offset, y_offset, use_otsu, blank_rate_th, rgbThresh, border_color, geometries=None):
    x_start, y_start = coord[0], coord[1]
    mask_start_x = int(np.floor(x_start / slide_x * bg_mask.shape[1]))
    mask_start_y = int(np.floor(y_start / slide_y * bg_mask.shape[0]))
    mask_end_x = int(np.ceil((x_start + x_size) / slide_x * bg_mask.shape[1]))
    mask_end_y = int(np.ceil((y_start + y_size) / slide_y * bg_mask.shape[0]))
    mask = bg_mask[mask_start_y:mask_end_y, mask_start_x:mask_end_x]
    
    # 检查是否在geojson定义的区域内
    save_path = save_dir
    if geometries:
        patch_polygon = shapely.geometry.box(x_start, y_start, x_start + x_size, y_start + y_size)
        if any(geometry.intersects(patch_polygon) for geometry in geometries):
            save_path = os.path.join(save_dir, 'mark')
            os.makedirs(save_path, exist_ok=True)
            border_color = (0, 255, 0)  # 如果在标注区域内，用绿色边框
            # 在 thumbnail_mask 上绘制对应区域
            cv2.rectangle(thumbnail_mask, (mask_start_x, mask_start_y), (mask_end_x, mask_end_y), (0, 255, 0), -1)
    
    patch_save_path = os.path.join(save_path, '{}_{}_{}_{}.png'.format(x_start, y_start, x_start + x_size, y_start + y_size))

    img_flag = False
    if not use_otsu:
        img_flag = True
    elif mask.size > 0 and (np.sum(mask == 0) / mask.size) < blank_rate_th:
        img_flag = True
    
    if img_flag:
        try:
            img = slide.read_region((x_start, y_start), WSI_level, (x_offset, y_offset))
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert('RGB')
            img.thumbnail((patch_w, patch_h))
            if not isWhitePatch(img) and not isNullPatch(img, rgbThresh=rgbThresh):
                # 在 marked_img 上绘制绿色边框 (geojson 标注区域)
                cv2.rectangle(marked_img, (mask_start_x, mask_start_y), (mask_end_x, mask_end_y), (0, 255, 0), 2)
                # 在 marked_img_red 上绘制红色边框 (所有图像块)
                cv2.rectangle(marked_img_red, (mask_start_x, mask_start_y), (mask_end_x, mask_end_y), (255, 0, 0), 2)
                img.save(patch_save_path)
            return img  # 返回绘制矩形的图像
        except Exception as e:
            print(str(e))
            return None
    return None
def read_geojson(geojson_path):
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    geometries = [shape(feature['geometry']) for feature in geojson_data['features']]
    return geometries

def is_patch_in_geojson(patch_coords, geometries, slide_dimensions):
    x, y, w, h = patch_coords
    patch_polygon = shapely.geometry.box(x, y, x + w, y + h)  # 使用完整的命名空间
    for geometry in geometries:
        if geometry.intersects(patch_polygon) or patch_polygon.intersects(geometry) or patch_polygon.contains(geometry):
            return True
    return False

class Slide2Patch():
    def __init__(self, args):
        # general params.
        self.patch_w, self.patch_h = args.patch_w, args.patch_h
        self.overlap_w, self.overlap_h = args.overlap_w, args.overlap_h
        self.which2cut = args.which2cut
        self.magnification = args.magnification
        self.resolution = args.resolution
        self.save_dir = args.save_dir
        self.null_th = args.null_th
        self.thumbnail_level = args.thumbnail_level

        self.use_otsu = args.use_otsu
        self.blank_rate_th = args.blank_rate_th

    def save_patch(self, data_path, geojson_path=None, **kwargs):
        if geojson_path:
            geometries = read_geojson(geojson_path)
        else:
            geometries = None

        if data_path.split(".")[-1] == "sdpc":
            from sdpc.Sdpc import Sdpc
            slide = Sdpc(data_path)
            thumbnail = slide.read_region((0, 0), slide.level_count - self.thumbnail_level, slide.level_dimensions[-self.thumbnail_level])
        else:
            import openslide
            slide = openslide.open_slide(data_path)
            thumbnail = slide.get_thumbnail(slide.level_dimensions[-self.thumbnail_level])
            
        if isinstance(thumbnail, np.ndarray):
            pass
        else:
            thumbnail = np.array(thumbnail.convert('RGB'))
        
        # 生成背景掩码（bg_mask），用于确定哪些区域是空白的
        bg_mask = get_bg_mask(thumbnail)  # 添加这一行
        marked_img = thumbnail.copy()      # 绿色边框 (标注区域)
        marked_img_red = thumbnail.copy()  # 红色边框 (所有图像块)
        thumbnail_mask = np.zeros_like(thumbnail)  # 初始化掩码图像

        x_size, y_size, x_step, y_step, x_offset, y_offset, WSI_level = mag_transfer(slide,
                                                                                    data_path,
                                                                                    self.magnification, 
                                                                                    self.resolution, 
                                                                                    self.patch_w, 
                                                                                    self.patch_h,
                                                                                    self.overlap_w,
                                                                                    self.overlap_h,
                                                                                    self.which2cut)

        slide_x, slide_y = slide.level_dimensions[0]
        thumbnail_save_dir = os.path.join(self.save_dir, os.path.basename(data_path).split('.')[0], 'thumbnail')
        os.makedirs(thumbnail_save_dir, exist_ok=True)
        save_dir = os.path.join(self.save_dir, os.path.basename(data_path).split('.')[0])
        
        if geojson_path:
            mark_dir = os.path.join(save_dir, 'mark')
            os.makedirs(mark_dir, exist_ok=True)

        if not kwargs.get('coords'):
            coords = []
            for i in range(int(np.floor((slide_x - x_size) / x_step + 1))):
                for j in range(int(np.floor((slide_y - y_size) / y_step + 1))):
                    coords.append([i * x_step, j * y_step])

        pool = ThreadPoolExecutor(20)
        with tqdm(total=len(coords)) as pbar:
            for i, coord in enumerate(coords):
                future = pool.submit(img_detect, save_dir, slide, coord, bg_mask, marked_img, marked_img_red, thumbnail_mask, WSI_level, slide_x, slide_y,
                                    self.patch_w, self.patch_h, x_size, y_size, x_offset, y_offset, 
                                    self.use_otsu, self.blank_rate_th, self.null_th, (255, 0, 0), geometries)
                pbar.update(1)
        pool.shutdown()

        # 保存缩略图：所有图像块为红色边框
        thumbnail_path = os.path.join(thumbnail_save_dir, 'thumbnail.png')
        Image.fromarray(marked_img_red).save(thumbnail_path)

        # 保存缩略图：标注区域为绿色边框
        thumbnail_mark_path = os.path.join(thumbnail_save_dir, 'thumbnail_mark.png')
        Image.fromarray(marked_img).save(thumbnail_mark_path)

        # 保存掩码图像
        thumbnail_mask_path = os.path.join(thumbnail_save_dir, 'thumbnail_mask.png')
        Image.fromarray(cv2.cvtColor(thumbnail_mask, cv2.COLOR_BGR2RGB)).save(thumbnail_mask_path)

        print(f"Thumbnail with red markings saved at: {thumbnail_path}")
        print(f"Thumbnail with green markings saved at: {thumbnail_mark_path}")
        print(f"Thumbnail mask saved at: {thumbnail_mask_path}")   
    def cut_with_annotation(self, annotation_dir, sdpc_path, color_annotation):
        json_name = os.path.basename(sdpc_path).split(".")[0] + ".*"
        annotation_paths = glob.glob(os.path.join(annotation_dir, json_name))
        for annotation_path in annotation_paths:
            if annotation_path.split(".")[-1] in ANNOTATION_FORMAT:
                print("processing {}!".format(annotation_path))
                with open(annotation_path, 'r', encoding='UTF-8') as f:
                    label_dic = json.load(f)
                    coords, colors = self.getcoords(sdpc_path, label_dic, color_annotation)
                    self.save_patch(sdpc_path, coords, colors)
                return None

        print("annotation of {} does not exist!".format(sdpc_path))
        return None

    def getcoords(self, data_path, label_dic, color_annotation):
        if data_path.split(".")[-1] == "sdpc":
            from sdpc.Sdpc import Sdpc
            slide = Sdpc(data_path)
        else:
            import openslide
            slide = openslide.open_slide(data_path)
        x_size, y_size, x_step, y_step, _, _, _ = mag_transfer(slide,
                                                            data_path,
                                                            self.magnification, 
                                                            self.resolution, 
                                                            self.patch_w, 
                                                            self.patch_h,
                                                            self.overlap_w,
                                                            self.overlap_h,
                                                            self.which2cut)
        coords = []
        colors = []
        if 'GroupModel' in label_dic.keys():
            counters = label_dic['GroupModel']['Labels']
        else:
            counters = label_dic['LabelRoot']['LabelInfoList']
        for counter in counters:
            if 'Type' in counter.keys():
                counter_type = counter['Type']
            else:
                counter_type = counter['LabelInfo']['ToolInfor']
            if counter_type == "btn_brush" or counter_type == "btn_pen":
                if 'LineColor' in counter.keys():
                    color = counter['LineColor']
                else:
                    color = counter['LabelInfo']['PenColor']
                if 'Coordinates' in counter.keys():
                    Pointsx = [int(point.get('X')) for point in counter['Coordinates']]
                    Pointsy = [int(point.get('Y')) for point in counter['Coordinates']]
                else:
                    Points = list(zip(*[list(map(int, point.split(', '))) for point in counter['PointsInfo']['ps']]))
                    Ref_x, Ref_y, _, _ = counter['LabelInfo']['CurPicRect'].split(', ')
                    Ref_x, Ref_y = int(Ref_x), int(Ref_y)
                    std_scale = counter['LabelInfo']['ZoomScale']
                    Pointsx = []
                    Pointsy = []
                    for i in range(len(Points[0])):
                        Pointsx.append(int((Points[0][i] + Ref_x) / std_scale))
                        Pointsy.append(int((Points[1][i] + Ref_y) / std_scale))
                SPA_x, SPA_y = (min(Pointsx), min(Pointsy))
                SPB_x, SPB_y = (max(Pointsx), max(Pointsy))
                Pointslist = np.array([Pointsx, Pointsy]).transpose(1, 0)
                
                x0 = np.mean(np.array(Pointsx[:-1]))
                y0 = np.mean(np.array(Pointsy[:-1]))
                start_kx = -np.ceil((x0 - SPA_x - x_size / 2) / x_step)
                end_kx = np.ceil((SPB_x - x0 - x_size / 2) / x_step)
                start_ky = -np.ceil((y0 - SPA_y - y_size / 2) / y_step)
                end_ky = np.ceil((SPB_y - y0 - y_size / 2) / y_step)

                for x in range(int(start_kx), int(end_kx) + 1):
                    for y in range(int(start_ky), int(end_ky) + 1):
                        test_x_left = int(x0 + x * x_step - x_size / 2)
                        test_y_bottom = int(y0 + y * y_step - y_size / 2)
                        test_x_right = int(x0 + x * x_step + x_size / 2)
                        test_y_top = int(y0 + y * y_step + y_size / 2)
                        test_list = [(x0 + x * x_step, y0 + y * y_step),
                                    (test_x_left, test_y_top), 
                                    (test_x_left, test_y_bottom), 
                                    (test_x_right, test_y_top), 
                                    (test_x_right, test_y_bottom)]
                        for test_point in test_list:
                            if cv2.pointPolygonTest(Pointslist, test_point, False) >= 0:
                                coords.append([test_x_left, test_y_bottom])
                                colors.append(color)
                                continue
        if color_annotation:
            return coords, colors
        else:
            return coords, None
        

if __name__ == '__main__':    
    args = parser.parse_args()
    Auto_Build = Slide2Patch(args)

    _files = []
    for root, _, files in os.walk(args.data_dir):
        for file in files:
            file_format = file.split(".")[-1]
            if file_format in SLIDE_FORMAT:
                _files.append(os.path.join(root, file))
    _files = sorted(_files)

    for i, file in enumerate(_files):
        file_name = os.path.basename(file)
        geojson_filename = file_name.replace('.' + file_name.split('.')[-1], '.geojson')
        geojson_path = os.path.join(args.annotation_dir, geojson_filename)
        if os.path.exists(geojson_path):
            print(f"Processing with geojson: {geojson_path}")
            Auto_Build.save_patch(file, geojson_path=geojson_path)
        else:
            print(f"No geojson file found for {file_name}, processing without geojson.")
            Auto_Build.save_patch(file)

    print('-----------------* Patch Reading Finished *---------------------')