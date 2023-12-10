import os
import cv2
import anyconfig
import torch
import numpy as np
import re
import time
import torch.nn as nn
import torchvision
import logging
from os import path as osp
import json
import io
from PIL import Image, ImageOps
import zipfile
import math
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import pyplot as plt
from torchvision import transforms

import gradio as gr
import mdtex2html
import ffmpeg
import librosa
import random
from scipy import signal
from moviepy.editor import *
from moviepy.video.tools.drawing import color_gradient
import soundfile as sf


def parse_config(config: dict) -> dict:
    base_file_list = config.pop('base')
    base_config = {}
    for base_file in base_file_list:
        tmp_config = anyconfig.load(open(base_file, 'rb'))
        if 'base' in tmp_config:
            tmp_config = parse_config(tmp_config)
        anyconfig.merge(tmp_config, base_config)
        base_config = tmp_config
    anyconfig.merge(base_config, config)
    return base_config

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], [], None

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

def separate_video_and_audio(video_file, output_dir):
    """音视频分离"""
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_file)
    video_stream_file = os.path.join(output_dir, video_name.split('.')[0] + '_video_stream.mp4')
    audio_stream_file = os.path.join(output_dir, video_name.split('.')[0] + '_audio_stream.mp3')
    # 当存在相同文件名时，会强制覆盖
    get_video_stream_cmd = f"ffmpeg -i {video_file} -vcodec copy -an {video_stream_file} -y"
    get_audio_stream_cmd = f"ffmpeg -i {video_file} {audio_stream_file} -y"
    os.system(get_video_stream_cmd)
    os.system(get_audio_stream_cmd)
    return video_stream_file, audio_stream_file

def get_meta_from_video(input_video):
    """获取视频第一帧"""
    if input_video is None:
        return None, None, None, ""

    print("获取输入视频的元信息")
    cap = cv2.VideoCapture(input_video)

    _, first_frame = cap.read()
    cap.release()
  
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    return first_frame, first_frame, first_frame, ""


def process_video(video_file, output_dir, aspect_ratio=16/9, move_up_rate=0, add_noice=False, disturb_frames=True):
    """process video"""
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_file)
    output_video_file = os.path.join(output_dir, video_name.split('.')[0] + '_soundless.mp4')
    
    cap = cv2.VideoCapture(video_file)
    # 调用cv2方法获取cap的视频帧（帧：每秒多少张图片）
    fps = cap.get(cv2.CAP_PROP_FPS)
    rval, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if width / height <= aspect_ratio:
        obj_h = min(height, int(width / aspect_ratio))
        h1 = max(0, int((height-obj_h)/2 - obj_h*move_up_rate))
        h2 = h1 + obj_h
        obj_w = width
    else:
        obj_w = min(width, int(height * aspect_ratio))
        w1 = int((width-obj_w)/2)
        w2 = w1 + obj_w
        obj_h = height
    # 定义编码格式mpge-4, TODO：如分辨率不能满足要求，则需要考虑其他格式
    # 一种视频格式，参数搭配固定，不同的编码格式对应不同的参数
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # proc_frame = frame[h1:h2]
    out_video = cv2.VideoWriter(output_video_file, fourcc, fps, (obj_w, obj_h))
    # 循环使用cv2的read()方法读取视频帧
    count = 0
    while rval:
        count += 1
        if count == 1:
            continue
        if disturb_frames and (count+10) % 90 == 0: # 抽帧
            continue
        if width / height <= aspect_ratio:
            proc_frame = frame[h1:h2]
        else:
            proc_frame = frame[:, w1:w2]
        
        # 加高斯噪声
        if add_noice:
            proc_frame = np.clip(proc_frame.astype(np.float) + np.random.random(proc_frame.shape) * np.random.uniform(0.8, 3), 0, 255).astype(np.uint8)
        
        # 使用VideoWriter类中的write(frame)方法，将图像帧写入视频文件
        out_video.write(proc_frame)
        if disturb_frames and count % 90 == 0:
            out_video.write(proc_frame)  # 补帧
        rval, frame = cap.read()

    # 释放窗口
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    return output_video_file


def get_duration(file_path):
    info = ffmpeg.probe(file_path)
    duration = float(info['format']['duration'])
    return duration


def reduce_audio_noice(audio_file, cutoff_freq=3000):
    output_denoice_audio_file = audio_file.replace('.mp3', '_denoice.wav')
    audio, sr = librosa.load(audio_file, sr=None)
    # 低通滤波器可以去除音频中的高频部分，包括尖锐刺耳的噪音和转音
    # cutoff_freq = 3000  # 替换为你想要的截止频率
    # b = librosa.filters.lowpass(sr, cutoff_freq)
    b, a = signal.butter(4, cutoff_freq, btype='lowpass', fs=sr)
    filtered_audio = signal.lfilter(b, a, audio)
    # librosa.output.write_wav(output_denoice_audio_file, filtered_audio, sr)
    sf.write(output_denoice_audio_file, filtered_audio, sr)
    return output_denoice_audio_file


def add_watermark(video_file, watermark_tag, img_dir='/data1/zjx/video_transport/raw_materials/watermark'):
    watermark_image_path = os.path.join(img_dir, watermark_tag+'.png')
    video = VideoFileClip(video_file)
    watermark = ImageClip(watermark_image_path).resize(height=25)
    watermark = watermark.set_duration(video.duration).margin(opacity=0.5).set_position(('right', 'bottom'))

    # 将文字水印与视频合并
    video_with_watermark = CompositeVideoClip([video, watermark])
    video_with_watermark.duration = video.duration  # Set the duration attribute
    # 将合并后的视频保存到文件中
    video_with_watermark.write_videofile(video_file, codec="libx264", audio_codec="aac")
    return


def merge_video_audio(video_file_list, audio_file, object_duration, output_video_file, 
                      audio_align_method='change_speech',
                      mix_audios=False):
    audio = AudioFileClip(audio_file)
    output_video_files = [output_video_file]
    video_list = []
    for video_file in video_file_list:
        video_list.append(VideoFileClip(video_file))
    video = concatenate_videoclips(video_list)
    object_duration = round(object_duration)
    
    # 处理音频
    if abs(audio.duration - object_duration) > 0.7:
        if audio_align_method == 'change_speech':
            audio_multiplier = audio.duration / object_duration
            # 使用速度特效函数(speedx)将音频按指定倍数加速
            audio = audio.fx(vfx.speedx, audio_multiplier)

        else:
            if audio.duration < object_duration:
                N = int(object_duration / audio.duration) + 1
                audio = concatenate_audioclips([audio] * N)
                
            audio = audio.subclip(0, object_duration)
        
    
    # 处理视频
    if abs(video.duration - object_duration) > 0.7:
        video_multiplier = video.duration / object_duration
        video = video.fx(vfx.speedx, video_multiplier)
    
    
    if mix_audios: # 保留原视频的音频
        new_audio = CompositeAudioClip([video.audio, audio])
        merge_video = video.set_audio(new_audio)  # .set_duration(video.duration)
        
    else:
        merge_video = video.set_audio(audio)
    
    merge_video.write_videofile(output_video_file, codec="libx264", audio_codec="aac")

    return output_video_files

def get_click_prompt(click_stack, point):
    
    click_stack[0].append(point["coord"])
    click_stack[1].append(point["mode"])

    prompt = {
        "points_coord": click_stack[0],
        "points_mode": click_stack[1],
        "multimask": "True",
    }

    return prompt


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

initialized_logger = {}
def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    """Get the root logger.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.
    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False

    if log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        # file_handler = logging.FileHandler(log_file, 'w')
        file_handler = logging.FileHandler(log_file, 'a') #Shangchen: keep the previous log
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger


IS_HIGH_VERSION = [int(m) for m in list(re.findall(r"^([0-9]+)\.([0-9]+)\.([0-9]+)([^0-9][a-zA-Z0-9]*)?(\+git.*)?$",\
    torch.__version__)[0][:3])] >= [1, 12, 0]

def gpu_is_available():
    if IS_HIGH_VERSION:
        if torch.backends.mps.is_available():
            return True
    return True if torch.cuda.is_available() and torch.backends.cudnn.is_available() else False

def get_device(gpu_id=None):
    if gpu_id is None:
        gpu_str = ''
    elif isinstance(gpu_id, int):
        gpu_str = f':{gpu_id}'
    else:
        raise TypeError('Input should be int value.')

    if IS_HIGH_VERSION:
        if torch.backends.mps.is_available():
            return torch.device('mps'+gpu_str)
    return torch.device('cuda'+gpu_str if torch.cuda.is_available() and torch.backends.cudnn.is_available() else 'cpu')


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def read_dirnames_under_root(root_dir):
    dirnames = [
        name for i, name in enumerate(sorted(os.listdir(root_dir)))
        if os.path.isdir(os.path.join(root_dir, name))
    ]
    print(f'Reading directories under {root_dir}, num: {len(dirnames)}')
    return dirnames


class TrainZipReader(object):
    file_dict = dict()

    def __init__(self):
        super(TrainZipReader, self).__init__()

    @staticmethod
    def build_file_dict(path):
        file_dict = TrainZipReader.file_dict
        if path in file_dict:
            return file_dict[path]
        else:
            file_handle = zipfile.ZipFile(path, 'r')
            file_dict[path] = file_handle
            return file_dict[path]

    @staticmethod
    def imread(path, idx):
        zfile = TrainZipReader.build_file_dict(path)
        filelist = zfile.namelist()
        filelist.sort()
        data = zfile.read(filelist[idx])
        #
        im = Image.open(io.BytesIO(data))
        return im


class TestZipReader(object):
    file_dict = dict()

    def __init__(self):
        super(TestZipReader, self).__init__()

    @staticmethod
    def build_file_dict(path):
        file_dict = TestZipReader.file_dict
        if path in file_dict:
            return file_dict[path]
        else:
            file_handle = zipfile.ZipFile(path, 'r')
            file_dict[path] = file_handle
            return file_dict[path]

    @staticmethod
    def imread(path, idx):
        zfile = TestZipReader.build_file_dict(path)
        filelist = zfile.namelist()
        filelist.sort()
        data = zfile.read(filelist[idx])
        file_bytes = np.asarray(bytearray(data), dtype=np.uint8)
        im = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # im = Image.open(io.BytesIO(data))
        return im


# ###########################################################################
# Data augmentation
# ###########################################################################


def to_tensors():
    return transforms.Compose([Stack(), ToTorchFormatTensor()])


class GroupRandomHorizontalFlowFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img_group, flowF_group, flowB_group):
        v = random.random()
        if v < 0.5:
            ret_img = [
                img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group
            ]
            ret_flowF = [ff[:, ::-1] * [-1.0, 1.0] for ff in flowF_group]
            ret_flowB = [fb[:, ::-1] * [-1.0, 1.0] for fb in flowB_group]
            return ret_img, ret_flowF, ret_flowB
        else:
            return img_group, flowF_group, flowB_group


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group],
                                axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(
                pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


# ###########################################################################
# Create masks with random shape
# ###########################################################################


def create_random_shape_with_random_motion(video_length,
                                           imageHeight=240,
                                           imageWidth=432):
    # get a random shape
    height = random.randint(imageHeight // 3, imageHeight - 1)
    width = random.randint(imageWidth // 3, imageWidth - 1)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8) / 10

    region = get_random_shape(edge_num=edge_num,
                              ratio=ratio,
                              height=height,
                              width=width)
    region_width, region_height = region.size
    # get random position
    x, y = random.randint(0, imageHeight - region_height), random.randint(
        0, imageWidth - region_width)
    velocity = get_random_velocity(max_speed=3)
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
    masks = [m.convert('L')]
    # return fixed masks
    if random.uniform(0, 1) > 0.5:
        return masks * video_length
    # return moving masks
    for _ in range(video_length - 1):
        x, y, velocity = random_move_control_points(x,
                                                    y,
                                                    imageHeight,
                                                    imageWidth,
                                                    velocity,
                                                    region.size,
                                                    maxLineAcceleration=(3,
                                                                         0.5),
                                                    maxInitSpeed=3)
        m = Image.fromarray(
            np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
        masks.append(m.convert('L'))
    return masks


def create_random_shape_with_random_motion_zoom_rotation(video_length, zoomin=0.9, zoomout=1.1, rotmin=1, rotmax=10, imageHeight=240, imageWidth=432):
    # get a random shape
    assert zoomin < 1, "Zoom-in parameter must be smaller than 1"
    assert zoomout > 1, "Zoom-out parameter must be larger than 1"
    assert rotmin < rotmax, "Minimum value of rotation must be smaller than maximun value !"
    height = random.randint(imageHeight//3, imageHeight-1)
    width = random.randint(imageWidth//3, imageWidth-1)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8)/10
    region = get_random_shape(
        edge_num=edge_num, ratio=ratio, height=height, width=width)
    region_width, region_height = region.size
    # get random position
    x, y = random.randint(
        0, imageHeight-region_height), random.randint(0, imageWidth-region_width)
    velocity = get_random_velocity(max_speed=3)
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
    masks = [m.convert('L')]
    # return fixed masks
    if random.uniform(0, 1) > 0.5:
        return masks*video_length  # -> directly copy all the base masks
    # return moving masks
    for _ in range(video_length-1):
        x, y, velocity = random_move_control_points(
            x, y, imageHeight, imageWidth, velocity, region.size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3)
        m = Image.fromarray(
            np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        ### add by kaidong, to simulate zoon-in, zoom-out and rotation
        extra_transform = random.uniform(0, 1)
        # zoom in and zoom out
        if extra_transform > 0.75:
            resize_coefficient = random.uniform(zoomin, zoomout)
            region = region.resize((math.ceil(region_width * resize_coefficient), math.ceil(region_height * resize_coefficient)), Image.NEAREST)
            m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
            region_width, region_height = region.size
        # rotation
        elif extra_transform > 0.5:
            m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
            m = m.rotate(random.randint(rotmin, rotmax))
            # region_width, region_height = region.size
        ### end
        else:
            m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
        masks.append(m.convert('L'))
    return masks


def get_random_shape(edge_num=9, ratio=0.7, width=432, height=240):
    '''
      There is the initial point and 3 points per cubic bezier curve.
      Thus, the curve will only pass though n points, which will be the sharp edges.
      The other 2 modify the shape of the bezier curve.
      edge_num, Number of possibly sharp edges
      points_num, number of points in the Path
      ratio, (0, 1) magnitude of the perturbation from the unit circle,
    '''
    points_num = edge_num*3 + 1
    angles = np.linspace(0, 2*np.pi, points_num)
    codes = np.full(points_num, Path.CURVE4)
    codes[0] = Path.MOVETO
    # Using this instead of Path.CLOSEPOLY avoids an innecessary straight line
    verts = np.stack((np.cos(angles), np.sin(angles))).T * \
        (2*ratio*np.random.random(points_num)+1-ratio)[:, None]
    verts[-1, :] = verts[0, :]
    path = Path(verts, codes)
    # draw paths into images
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='black', lw=2)
    ax.add_patch(patch)
    ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.axis('off')  # removes the axis to leave only the shape
    fig.canvas.draw()
    # convert plt images into numpy images
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape((fig.canvas.get_width_height()[::-1] + (3,)))
    plt.close(fig)
    # postprocess
    data = cv2.resize(data, (width, height))[:, :, 0]
    data = (1 - np.array(data > 0).astype(np.uint8))*255
    corrdinates = np.where(data > 0)
    xmin, xmax, ymin, ymax = np.min(corrdinates[0]), np.max(
        corrdinates[0]), np.min(corrdinates[1]), np.max(corrdinates[1])
    region = Image.fromarray(data).crop((ymin, xmin, ymax, xmax))
    return region


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration
    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    return (speed, angle)


def get_random_velocity(max_speed=3, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)


def random_move_control_points(X,
                               Y,
                               imageHeight,
                               imageWidth,
                               lineVelocity,
                               region_size,
                               maxLineAcceleration=(3, 0.5),
                               maxInitSpeed=3):
    region_width, region_height = region_size
    speed, angle = lineVelocity
    X += int(speed * np.cos(angle))
    Y += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(lineVelocity,
                                     maxLineAcceleration,
                                     dist='guassian')
    if ((X > imageHeight - region_height) or (X < 0)
            or (Y > imageWidth - region_width) or (Y < 0)):
        lineVelocity = get_random_velocity(maxInitSpeed, dist='guassian')
    new_X = np.clip(X, 0, imageHeight - region_height)
    new_Y = np.clip(Y, 0, imageWidth - region_width)
    return new_X, new_Y, lineVelocity

# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt