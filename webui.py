import os
import argparse
import anyconfig
from tools import AIWrapper

from utils.misc import parse_config
from utils.gradio_utils import *
import gradio as gr
from gradio.themes.utils import colors
from utils.gradio_tabs import *


os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

def launch_webui(yaml_file, device_ids=None, share=None, **kwargs):
    if device_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
    with open(yaml_file, 'rb') as f:
        args = anyconfig.load(f)
        if 'base' in args:
            args = parse_config(args)

    video_convertor_args = args.get('video_convertor', {})
    video_inpainter_args = args.get('video_inpainter', {})
    segmentation_args = args.get('segmentation_task', {})
    chat_args = args.get('chatbot', {})
    visualchat_args = args.get('visualchat', {})
    asr_args = args.get('asr_task', {})
    tts_args = args.get('tts_task', {})
    
    # 初始化AI引擎
    ai_handler = AIWrapper(args)
    
    # seafoam = gradio_utils.Seafoam()
    theme=gr.themes.Soft(primary_hue=colors.gray, neutral_hue=colors.neutral)
    with gr.Blocks(theme=theme) as web:
        # gr.Markdown(args["home_name"])
        gr.HTML(f"""<h1 align="center">{args["home_desc"]}</h1>""")
        # Process text, audio or video file using this web
        
        # 视频剪辑
        if video_convertor_args.get('switch'):
            video_convertor_tab(video_convertor_args, ai_handler)

        # 视频修复
        if video_inpainter_args.get('switch'):
            video_inpainter_tab(video_inpainter_args, ai_handler)

        # 图像分割
        if segmentation_args.get('switch'):
            sam_tab(segmentation_args, ai_handler)
        
        # 聊天问答
        if chat_args.get('switch'):
            chat_tab(chat_args, tts_args, ai_handler)
        # 多模态问答
        if visualchat_args.get('switch'):
            visualchat_tab(visualchat_args, ai_handler)

        # 语音识别
        if asr_args.get('switch'):
            asr_tab(asr_args, ai_handler)
        
        # 语音合成
        if tts_args.get('switch'):
            tts_tab(tts_args, ai_handler)

    web.queue().launch(share=share, server_name=args["server_name"], server_port=args["server_port"])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '-c', '-cfg', '--yaml_file', type=str, default='configs/webui_configs.yml', 
                        help='yaml config file'
                        )
    parser.add_argument('-d', '--device_ids', type=str, default=None, help='device ids')
    parser.add_argument('-s', "--share", action="store_true", help='whether public url')
    opt = parser.parse_args()
    return opt
 

if __name__ == "__main__":
    opt = parse_opt()
    launch_webui(**opt.__dict__)
