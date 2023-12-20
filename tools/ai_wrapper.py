import os
from .chatglm_handler import ChatGLMHandler
from .visualchat_handler import VisualChatHandler
from .whisper_handler import WhisperHandler
from .edgetts_handler import EdgeTTSHandler
from .gpt_handler import GPTHandler
from .fastsam_handler import FastSAMHandler
from .sam_handler import SAMHandler
from .tracker_handler import TrackerHandler
from .inpainting_handler import InpaintingHandler
from moviepy.editor import *
from utils.misc import *


class AIWrapper:
    """
    AIWrapper class
    """

    def __init__(self, args, **kwargs):
        """
        Initialize AIWrapper class
        :param args: arguments used to initialize AI Handlers
        """
        self.args = args
        self.chatglm_handler = ChatGLMHandler(args)
        self.visualchat_handler = VisualChatHandler(args)
        self.gpt_handler = GPTHandler(args)
        self.whisper_handler = WhisperHandler(args)
        self.edgetts_handler = EdgeTTSHandler(args)
        self.fastsam_handler = FastSAMHandler(args)
        self.sam_handler = SAMHandler(args)
        self.inpainting_handler = InpaintingHandler(args)
        self.tracker_handler = TrackerHandler(args)
        

    def clip_video(self, video_file, subtitling_file, aspect_ratio='16/9', move_up_rate=0.12, add_noice=False, 
                   audio_speech_rate=1, subtitling_recognition=True, subtitling_language='英语', translate_engine='chatgpt',
                   collage_short_video=False, voice_role=None, bgm_name=None, watermark='川陀', **kwargs):
        """视频剪辑 """
        video_tag = os.path.basename(video_file).split('.')[0]
        output_dir = os.path.dirname(video_file)
        if isinstance(aspect_ratio, str): aspect_ratio = eval(aspect_ratio)
        video_stream_file, audio_stream_file = separate_video_and_audio(video_file, output_dir)
        video_soundless_file = process_video(video_stream_file, output_dir, aspect_ratio, move_up_rate, add_noice)
        output_files = [video_soundless_file]
        # add bgm
        video_bgm_file = os.path.join(output_dir, f'{video_tag}_bgm.mp4')
        object_duration = VideoFileClip(video_soundless_file).duration
        bgm_file = os.path.join('./demo/bgm', f'{bgm_name}.mp3')
        
        out_files = merge_video_audio([video_soundless_file], bgm_file, object_duration, video_bgm_file, audio_align_method='concat_clip')
        output_files.extend(out_files)
            
        # 提取字幕
        srt_file = os.path.join(output_dir, f'{video_tag}_cn.srt')  # 字幕文件路径
        if subtitling_recognition or collage_short_video:
            if subtitling_file:
                text = ''
                subtitling_file = subtitling_file.name
                with open(subtitling_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if (len(line) <= 3) or (' --> ' in line):
                            continue
                        text += ' ' + line
                            
            else:
                if audio_speech_rate != 1.:
                    audio_speech_rate_tag = f"{audio_speech_rate:0.1f}"
                    audio_stream_file2 =  audio_stream_file.replace('.mp3', audio_speech_rate_tag.replace('.', '-') + '.mp3')
                    os.system(f'ffmpeg -y -i {audio_stream_file} -filter:a "atempo={audio_speech_rate_tag}" {audio_stream_file2}')
                else:
                    audio_stream_file2 =  audio_stream_file
                # 在ASR环节进行翻译，但准确性非常差，尽量不要用 , language=subtitling_language
                asr_result= self.whisper_handler.infer(audio_stream_file2)
                # 使用本地llm翻译  chatglm3翻译遇到的问题，中文里夹带着英文单词或短句是非常普遍的现象
                if asr_result['language'] == 'en':
                    en_srt_file = srt_file.replace('_cn.srt', '_en.srt')
                    with open(en_srt_file, 'w', encoding='utf-8') as f:
                        f.write(f"1\n00:00:01,666 --> 00:01:01,888\n") 
                        f.write(asr_result['text'])
                    output_files.append(en_srt_file)
                    prompt = '请帮我把下面的英文段落翻译成中文，翻译人名时，风格尽量偏向网络语言。如"Hey bro"可翻译为"小黑", "Lazy"翻译为"懒懒", "sanguine"翻译为"丧彪"等等\n'
                    input_text = prompt + asr_result['text']
                    try:
                        text, _ = self.llm_infer(model_type=translate_engine, text = input_text, max_length=8192, top_p=0.9, temperature=0.9)
                    except:
                        return output_files
                    # input_text = "请把下面中文段落中出现的英文单词或句子翻译成中文，本身是中文的保持不变：\n" + text
                    # text, _ = self.llm_infer(model_type='chatglm', text = input_text, max_length=8192, top_p=0.9, temperature=0.9)
                    
                else:
                    text = asr_result['text']
            
            with open(srt_file, 'w', encoding='utf-8') as f:
                f.write(f"1\n00:00:01,666 --> 00:01:01,888\n") 
                f.write(text)
            output_files.append(srt_file)
        # 一键成片
        if collage_short_video:
            video_collage_file = os.path.join(output_dir, f'{video_tag}_collage.mp4')
            # TODO: 视频时长的制定策略。目前是基于audio_speech_rate的倒数
            tts_speech_rate = int(1. / audio_speech_rate * 100 - 100)
            audio_file, file_list = self.tts_infer(None, text, voice_role, tts_speech_rate, 35, -5)
            
            # 音频增大音量和去噪
            output_denoice_audio_file = reduce_audio_noice(audio_file, cutoff_freq=3000)
            # 增大音量
            audio = AudioFileClip(output_denoice_audio_file)
            audio = audio.volumex(1.6)  # 可以调整音量增益值
            audio.write_audiofile(audio_file)
            
            # 音视频合并 -- 时长跟tts输出的音频对齐
            object_duration = get_duration(audio_file)
            merge_files = merge_video_audio([video_bgm_file], audio_file, object_duration, video_collage_file, mix_audios=True)
            # add_watermark(video_collage_file, watermark)
            output_files += merge_files + file_list
        return output_files
    
    def stream_chat(self, model_type, text, chatbot, max_length, top_p, temperature, history, past_key_values, **kwargs):
        if model_type == 'chatglm':
            return self.chatglm_handler.stream_chat(text, chatbot, max_length, top_p, temperature, history, past_key_values)
        
        else:
            raise NotImplementedError(f"{model_type}模型还没有部署！")
    
    def llm_infer(self, model_type, text=None, audio_file=None, history=[], max_length=2048, top_p=90, temperature=95, **kwargs):
        if model_type == 'chatglm':
            if audio_file:  # todo: 实现语音对话功能
                asr_result = self.whisper_handler.infer(audio_file)
                text = asr_result['text']
            
            response, history = self.chatglm_handler.infer(text, history, max_length, top_p, temperature)
            if audio_file:
                response, _ = self.edgetts_handler.infer(None, response)
            
            return response, history
        
        elif model_type == 'chatgpt':
            if audio_file:  # todo: 实现语音对话功能
                asr_result = self.whisper_handler.infer(audio_file)
                text = asr_result['text']
                    
            response, history = self.gpt_handler.infer(text)
            if audio_file:
                response, _ = self.gpt_handler.infer(None, response)
            
            return response, history
        
        else:
            raise NotImplementedError(f"{model_type}模型还没有部署！")

    def audio_chat(self, model_type, chat_tts_voice, audio_file=None, history=[], max_length=2048, top_p=90, temperature=95, **kwargs):
        if model_type == 'chatglm':
            if audio_file:  # todo: 实现语音对话功能
                asr_result = self.whisper_handler.infer(audio_file)
                text = asr_result['text']
            
            response, history = self.chatglm_handler.infer(text, history, max_length, top_p, temperature)
            if audio_file:
                response, _ = self.edgetts_handler.infer(None, response, voice=chat_tts_voice)
            
            return response, history
            
        else:
            raise NotImplementedError(f"{model_type}模型还没有部署！")

    def tts_infer(self, tts_text_file, tts_text, voice, rate, volume, pitch):
        audio_file, file_list = self.edgetts_handler.infer(tts_text_file, tts_text, voice, rate, volume, pitch)
        return audio_file, file_list
    
    def asr_infer(self, audio_stream_file):
        result = self.whisper_handler.infer(audio_stream_file)
        text = result['text']   
        excel_file = result['segments']
        return text, excel_file
    