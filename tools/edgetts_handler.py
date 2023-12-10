import os
import edge_tts
import random
import asyncio
import pandas as pd
from datetime import datetime
from .base import BaseHandler
from loguru import logger


async def streaming_with_subtitles(text, audio_file, webvtt_file, voice, rate, volume, pitch) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume, pitch=pitch)
    submaker = edge_tts.SubMaker()
    with open(audio_file, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                submaker.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])

    with open(webvtt_file, "w", encoding="utf-8") as file:
        subtitles_info = submaker.generate_subs()
        file.write(subtitles_info)
    return subtitles_info


class EdgeTTSHandler(BaseHandler):
    def __init__(self, args, **kwargs):
        # output_dir='/data1/zjx/ai_webui/products/audio_tmp', 
        super().__init__(args)
        self.output_dir = self.handle_args.get("output_dir", "/tmp")

    def infer(self, tts_text_file=None, tts_text=None, voice='zh-CN-YunxiNeural', rate=0, volume=0, pitch=0, **kwargs):
        # 格式适配
        if rate >= 0:
            rate = f"+{rate}%"
        else:   
            rate = f"{rate}%"
        if volume >= 0:
            volume = f"+{volume}%"
        else:
            volume = f"{volume}%"
        if pitch >= 0:
            pitch = f"+{pitch}Hz"
        else:
            pitch = f"{pitch}Hz"

        if tts_text_file:
            tts_text_file_path = tts_text_file.name
            text = ""
            with open(tts_text_file_path, "r") as f:
                for line in f:
                    text += ' ' + line.rstrip()
            self.output_dir = os.path.dirname(tts_text_file_path)
            file_tag = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "_" + os.path.basename(tts_text_file_path).split('.')[0]
        elif tts_text:
            file_tag = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "_" + str(random.randint(0, 1000))
            text = tts_text
        
        audio_file = os.path.join(self.output_dir, file_tag + ".mp3")
        webvtt_file = os.path.join(self.output_dir, file_tag + ".vtt")
        loop = asyncio.new_event_loop()  # .get_event_loop_policy()
        asyncio.set_event_loop(loop)
        subtitles_info = loop.run_until_complete(streaming_with_subtitles(text, audio_file, webvtt_file, voice, rate, volume, pitch))
        loop.close()
        
        # 后处理
        proc_subtitles = []
        contents = subtitles_info.split("\r\n")
        for idx in range(1, len(contents)):
            if ' --> ' not in contents[idx]:
                continue
            start, end = contents[idx].split(' --> ')
            sentence = contents[idx+1].replace(' ', '')
            proc_subtitles.append({
                "start": start,
                "end": end,
                "sentence": sentence
            })
        df = pd.DataFrame(proc_subtitles)
        srt_file = webvtt_file.replace(".vtt", ".srt")
        excel_file = webvtt_file.replace(".vtt", ".xlsx")
        with open(srt_file, "w", encoding="utf-8") as f:
            for idx, row in enumerate(proc_subtitles):
                f.write(f"{idx+1}\n")
                f.write(f"{row['start']} --> {row['end']}\n")
                f.write(f"{row['sentence']}\n\n")
        
        df.to_excel(excel_file, index=False)
        file_list = [audio_file, webvtt_file, srt_file, excel_file]
        return audio_file, file_list

    def init_model(self):
        # Nothing to do
        logger.warning("## 无需初始化EdgeTTSHandler的模型！")

if __name__ == "__main__":
    text_file = "/data1/zjx/ai_webui/raw_materials/test.txt"
    text_type = "file"
    audio_file = "test.mp3" 
    webvtt_file = "test.vtt"
    voice = "zh-CN-YunxiNeural"
    edge_tts_handle = EdgeTTSHandle()
    # edge_tts_engine.streaming_with_subtitles(text_file, text_type, audio_file, webvtt_file, voice)
    # asyncio.run(amain())


