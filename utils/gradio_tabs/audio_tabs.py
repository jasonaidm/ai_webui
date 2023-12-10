import gradio as gr


# 语音识别
def asr_tab(asr_args, ai_handler):
        with gr.Tab(asr_args['name']):
            audio_file = gr.Audio(type="filepath", autoplay=True)
            # text_input = gr.Textbox(label="Question")
            audio_asr_text = gr.Textbox(label="text")
            audio_asr_files = gr.components.File(label="音频识别输出的文件")
            audio_b1 = gr.Button("Recognize speech", variant="primary")
            
            audio_b1.click(ai_handler.asr_infer, inputs=[audio_file], outputs=[audio_asr_text, audio_asr_files])

def tts_tab(tts_args, ai_handler):
        with gr.Tab(tts_args['name']):
            model_type = gr.Dropdown(label="模型类型", choices=tts_args['model_type']['choices'], value=tts_args['model_type']['value'])
            with gr.Row():
                tts_text_file = gr.components.File(label="上传txt文本")
                tts_text = gr.Textbox(label="text")
                
            tts_voice = gr.Dropdown(label="选择发音人", choices=tts_args['tts_voice']['choices'], value=tts_args['tts_voice']['value'])
            tts_rate = gr.Slider(label="语速", minimum=tts_args['tts_rate']['minimum'], max=tts_args['tts_rate']['maximum'], value=tts_args['tts_rate']['value'], step=tts_args['tts_rate']['step'])
            tts_volume = gr.Slider(label="音量", minimum=tts_args['tts_volume']['minimum'], max=tts_args['tts_volume']['maximum'], value=tts_args['tts_volume']['value'], step=tts_args['tts_volume']['step'])
            tts_pitch = gr.Slider(label="语调", minimum=tts_args['tts_pitch']['minimum'], max=tts_args['tts_pitch']['maximum'], value=tts_args['tts_pitch']['value'], step=tts_args['tts_pitch']['step'])
            tts_b1 = gr.Button("合成", variant="primary")
            # 输出可下载文件
            tts_audio_file = gr.Audio(type="filepath", autoplay=True, label="合成音频")
            tts_out_files = gr.components.File(label="合成文件下载列表")
            
            tts_b1.click(ai_handler.tts_infer, 
                            inputs=[tts_text_file, tts_text, tts_voice, tts_rate, tts_volume, tts_pitch],
                            outputs=[tts_audio_file, tts_out_files]
                            )