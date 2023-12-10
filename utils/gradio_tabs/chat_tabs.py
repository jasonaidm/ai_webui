import gradio as gr
from utils import misc

def chat_tab(chatbot_args, tts_args, ai_handler):
    # 聊天机器人  text_args['']
    with gr.Tab(chatbot_args['name']):
        gr.Chatbot.postprocess = misc.postprocess
        chatbot = gr.Chatbot(height=chatbot_args['chatbot_win']['height'],)   
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=5).style(
                        container=False)
                    
                with gr.Column(min_width=32, scale=1):
                    text_submitBtn = gr.Button("发送文字聊天信息", variant="primary")
                # speech_file = gr.Audio(type="filepath", min_width=25, autoplay=False, label="语音文件or录音")
                speech_file = gr.Audio(label="Audio", sources="microphone", type="filepath", elem_id='audio')
                speech_submitBtn = gr.Button("发送语音信息")
                chat_replay_audio = gr.Audio(type="filepath", autoplay=True, label="AI回话内容")
                
            with gr.Column(scale=1):
                llm_model_type = gr.Dropdown(choices=chatbot_args['llm_model_type']['choices'],
                                                value=chatbot_args['llm_model_type']['value'],  
                                                label="大语言模型")
                chat_tts_voice = gr.Dropdown(label="选择发音人", choices=tts_args['tts_voice']['choices'], value=tts_args['tts_voice']['value'])
                text_emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.95, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.85, step=0.01, label="Temperature", interactive=True)
                
        history = gr.State([])
        past_key_values = gr.State(None)

        text_submitBtn.click(ai_handler.chatglm_handler.stream_chat, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                        [chatbot, history, past_key_values], show_progress=True)
        text_submitBtn.click(misc.reset_user_input, [], [user_input])
        
        # speech_submitBtn.click(ai_handler.llm_infer, [llm_model_type, user_input, speech_file, history, max_length, top_p, temperature],
        #                 [chat_replay_audio, history], show_progress=True)
        speech_submitBtn.click(ai_handler.audio_chat, [llm_model_type, chat_tts_voice, speech_file, history, max_length, top_p, temperature],
                        [chat_replay_audio, history], show_progress=True)
        
        # speech_submitBtn.click(fn=action, inputs=speech_submitBtn, outputs=speech_submitBtn).\
        #   then(fn=lambda: None, _js=click_js()).\
        #   then(fn=check_btn, inputs=speech_submitBtn).\
        #   success(fn=ai_handler.llm_infer, inputs=[llm_model_type, user_input, speech_file, history, max_length, top_p, temperature], outputs=[chat_replay_audio, history])
        
        
        text_emptyBtn.click(misc.reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)
