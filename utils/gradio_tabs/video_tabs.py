import gradio as gr
from utils.painter import mask_painter
from utils.misc import *


def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider
    operation_log = [("",""),("Select tracking finish frame {}.Try to click the image to add masks for tracking.".format(track_pause_number_slider),"Normal")]

    return video_state["painted_images"][track_pause_number_slider],interactive_state, operation_log, operation_log


def add_multi_mask(video_state, interactive_state, mask_dropdown):
    try:
        mask = video_state["masks"][video_state["select_frame_number"]]
        interactive_state["multi_mask"]["masks"].append(mask)
        interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        select_frame, _, _ = show_mask(video_state, interactive_state, mask_dropdown)
        operation_log = [("",""),("Added a mask, use the mask select for target tracking or inpainting.","Normal")]
    except:
        operation_log = [("Please click the image in step2 to generate masks.", "Error"), ("","")]
    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame, [[],[]], operation_log, operation_log

def clear_click(video_state, click_state):
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    operation_log = [("",""), ("Cleared points history and refresh the image.","Normal")]
    return template_frame, click_state, operation_log, operation_log

def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"]= []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("",""), ("Remove all masks. Try to add new masks","Normal")]
    return interactive_state, gr.update(choices=[],value=[]), operation_log, operation_log

def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    select_frame = video_state["origin_images"][video_state["select_frame_number"]]
    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        mask = interactive_state["multi_mask"]["masks"][mask_number]
        select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number+2)
    
    operation_log = [("",""), ("Added masks {}. If you want to do the inpainting with current masks, please go to step3, and click the Tracking button first and then Inpainting button.".format(mask_dropdown),"Normal")]
    return select_frame, operation_log, operation_log


def restart(mask_save=False):
    operation_log = [("",""), ("Try to upload your video and click the Get video info button to get started!", "Normal")]
    return {
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "inpaint_masks": None,
            "logits": None,
            "select_frame_number": 0,
            "fps": 30
        }, {
            "inference_times": 0,
            "negative_click_times" : 0,
            "positive_click_times": 0,
            "mask_save": mask_save,
            "multi_mask": {
                "mask_names": [],
                "masks": []
            },
            "track_end_number": None,
        }, [[],[]], None, None, None, \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),\
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "", \
        gr.update(visible=True, value=operation_log), gr.update(visible=False, value=operation_log)


def video_convertor_tab(video_clip_args, ai_handler):
    with gr.Tab(video_clip_args['name']):
        with gr.Row():
            video_file = gr.Video(height= video_clip_args['video_upload_win']['height'], 
                                  width= video_clip_args['video_upload_win']['width'], autoplay=True)
            srt_file = gr.components.File(label="上传字幕文件(非必选项)")
            with gr.Column():
                with gr.Row():
                    aspect_ratio_box = gr.Dropdown( video_clip_args['aspect_ratio_box']['choices'], value= video_clip_args['aspect_ratio_box']['value'], 
                                                    label="视频画面比例")
                    move_up_rate = gr.Slider(minimum=video_clip_args['move_up_rate']['minimum'], 
                                                maximum=video_clip_args['move_up_rate']['maximum'], 
                                                step=video_clip_args['move_up_rate']['step'], 
                                                value= video_clip_args['move_up_rate']['value'], 
                                                label="视频画面裁剪重心往上移的比例")
                add_noice = gr.Checkbox(value= video_clip_args['add_noice']['value'], label="给视频添加噪声(提升原创度, 但处理时间会长很多)")
                # video_segment_length = gr.Slider(minimum=1, maximum=10, step=1, label="视频片段时长, 单位为分钟") # step=1, , default=1
                
                with gr.Row():
                    with gr.Column():
                        subtitling_recognition = gr.Checkbox(value= video_clip_args['subtitling_recognition']['value'], label="字幕识别")
                        audio_speech_rate = gr.Slider(minimum= video_clip_args['audio_speech_rate']['minimum'], 
                                                        maximum= video_clip_args['audio_speech_rate']['maximum'], 
                                                        step= video_clip_args['audio_speech_rate']['step'], 
                                                        value= video_clip_args['audio_speech_rate']['value'], 
                                                        label="音频放慢率")
                        with gr.Row():
                            subtitling_language = gr.Radio(video_clip_args['subtitling_language']['choices'],
                                                            value= video_clip_args['subtitling_language']['value'], 
                                                            label="识别语言")
                            translate_engine = gr.Dropdown(choices=video_clip_args['translate_engine']['choices'],
                                                        value=video_clip_args['translate_engine']['value'], 
                                                        label="翻译引擎")
                              
        video_clip_b1 = gr.Button("视频剪辑", variant="primary")
        video_clip_outputs = gr.components.File(label="处理得到的文件")
        with gr.Row():
            collage_short_video = gr.Checkbox(value= video_clip_args['collage_short_video']['value'], label="一键成片")
            voice_role = gr.Dropdown(choices=video_clip_args['voice_role']['choices'],
                                    value= video_clip_args['voice_role']['value'],
                                    label="语音角色")
            bgm_name = gr.Dropdown(choices=video_clip_args['bgm_name']['choices'],
                                    value= video_clip_args['bgm_name']['value'],
                                    label="背景音乐")
            watermark = gr.Dropdown(choices=video_clip_args['watermark']['choices'],
                                    value= video_clip_args['watermark']['value'],
                                    label="水印")
            
        video_clip_b1.click(ai_handler.clip_video, 
                    inputs=[video_file, 
                            srt_file,
                            aspect_ratio_box, 
                            move_up_rate, 
                            add_noice, 
                            audio_speech_rate,
                            subtitling_recognition,
                            subtitling_language,
                            translate_engine,
                            collage_short_video,
                            voice_role,
                            bgm_name,
                            watermark], 
                    outputs=video_clip_outputs)


def video_inpainter_tab(args, ai_handler):
    with gr.Tab(args['name']):
        click_state = gr.State([[],[]])
        interactive_state = gr.State({
            "inference_times": 0,
            "negative_click_times" : 0,
            "positive_click_times": 0,
            "mask_save": args['interactive_state']['mask_save'],
            "multi_mask": {
                "mask_names": [],
                "masks": []
            },
            "track_end_number": None,
            }
        )

        video_state = gr.State(
            {
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "inpaint_masks": None,
            "logits": None,
            "select_frame_number": 0,
            "fps": 30
            }
        )

        with gr.Group(elem_classes="gr-monochrome-group"):
            with gr.Row():
                with gr.Accordion('ProPainter Parameters', open=False):
                    with gr.Row():
                        resize_ratio_number = gr.Slider(label='Resize ratio',
                                                minimum=0.01,
                                                maximum=1.0,
                                                step=0.01,
                                                value=1.0)
                        raft_iter_number = gr.Slider(label='Iterations for RAFT inference.',
                                                minimum=5,
                                                maximum=20,
                                                step=1,
                                                value=20,)
                    with gr.Row():
                        dilate_radius_number = gr.Slider(label='Mask dilation for video and flow masking.',
                                                minimum=0,
                                                maximum=10,
                                                step=1,
                                                value=8,)

                        subvideo_length_number = gr.Slider(label='Length of sub-video for long video inference.',
                                                minimum=40,
                                                maximum=200,
                                                step=1,
                                                value=80,)
                    with gr.Row():
                        neighbor_length_number = gr.Slider(label='Length of local neighboring frames.',
                                                minimum=5,
                                                maximum=20,
                                                step=1,
                                                value=10,)
                        
                        ref_stride_number = gr.Slider(label='Stride of global reference frames.',
                                                minimum=5,
                                                maximum=20,
                                                step=1,
                                                value=10,)
    
        with gr.Column():
            # input video
            gr.Markdown("## Step1: Upload video")
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):      
                    video_input = gr.Video(elem_classes="video")
                    extract_frames_button = gr.Button(value="Get video info", interactive=True, variant="primary") 
                with gr.Column(scale=2):
                    run_status = gr.HighlightedText(value=[("",""), ("Try to upload your video and click the Get video info button to get started!", "Normal")],
                                                    color_map={"Normal": "green", "Error": "red", "Clear clicks": "gray", "Add mask": "green", "Remove mask": "red"})
                    video_info = gr.Textbox(label="Video Info")
                    
            
            # add masks
            step2_title = gr.Markdown("---\n## Step2: Add masks", visible=False)
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    template_frame = gr.Image(type="pil",interactive=True, elem_id="template_frame", visible=False, elem_classes="image")
                    image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track start frame", visible=False)
                    track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
                with gr.Column(scale=2, elem_classes="jc_center"):
                    run_status2 = gr.HighlightedText(value=[("",""), ("Try to upload your video and click the Get video info button to get started!", "Normal")],
                                                    color_map={"Normal": "green", "Error": "red", "Clear clicks": "gray", "Add mask": "green", "Remove mask": "red"})
                    with gr.Row():
                        with gr.Column(scale=2, elem_classes="mask_button_group"):
                            clear_button_click = gr.Button(value="Clear clicks", interactive=True, visible=False)
                            remove_mask_button = gr.Button(value="Remove mask", interactive=True, visible=False, elem_classes="remove_button")
                            Add_mask_button = gr.Button(value="Add mask", interactive=True, visible=False, elem_classes="add_button")
                        point_prompt = gr.Radio(
                            choices=["Positive", "Negative"],
                            value="Positive",
                            label="Point prompt",
                            interactive=True,
                            visible=False,
                            min_width=100,
                            scale=1)
                    mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask selection", info=".", visible=False)
                
            # output video
            step3_title = gr.Markdown("---\n## Step3: Track masks and get the inpainting result", visible=False)
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    tracking_video_output = gr.Video(visible=False, elem_classes="video")
                    tracking_video_predict_button = gr.Button(value="1. Tracking", visible=False, elem_classes="margin_center")
                with gr.Column(scale=2):
                    inpaiting_video_output = gr.Video(visible=False, elem_classes="video")
                    inpaint_video_predict_button = gr.Button(value="2. Inpainting", visible=False, elem_classes="margin_center")

        # first step: get the video information 
        extract_frames_button.click(
            fn=ai_handler.sam_handler.get_frames_from_video,
            inputs=[
                video_input, video_state
            ],
            outputs=[video_state, video_info, template_frame,
                    image_selection_slider, track_pause_number_slider,point_prompt, clear_button_click, Add_mask_button, template_frame,
                    tracking_video_predict_button, tracking_video_output, inpaiting_video_output, remove_mask_button, 
                    inpaint_video_predict_button, step2_title, step3_title,mask_dropdown, run_status, run_status2
                    ]
        )   

        # second step: select images from slider
        image_selection_slider.release(fn=ai_handler.sam_handler.select_template, 
                                    inputs=[image_selection_slider, video_state, interactive_state], 
                                    outputs=[template_frame, video_state, interactive_state, run_status, run_status2], api_name="select_image")
        track_pause_number_slider.release(fn=get_end_number, 
                                    inputs=[track_pause_number_slider, video_state, interactive_state], 
                                    outputs=[template_frame, interactive_state, run_status, run_status2], api_name="end_image")
        
        # click select image to get mask using sam
        template_frame.select(
            fn=ai_handler.sam_handler.sam_refine,
            inputs=[video_state, point_prompt, click_state, interactive_state],
            outputs=[template_frame, video_state, interactive_state, run_status, run_status2]
        )

        # add different mask
        Add_mask_button.click(
            fn=add_multi_mask,
            inputs=[video_state, interactive_state, mask_dropdown],
            outputs=[interactive_state, mask_dropdown, template_frame, click_state, run_status, run_status2]
        )

        remove_mask_button.click(
            fn=remove_multi_mask,
            inputs=[interactive_state, mask_dropdown],
            outputs=[interactive_state, mask_dropdown, run_status, run_status2]
        )

        # tracking video from select image and mask
        tracking_video_predict_button.click(
            fn=ai_handler.tracker_handler.vos_tracking_video,
            inputs=[video_state, interactive_state, mask_dropdown],
            outputs=[tracking_video_output, video_state, interactive_state, run_status, run_status2]
        )

        # inpaint video from select image and mask
        inpaint_video_predict_button.click(
            fn=ai_handler.inpainting_handler.inpaint_video,
            inputs=[video_state, resize_ratio_number, dilate_radius_number, raft_iter_number, subvideo_length_number, neighbor_length_number, ref_stride_number, mask_dropdown],
            outputs=[inpaiting_video_output, run_status, run_status2]
        )

        # click to get mask
        mask_dropdown.change(
            fn=show_mask,
            inputs=[video_state, interactive_state, mask_dropdown],
            outputs=[template_frame, run_status, run_status2]
        )
        
        # clear input
        video_input.change(
            fn=restart,
            inputs=[],
            outputs=[ 
                video_state,
                interactive_state,
                click_state,
                tracking_video_output, inpaiting_video_output,
                template_frame,
                tracking_video_predict_button, image_selection_slider , track_pause_number_slider,point_prompt, clear_button_click, 
                Add_mask_button, template_frame, tracking_video_predict_button, tracking_video_output, inpaiting_video_output, remove_mask_button,inpaint_video_predict_button, step2_title, step3_title, mask_dropdown, video_info, run_status, run_status2
            ],
            queue=False,
            show_progress=False)
        
        video_input.clear(
            fn=restart,
            inputs=[],
            outputs=[ 
                video_state,
                interactive_state,
                click_state,
                tracking_video_output, inpaiting_video_output,
                template_frame,
                tracking_video_predict_button, image_selection_slider , track_pause_number_slider,point_prompt, clear_button_click, 
                Add_mask_button, template_frame, tracking_video_predict_button, tracking_video_output, inpaiting_video_output, remove_mask_button,inpaint_video_predict_button, step2_title, step3_title, mask_dropdown, video_info, run_status, run_status2
            ],
            queue=False,
            show_progress=False)
        
        # points clear
        clear_button_click.click(
            fn = clear_click,
            inputs = [video_state, click_state,],
            outputs = [template_frame,click_state, run_status, run_status2],
        )

        # set example
        gr.Markdown("## Examples")
        gr.Examples(
            examples=[os.path.join("./demo/video_inpainter/", test_sample) for test_sample in ["test-sample0.mp4", "test-sample1.mp4", "test-sample2.mp4", "test-sample3.mp4", "test-sample4.mp4"]],
            inputs=[video_input],
        )
