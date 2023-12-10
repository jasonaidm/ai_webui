import gradio as gr

examples = [["demo/fastsam/examples/sa_8776.jpg"], ["demo/fastsam/examples/sa_414.jpg"], 
            ["demo/fastsam/examples/sa_1309.jpg"], ["demo/fastsam/examples/sa_11025.jpg"],
            ["demo/fastsam/examples/sa_561.jpg"], ["demo/fastsam/examples/sa_192.jpg"], 
            ["demo/fastsam/examples/sa_10039.jpg"], ["demo/fastsam/examples/sa_862.jpg"]
            ]
default_example = examples[0]
cond_img_e = gr.Image(label="Input", value=default_example[0], type='pil')
cond_img_p = gr.Image(label="Input with points", value=default_example[0], type='pil')
cond_img_t = gr.Image(label="Input with text", value="demo/fastsam/examples/dogs.jpg", type='pil')

segm_img_e = gr.Image(label="Segmented Image", interactive=False, type='pil')
segm_img_p = gr.Image(label="Segmented Image with points", interactive=False, type='pil')
segm_img_t = gr.Image(label="Segmented Image with text", interactive=False, type='pil')

input_size_slider = gr.components.Slider(minimum=512,
                                        maximum=1024,
                                        value=1024,
                                        step=64,
                                        label='Input_size',
                                        info='Our model was trained on a size of 1024')

def sam_tab(sam_args, ai_handler):
    # css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

    with gr.Tab(sam_args['name']):
        with gr.Tab("Everything mode"):
            # Images
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    cond_img_e.render()

                with gr.Column(scale=1):
                    segm_img_e.render()

            # Submit & Clear
            with gr.Row():
                with gr.Column():
                    input_size_slider.render()

                    with gr.Row():
                        contour_check = gr.Checkbox(value=True, label='withContours', info='draw the edges of the masks')

                        with gr.Column():
                            segment_btn_e = gr.Button("Segment Everything", variant='primary')
                            clear_btn_e = gr.Button("Clear", variant="secondary")

                    gr.Markdown("Try some of the examples below ⬇️")
                    gr.Examples(examples=examples,
                                inputs=[cond_img_e],
                                outputs=segm_img_e,
                                fn=ai_handler.fastsam_handler.segment_everything,
                                cache_examples=True,
                                examples_per_page=4)

                with gr.Column():
                    with gr.Accordion("Advanced options", open=False):
                        iou_threshold = gr.Slider(0.1, 0.9, 0.7, step=0.1, label='iou', info='iou threshold for filtering the annotations')
                        conf_threshold = gr.Slider(0.1, 0.9, 0.25, step=0.05, label='conf', info='object confidence threshold')
                        with gr.Row():
                            mor_check = gr.Checkbox(value=False, label='better_visual_quality', info='better quality using morphologyEx')
                            with gr.Column():
                                retina_check = gr.Checkbox(value=True, label='use_retina', info='draw high-resolution segmentation masks')


        segment_btn_e.click(ai_handler.fastsam_handler.segment_everything,
                            inputs=[
                                cond_img_e,
                                input_size_slider,
                                iou_threshold,
                                conf_threshold,
                                mor_check,
                                contour_check,
                                retina_check,
                            ],
                            outputs=segm_img_e)

        with gr.Tab("Points mode"):
            # Images
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    cond_img_p.render()

                with gr.Column(scale=1):
                    segm_img_p.render()
                    
            # Submit & Clear
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        add_or_remove = gr.Radio(["Add Mask", "Remove Area"], value="Add Mask", label="Point_label (foreground/background)")

                        with gr.Column():
                            segment_btn_p = gr.Button("Segment with points prompt", variant='primary')
                            clear_btn_p = gr.Button("Clear points", variant='secondary')

                    gr.Markdown("Try some of the examples below ⬇️")
                    gr.Examples(examples=examples,
                                inputs=[cond_img_p],
                                # outputs=segm_img_p,
                                # fn=segment_with_points,
                                # cache_examples=True,
                                examples_per_page=4)


        cond_img_p.select(ai_handler.fastsam_handler.get_points_with_draw, [cond_img_p, add_or_remove], cond_img_p)

        segment_btn_p.click(ai_handler.fastsam_handler.segment_with_points,
                            inputs=[cond_img_p],
                            outputs=[segm_img_p, cond_img_p])

        with gr.Tab("Text mode"):
            # Images
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    cond_img_t.render()

                with gr.Column(scale=1):
                    segm_img_t.render()

            # Submit & Clear
            with gr.Row():
                with gr.Column():
                    input_size_slider_t = gr.components.Slider(minimum=512,
                                                            maximum=1024,
                                                            value=1024,
                                                            step=64,
                                                            label='Input_size',
                                                            info='Our model was trained on a size of 1024')
                    with gr.Row():
                        with gr.Column():
                            contour_check = gr.Checkbox(value=True, label='withContours', info='draw the edges of the masks')
                            text_box = gr.Textbox(label="text prompt", value="a black dog")

                        with gr.Column():
                            segment_btn_t = gr.Button("Segment with text", variant='primary')
                            clear_btn_t = gr.Button("Clear", variant="secondary")

                    gr.Markdown("Try some of the examples below ⬇️")
                    gr.Examples(examples=[["demo/fastsam/examples/dogs.jpg"]] + examples,
                                inputs=[cond_img_e],
                                # outputs=segm_img_e,
                                # fn=segment_everything,
                                # cache_examples=True,
                                examples_per_page=4)

                with gr.Column():
                    with gr.Accordion("Advanced options", open=False):
                        iou_threshold = gr.Slider(0.1, 0.9, 0.7, step=0.1, label='iou', info='iou threshold for filtering the annotations')
                        conf_threshold = gr.Slider(0.1, 0.9, 0.25, step=0.05, label='conf', info='object confidence threshold')
                        with gr.Row():
                            mor_check = gr.Checkbox(value=False, label='better_visual_quality', info='better quality using morphologyEx')
                            retina_check = gr.Checkbox(value=True, label='use_retina', info='draw high-resolution segmentation masks')
                            wider_check = gr.Checkbox(value=False, label='wider', info='wider result')

        segment_btn_t.click(ai_handler.fastsam_handler.segment_everything,
                            inputs=[
                                cond_img_t,
                                input_size_slider_t,
                                iou_threshold,
                                conf_threshold,
                                mor_check,
                                contour_check,
                                retina_check,
                                text_box,
                                wider_check,
                            ],
                            outputs=segm_img_t)

        def clear():
            return None, None
        
        def clear_text():
            return None, None, None

        clear_btn_e.click(clear, outputs=[cond_img_e, segm_img_e])
        clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p])
        clear_btn_t.click(clear_text, outputs=[cond_img_p, segm_img_p, text_box])
