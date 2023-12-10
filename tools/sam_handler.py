import os
import numpy as np
from .base import BaseHandler
import torch
import time
import psutil
import cv2
import gradio as gr
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from utils.painter import mask_painter, point_painter
from utils.misc import get_prompt

mask_color = 3
mask_alpha = 0.7
contour_color = 1
contour_width = 5
point_color_ne = 8
point_color_ps = 50
point_alpha = 0.9
point_radius = 15
contour_color = 2
contour_width = 5


class SAMHandler(BaseHandler):
    def __init__(self, args):
        super().__init__(args)
        self.model_ckpt = self.handle_args.get('model_ckpt')
        self.model_type = self.handle_args.get('model_type')
        assert self.model_type in ['vit_b', 'vit_l', 'vit_h'], 'model_type must be vit_b, vit_l, or vit_h'
        self.device = self.handle_args.get('device')
        self.torch_dtype = torch.float16 if 'cuda' in self.device else torch.float32
        self.embedded = False

    def init_model(self):
        if self.model is None:
            self.model = sam_model_registry[self.model_type](checkpoint=self.model_ckpt)
            self.model.to(device=self.device)
            self.predictor = SamPredictor(self.model)

    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True, mask_color=3):
        '''
        it is used in first frame in video
        return: mask, logit, painted image(mask+point)
        '''
        # self.sam_controler.set_image(image)
        neg_flag = labels[-1]
        if neg_flag==1:
            #find neg
            prompts = {
                'point_coords': points,
                'point_labels': labels,
            }
            masks, scores, logits = self.infer(prompts, 'point', multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
            prompts = {
                'point_coords': points,
                'point_labels': labels,
                'mask_input': logit[None, :, :]
            }
            masks, scores, logits = self.infer(prompts, 'both', multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        else:
           #find positive
            prompts = {
                'point_coords': points,
                'point_labels': labels,
            }
            masks, scores, logits = self.infer(prompts, 'point', multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
            
        
        assert len(points)==len(labels)
        
        painted_image = mask_painter(image, mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)
        painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels>0)],axis = 1), point_color_ne, point_alpha, point_radius, contour_color, contour_width)
        painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels<1)],axis = 1), point_color_ps, point_alpha, point_radius, contour_color, contour_width)
        painted_image = Image.fromarray(painted_image)
        
        return mask, logit, painted_image
    

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        # PIL.open(image_path) 3channel: RGB
        # image embedding: avoid encode the same image multiple times
        self.orignal_image = image
        if self.embedded:
            print('repeat embedding, please reset_image.')
            return
        self.predictor.set_image(image)
        self.embedded = True
        return
    
    @torch.no_grad()
    def reset_image(self):
        # reset image embeding
        self.predictor.reset_image()
        self.embedded = False

    def infer(self, prompts, mode, multimask=True):
        """
        image: numpy array, h, w, 3
        prompts: dictionary, 3 keys: 'point_coords', 'point_labels', 'mask_input'
        prompts['point_coords']: numpy array [N,2]
        prompts['point_labels']: numpy array [1,N]
        prompts['mask_input']: numpy array [1,256,256]
        mode: 'point' (points only), 'mask' (mask only), 'both' (consider both)
        mask_outputs: True (return 3 masks), False (return 1 mask only)
        whem mask_outputs=True, mask_input=logits[np.argmax(scores), :, :][None, :, :]
        """
        assert self.embedded, 'prediction is called before set_image (feature embedding).'
        assert mode in ['point', 'mask', 'both'], 'mode must be point, mask, or both'
        
        if mode == 'point':
            masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'], 
                                point_labels=prompts['point_labels'], 
                                multimask_output=multimask)
        elif mode == 'mask':
            masks, scores, logits = self.predictor.predict(mask_input=prompts['mask_input'], 
                                multimask_output=multimask)
        elif mode == 'both':   # both
            masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'], 
                                point_labels=prompts['point_labels'], 
                                mask_input=prompts['mask_input'], 
                                multimask_output=multimask)
        else:
            raise("Not implement now!")
        # masks (n, h, w), scores (n,), logits (n, 256, 256)
        return masks, scores, logits

    # use sam to get the mask
    def sam_refine(self, video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData):
        """
        Args:
            template_frame: PIL.Image
            point_prompt: flag for positive or negative button click
            click_state: [[points], [labels]]
        """
        if point_prompt == "Positive":
            coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
            interactive_state["positive_click_times"] += 1
        else:
            coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
            interactive_state["negative_click_times"] += 1
        self.init_model()
        # prompt for sam model
        self.reset_image()
        self.set_image(video_state["origin_images"][video_state["select_frame_number"]])
        prompt = get_prompt(click_state=click_state, click_input=coordinate)

        mask, logit, painted_image = self.first_frame_click( 
                                                        image=video_state["origin_images"][video_state["select_frame_number"]], 
                                                        points=np.array(prompt["input_point"]),
                                                        labels=np.array(prompt["input_label"]),
                                                        multimask=prompt["multimask_output"],
                                                        )
        video_state["masks"][video_state["select_frame_number"]] = mask
        video_state["logits"][video_state["select_frame_number"]] = logit
        video_state["painted_images"][video_state["select_frame_number"]] = painted_image

        operation_log = [("[Must Do]", "Add mask"), (": add the current displayed mask for video segmentation.\n", None),
                        ("[Optional]", "Remove mask"), (": remove all added masks.\n", None),
                        ("[Optional]", "Clear clicks"), (": clear current displayed mask.\n", None),
                        ("[Optional]", "Click image"), (": Try to click the image shown in step2 if you want to generate more masks.\n", None)]
        return painted_image, video_state, interactive_state, operation_log, operation_log

    # extract frames from upload video
    def get_frames_from_video(self, video_input, video_state):
        """
        Args:
            video_path:str
            timestamp:float64
        Return 
            [[0:nearest_frame], [nearest_frame:], nearest_frame]
        """
        self.init_model()
        video_path = video_input
        frames = []
        user_name = time.time()
        operation_log = [("[Must Do]", "Click image"), (": Video uploaded! Try to click the image shown in step2 to add masks.\n", None)]
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    current_memory_usage = psutil.virtual_memory().percent
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if current_memory_usage > 90:
                        operation_log = [("Memory usage is too high (>90%). Stop the video extraction. Please reduce the video resolution or frame rate.", "Error")]
                        print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                        break
                else:
                    break
        except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
            print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
        image_size = (frames[0].shape[0],frames[0].shape[1]) 
        # initialize video_state
        video_state = {
            "user_name": user_name,
            "video_name": os.path.split(video_path)[-1],
            "origin_images": frames,
            "painted_images": frames.copy(),
            "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
            "logits": [None]*len(frames),
            "select_frame_number": 0,
            "fps": fps
            }
        video_info = "Video Name: {},\nFPS: {},\nTotal Frames: {},\nImage Size:{}".format(video_state["video_name"], round(video_state["fps"], 0), len(frames), image_size)
        self.reset_image() 
        self.set_image(video_state["origin_images"][0])
        return video_state, video_info, video_state["origin_images"][0], gr.update(visible=True, maximum=len(frames), value=1), gr.update(visible=True, maximum=len(frames), value=len(frames)), \
                            gr.update(visible=True), gr.update(visible=True), \
                            gr.update(visible=True), gr.update(visible=True),\
                            gr.update(visible=True), gr.update(visible=True), \
                            gr.update(visible=True), gr.update(visible=True), \
                            gr.update(visible=True), gr.update(visible=True), \
                            gr.update(visible=True), gr.update(visible=True, choices=[], value=[]), \
                            gr.update(visible=True, value=operation_log), gr.update(visible=True, value=operation_log)

    # get the select frame from gradio slider
    def select_template(self, image_selection_slider, video_state, interactive_state, mask_dropdown):

        # images = video_state[1]
        image_selection_slider -= 1
        video_state["select_frame_number"] = image_selection_slider

        # once select a new template frame, set the image in sam

        self.reset_image()
        self.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

        operation_log = [("",""), ("Select tracking start frame {}. Try to click the image to add masks for tracking.".format(image_selection_slider),"Normal")]

        return video_state["painted_images"][image_selection_slider], video_state, interactive_state, operation_log, operation_log
