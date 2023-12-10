import numpy as np
from ultralytics import YOLO
from .base import BaseHandler
from PIL import ImageDraw
import gradio as gr
from utils.fastsam.tools import format_results, fast_process, point_prompt, text_prompt


class FastSAMHandler(BaseHandler):
    """
    FastSAMHandler is a handler for FastSAM.
    """

    def __init__(self, args):
        super().__init__(args)
        self.fastsam_model_path = self.handle_args.get('fastsam_model_path')
        self.device = self.handle_args.get('device')
        self.global_points = []
        self.global_point_label = []

    def segment_everything(
        self,
        input,
        input_size=1024, 
        iou_threshold=0.7,
        conf_threshold=0.25,
        better_quality=False,
        withContours=True,
        use_retina=True,
        text="",
        wider=False,
        mask_random_color=True,
    ):
        self.init_model()
        input_size = int(input_size)  # 确保 imgsz 是整数
        # Thanks for the suggestion by hysts in HuggingFace.
        w, h = input.size
        scale = input_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        input = input.resize((new_w, new_h))

        results = self.model(input,
                        device=self.device,
                        retina_masks=True,
                        iou=iou_threshold,
                        conf=conf_threshold,
                        imgsz=input_size,)

        if len(text) > 0:
            results = format_results(results[0], 0)
            annotations, _ = text_prompt(results, text, input, device=self.device, wider=wider)
            annotations = np.array([annotations])
        else:
            annotations = results[0].masks.data
        
        fig = fast_process(annotations=annotations,
                        image=input,
                        device=self.device,
                        scale=(1024 // input_size),
                        better_quality=better_quality,
                        mask_random_color=mask_random_color,
                        bbox=None,
                        use_retina=use_retina,
                        withContours=withContours,)
        return fig

    def segment_with_points(
        self,
        input,
        input_size=1024, 
        iou_threshold=0.7,
        conf_threshold=0.25,
        better_quality=False,
        withContours=True,
        use_retina=True,
        mask_random_color=True,
    ):
        self.init_model()
        input_size = int(input_size)  # 确保 imgsz 是整数
        # Thanks for the suggestion by hysts in HuggingFace.
        w, h = input.size
        scale = input_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        input = input.resize((new_w, new_h))
        
        scaled_points = [[int(x * scale) for x in point] for point in self.global_points]

        results = self.model(input,
                        device=self.device,
                        retina_masks=True,
                        iou=iou_threshold,
                        conf=conf_threshold,
                        imgsz=input_size,)
        
        results = format_results(results[0], 0)
        annotations, _ = point_prompt(results, scaled_points, self.global_point_label, new_h, new_w)
        annotations = np.array([annotations])

        fig = fast_process(annotations=annotations,
                        image=input,
                        device=self.device,
                        scale=(1024 // input_size),
                        better_quality=better_quality,
                        mask_random_color=mask_random_color,
                        bbox=None,
                        use_retina=use_retina,
                        withContours=withContours,)

        self.global_points = []
        self.global_point_label = []
        return fig, None

    def get_points_with_draw(self, image, label, evt: gr.SelectData):
        x, y = evt.index[0], evt.index[1]
        point_radius, point_color = 15, (255, 255, 0) if label == 'Add Mask' else (255, 0, 255)
        self.global_points.append([x, y])
        self.global_point_label.append(1 if label == 'Add Mask' else 0)
        
        print(x, y, label == 'Add Mask')
        
        # 创建一个可以在图像上绘图的对象
        draw = ImageDraw.Draw(image)
        draw.ellipse([(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], fill=point_color)
        return image


    def init_model(self):
        if self.model is None:
            self.model = YOLO(self.fastsam_model_path)
