import os
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from models.tracker.config import CONFIG
from models.tracker.model.cutie import CUTIE
from models.tracker.inference.inference_core import InferenceCore
from models.tracker.utils.mask_mapper import MaskMapper
from utils.misc import generate_video_from_frames
from utils.painter import mask_painter
from .base import BaseHandler


class TrackerHandler(BaseHandler):
    def __init__(self, args):
        """
        device: model device
        cutie_ckpt: checkpoint of XMem model
        """
        super().__init__(args)
        self.config = OmegaConf.create(CONFIG)
        self.cutie_ckpt = self.handle_args.get("cutie_ckpt")
        self.device = self.handle_args.get("device", 'cuda:0')

    
    def init_model(self):
        if self.model is None:
            # initialise XMem
            network = CUTIE(self.config).to(self.device).eval()
            model_weights = torch.load(self.cutie_ckpt, map_location=self.device)
            network.load_weights(model_weights)

            # initialise IncerenceCore
            self.model = InferenceCore(network, self.config)
            
            # changable properties
            self.mapper = MaskMapper()
            self.initialised = False

    @torch.no_grad()
    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), 
                    mode='nearest')

    @torch.no_grad()
    def image_to_torch(self, frame: np.ndarray, device: str = 'cuda'):
            # frame: H*W*3 numpy array
            frame = frame.transpose(2, 0, 1)
            frame = torch.from_numpy(frame).float().to(device, non_blocking=True) / 255
            return frame
    
    @torch.no_grad()
    def track(self, frame, first_frame_annotation=None):
        """
        Input: 
        frames: numpy arrays (H, W, 3)
        logit: numpy array (H, W), logit

        Output:
        mask: numpy arrays (H, W)
        logit: numpy arrays, probability map (H, W)
        painted_image: numpy array (H, W, 3)
        """
        self.init_model()
        if first_frame_annotation is not None:   # first frame mask
            # initialisation
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.Tensor(mask).to(self.device)
        else:
            mask = None
            labels = None

        # prepare inputs
        frame_tensor = self.image_to_torch(frame, self.device)
        
        # track one frame
        probs = self.model.step(frame_tensor, mask, labels)   # logits 2 (bg fg) H W

        # convert to mask
        out_mask = torch.argmax(probs, dim=0)
        out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

        final_mask = np.zeros_like(out_mask)
        
        # map back
        for k, v in self.mapper.remappings.items():
            final_mask[out_mask == v] = k

        num_objs = final_mask.max()
        painted_image = frame
        for obj in range(1, num_objs+1):
            if np.max(final_mask==obj) == 0:
                continue
            painted_image = mask_painter(painted_image, (final_mask==obj).astype('uint8'), mask_color=obj+1)

        return final_mask, final_mask, painted_image

    @torch.no_grad()
    def clear_memory(self):
        self.model.clear_memory()
        self.mapper.clear_labels()
        torch.cuda.empty_cache()

    def generator(self, images: list, template_mask:np.ndarray):
        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i==0:           
                mask, logit, painted_image = self.track(images[i], template_mask)
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
            else:
                mask, logit, painted_image = self.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images

    def vos_tracking_video(self, video_state, interactive_state, mask_dropdown):
        self.init_model()
        operation_log = [("",""), ("Tracking finished! Try to click the Inpainting button to get the inpainting result.","Normal")]
        self.clear_memory()
        if interactive_state["track_end_number"]:
            following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
        else:
            following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

        if interactive_state["multi_mask"]["masks"]:
            if len(mask_dropdown) == 0:
                mask_dropdown = ["mask_001"]
            mask_dropdown.sort()
            template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
            for i in range(1,len(mask_dropdown)):
                mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
                template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
            video_state["masks"][video_state["select_frame_number"]]= template_mask
        else:      
            template_mask = video_state["masks"][video_state["select_frame_number"]]
        fps = video_state["fps"]

        # operation error
        if len(np.unique(template_mask))==1:
            template_mask[0][0]=1
            operation_log = [("Please add at least one mask to track by clicking the image in step2.","Error"), ("","")]
            # return video_output, video_state, interactive_state, operation_error
        masks, logits, painted_images = self.generator(images=following_frames, template_mask=template_mask)
        # clear GPU memory
        self.clear_memory()

        if interactive_state["track_end_number"]: 
            video_state["masks"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = masks
            video_state["logits"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = logits
            video_state["painted_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = painted_images
        else:
            video_state["masks"][video_state["select_frame_number"]:] = masks
            video_state["logits"][video_state["select_frame_number"]:] = logits
            video_state["painted_images"][video_state["select_frame_number"]:] = painted_images

        video_output = generate_video_from_frames(video_state["painted_images"], output_path="./results/track/{}".format(video_state["video_name"]), fps=fps) # import video_input to name the output video
        interactive_state["inference_times"] += 1
        
        print("For generating this tracking result, inference times: {}, click times: {}, positive: {}, negative: {}".format(interactive_state["inference_times"], 
                                                                                                                                            interactive_state["positive_click_times"]+interactive_state["negative_click_times"],
                                                                                                                                            interactive_state["positive_click_times"],
                                                                                                                                            interactive_state["negative_click_times"]))

        #### shanggao code for mask save
        if interactive_state["mask_save"]:
            if not os.path.exists('./results/mask/{}'.format(video_state["video_name"].split('.')[0])):
                os.makedirs('./results/mask/{}'.format(video_state["video_name"].split('.')[0]))
            i = 0
            print("save mask")
            for mask in video_state["masks"]:
                np.save(os.path.join('./results/mask/{}'.format(video_state["video_name"].split('.')[0]), '{:05d}.npy'.format(i)), mask)
                i+=1
            # save_mask(video_state["masks"], video_state["video_name"])
        #### shanggao code for mask save
        return video_output, video_state, interactive_state, operation_log, operation_log
