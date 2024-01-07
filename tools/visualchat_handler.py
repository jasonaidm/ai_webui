import torch
from .base import BaseHandler
from utils.chatglm_utils import *


class VisualChatHandler(BaseHandler):
    def __init__(self, args):
        super().__init__(args)
        self.model_path = self.handle_args.get('model_path')
        self.trust_remote_code = self.handle_args.get('trust_remote_code', True)
        self.device = self.handle_args.get('device', 'cuda:0')
        self.quant = self.handle_args.get('quant')

    def init_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
            if self.quant in [4, 8]:
                self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code).quantize(self.quant).half().to(self.device)
            else:
                self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code).half().to(self.device)
            self.model = self.model.eval()

    def stream_chat(self, input, image_path, chatbot, max_length, top_p, temperature, history):
        self.init_model()
        if image_path is None:
            return [(input, "图片不能为空。请重新上传图片并重试。")], []
        chatbot.append((parse_text(input), ""))
        with torch.no_grad():
            for response, history in self.model.stream_chat(self.tokenizer, image_path, input, history, max_length=max_length, top_p=top_p,
                                                temperature=temperature):
                chatbot[-1] = (parse_text(input), parse_text(response))

                yield chatbot, history

    def stream_chat2(self, image_path, chatbot, max_length, top_p, temperature):
        self.init_model()
        input, history = "描述这张图片。", []
        chatbot.append((parse_text(input), ""))
        with torch.no_grad():
            for response, history in self.model.stream_chat(self.tokenizer, image_path, input, history, max_length=max_length,
                                                top_p=top_p,
                                                temperature=temperature):
                chatbot[-1] = (parse_text(input), parse_text(response))

                yield chatbot, history


