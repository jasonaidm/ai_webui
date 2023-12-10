from .base import BaseHandler
from utils.chatglm_utils import *


class ChatGLMHandler(BaseHandler):
    def __init__(self, args):
        super().__init__(args)
        self.llm_model_path = self.handle_args.get('llm_model_path')
        self.num_gpus = self.handle_args.get('num_gpus', 2)
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            self.num_gpus = min(self.num_gpus, len(os.environ['CUDA_VISIBLE_DEVICES']))

        self.trust_remote_code = self.handle_args.get('trust_remote_code', True)

    def init_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path, trust_remote_code=True)
            if 'int' in os.path.basename(self.llm_model_path):
                self.model = AutoModel.from_pretrained(self.llm_model_path, trust_remote_code=True).cuda()
            else:
                self.model = load_model_on_gpus(self.llm_model_path, num_gpus=self.num_gpus)
            self.model = self.model.eval()

    def infer(self, input_text, history=[], max_length=2048, top_p=90, temperature=95, **kwargs):
        self.init_model()
        response, history = self.model.chat(self.tokenizer, input_text, history=history, max_length=max_length, top_p=top_p, temperature=temperature)
        return response, history

    def stream_chat(self, input_text, chatbot, max_length, top_p, temperature, history, past_key_values):
        self.init_model()
        chatbot.append((parse_text(input_text), ""))
        for response, history, past_key_values in self.model.stream_chat(self.tokenizer, input_text, history, past_key_values=past_key_values,
                                                                    return_past_key_values=True,
                                                                    max_length=max_length, top_p=top_p,
                                                                    temperature=temperature):
            chatbot[-1] = (parse_text(input_text), parse_text(response))

            yield chatbot, history, past_key_values
        

