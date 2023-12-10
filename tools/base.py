

class BaseHandler:
    def __init__(self, args) -> None:
        self.model = None
        self.handle_args = args.get(type(self).__name__, {})
        if self.handle_args.get('init_model_when_start_server'):
            self.init_model()
    
    def infer(self, input_data, **kwargs):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError