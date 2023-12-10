import os
from .base import BaseHandler
import whisper
import pandas as pd

class WhisperHandler(BaseHandler):
    def __init__(self, args, **kwargs):
        # model_name, model_dir, device='cuda:0'
        super().__init__(args)
        self.model_name = self.handle_args.get('model_name')
        self.device = self.handle_args.get('device', 'cuda:0')
        self.model_dir = self.handle_args.get('model_dir', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
        self.language_map = self.handle_args.get('language_map', {'英语': 'en', '普通话': 'zh'})
        
    def infer(self, audio_file, language=None):
        self.init_model()
        if not isinstance(audio_file, str):
            audio_file = audio_file.name
        save_dir = os.path.dirname(audio_file)
        audio_name = os.path.basename(audio_file).split('.')[0]
        os.makedirs(save_dir, exist_ok=True)
        excel_file = os.path.join(save_dir, f"{audio_name}_speech_res.xlsx")
        result = self.model.transcribe(audio_file, language=self.language_map.get(language), verbose=True)
        # rec_language = result['language']
        df = pd.DataFrame(result['segments'])
        df = df[['id', 'no_speech_prob', 'start', 'end', 'text']]
        df.to_excel(excel_file, index=False)
        text = ' '.join(df['text'].tolist())
        result['text'] = text
        result['segments'] = excel_file
        return result

    def predict_v2(self, audio_file):
        # load excel_file
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)

        # print the recognized text
        print(result.text)
        
    def init_model(self):
        if self.model is None:
            self.model = whisper.load_model(name=self.model_name, device=self.device, download_root=self.model_dir)
            

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}