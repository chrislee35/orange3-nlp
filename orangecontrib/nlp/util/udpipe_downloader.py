import requests
from Orange.misc.environ import data_dir_base
import os

class UDPipeDownloader:
    @staticmethod
    def model_path(language_code: str) -> str:
        model_dir = os.path.join(data_dir_base(), 'Orange', 'udpipe')
        model_path = f"{model_dir}/{language_code}.udpipe"
        return model_path

    @staticmethod
    def download(language_code: str) -> bool:
        UDPIPE_LANGUAGE_MODELS = {
            'af': 'afrikaans-afribooms',
            'ar': 'arabic-padt',
            'bg': 'bulgarian-btb',
            'bn': 'bengali-brc',
            'ca': 'catalan-ancora',
            'cs': 'czech-pdt',
            'cu': 'old-church-slavonic',
            'da': 'danish-ddt',
            'de': 'german-hdt',
            'el': 'greek-gdt',
            'en': 'english-ewt',
            'es': 'spanish-ancora',
            'et': 'estonian-edt',
            'eu': 'basque-bdt',
            'fa': 'persian-seraji',
            'fi': 'finnish-tdt',
            'fr': 'french-gsd',
            'ga': 'irish-idt',
            'gl': 'galician-ctg',
            'he': 'hebrew-htb',
            'hi': 'hindi-hdtb',
            'hr': 'croatian-set',
            'hu': 'hungarian-szeged',
            'id': 'indonesian-gsd',
            'is': 'icelandic-icepahc',
            'it': 'italian-isdt',
            'ja': 'japanese-gsd',
            'kk': 'kazakh-kkk',
            'ko': 'korean-kaist',
            'la': 'latin-ittb',
            'lv': 'latvian-lvtb',
            'lt': 'lithuanian-alksnis',
            'mr': 'marathi-ufal',
            'mt': 'maltese-mudt',
            'nl': 'dutch-alpino',
            'no': 'norwegian-bokmaal',
            'pl': 'polish-lfg',
            'pt': 'portuguese-bosque',
            'ro': 'romanian-rrt',
            'ru': 'russian-syntagrus',
            'sk': 'slovak-snk',
            'sl': 'slovenian-ssj',
            'sr': 'serbian-set',
            'sv': 'swedish-talbanken',
            'ta': 'tamil-ttb',
            'te': 'telugu-mtg',
            'th': 'thai-pudt',
            'tr': 'turkish-ttb',
            'uk': 'ukrainian-iu',
            'ur': 'urdu-udtb',
            'vi': 'vietnamese-vtb',
            'zh': 'chinese-gsd'
        }
        model_name = UDPIPE_LANGUAGE_MODELS.get(language_code, None)
        if not model_name:
            return False

        ud_ver = "ud-2.5-191206"
        model_filename = f"{model_name}-{ud_ver}.udpipe"
        model_url = f"https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/{model_filename}"

        response = requests.get(model_url)
        response.raise_for_status()

        model_path = UDPipeDownloader.model_path(language_code)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        return True

