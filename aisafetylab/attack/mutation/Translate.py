from typing import List

import requests
from aisafetylab.attack.mutation import BaseMutation

class Translate(BaseMutation):
    """
    Translate is a class for translating the query to another language.
    """
    def __init__(self, attr_name='query', language='en'):
        self.attr_name = attr_name
        self.language = language
        languages_supported = {
            'en': 'English',
            'zh-CN': 'Chinese',
            'it': 'Italian',
            'vi': 'Vietnamese',
            'ar': 'Arabic',
            'ko': 'Korean',
            'th': 'Thai',
            'bn': 'Bengali',
            'sw': 'Swahili',
            'jv': 'Javanese'
        }
        if self.language in languages_supported:
            self.lang = languages_supported[self.language]
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    
    def translate(self, text, src_lang='auto'):
        """
        translate the text to another language
        """
        googleapis_url = 'https://translate.googleapis.com/translate_a/single'
        url = '%s?client=gtx&sl=%s&tl=%s&dt=t&q=%s' % (googleapis_url,src_lang,self.language,text)
        data = requests.get(url).json()
        res = ''.join([s[0] for s in data[0]])
        return res
    
    