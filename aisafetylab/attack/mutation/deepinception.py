from typing import List

import requests
from aisafetylab.attack.mutation import BaseMutation

class DeepInception(BaseMutation):
    """
    Translate is a class for translating the query to another language.
    """
    def __init__(self, 
            layer_num, 
            character_num, 
            scene,
            template,
            language='en'
            ):
        self.layer_num = layer_num
        self.character_num = character_num
        self.scene = scene
        self.language = language
        self.template = template
        self.template = self.template.replace("{character_num}", str(self.character_num))
        self.template = self.template.replace("{layer_num}", str(self.layer_num))
        self.template = self.template.replace("{scene}", self.scene)

    def apply_template(self, query: str):
        # Replace placeholders in the template with actual values
        final_query = self.template
        final_query = final_query.replace("{query}", query)
        return final_query