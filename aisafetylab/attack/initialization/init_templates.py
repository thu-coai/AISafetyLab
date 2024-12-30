import os
import json
import random

class InitTemplates:
    def __init__(self):
        pass
    
    def get_templates(self, name, num):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, './init_templates.json')
        with open(file_path) as f:
            templates = json.load(f)
        all_templates = templates['attack'][name]
        
        if num == -1:
            return all_templates
        
        res = random.sample(all_templates, num)
        return res
    
class PopulationInitializer:
    def init_population(self, data_path):
        with open(data_path) as f:
            population = json.load(f)
        return population