import jsonlines
import os
from loguru import logger

class BaseAttackManager:
    def __init__(self, res_save_path=None, delete_existing_res=False):
        self.res_save_path = res_save_path
        logger.info(f"Results will be saved to '{res_save_path}'")
        res_dir = os.path.dirname(res_save_path)
        if os.path.exists(res_save_path):
            if os.path.isfile(res_save_path) and delete_existing_res:
                logger.warning(f"The path '{res_save_path}' already exists. Deleting it.")
                os.remove(res_save_path)
            else:
                pass
        os.makedirs(res_dir, exist_ok=True)

    
    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)
         
    def update_res(self, res:dict):
        if self.res_save_path is not None:
            with jsonlines.open(self.res_save_path, mode='a') as writer:
                writer.write(res)
        else:
            logger.warning('You did not specify a path to save the results. The results will not be saved.')
    
    def log(self, data, save=False):
        if save:
            assert isinstance(data, dict), 'Data to be saved must be a dictionary'
            logger.info(data)
            self.update_res(data)
        else:
            logger.info(data)

    def save(self, data):
        assert isinstance(data, dict), 'Data to be saved must be a dictionary'
        self.update_res(data)

    def attack(self):
        pass