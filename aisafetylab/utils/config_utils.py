# ----------------------- Configuration Management -----------------------
from loguru import logger
import yaml
import argparse


def parse_arguments():
    """Parse command-line arguments for GCG Attack configuration."""
    parser = argparse.ArgumentParser(description="GCG Attack Configuration")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


class ConfigManager:
    def __init__(self, config_path="config.yaml"):
        self.config = self.load_config(config_path)
        self.validate_config()

    def load_config(self, config_path):
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            raise e

    def validate_config(self):
        # Validate essential fields are present
        # required_fields = ['model_path', 'template_path', 'data_path']
        required_fields = []
        for field in required_fields:
            if field not in self.config:
                logger.error(f"Missing required configuration field: {field}")
                raise ValueError(f"Missing required configuration field: {field}")
