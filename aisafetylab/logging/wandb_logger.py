import wandb


class WandBLogger:
    """WandB logger."""

    def __init__(self, args):
        self.logger = wandb.init(
            project = "llm-rs",
            config = vars(args),
        )

    def log(self, dict):
        self.logger.log(dict)

    def finish(self):
        self.logger.finish()

