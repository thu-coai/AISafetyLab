import hydra
from omegaconf import DictConfig, OmegaConf
from aisafetylab.logging import setup_logger
from loguru import logger

from aisafetylab.attack.attackers.advprompter import AdvprompterWorkspace, EvalWorkspace, EvalSuffixDatasetsWorkspace
setup_logger(log_file_path='logs/autodan_vicuna_{time}.log', stderr_level='INFO')

@hydra.main(version_base=None, config_path="./configs/advprompter_configs", config_name="train")
def main(cfg: DictConfig):
    logger.info("Starting run...")
    logger.info(f"Using parameters: \n{OmegaConf.to_yaml(cfg)}")
    if cfg.mode == "train":
        logger.info("Start training advprompter")
        workspace = AdvprompterWorkspace(cfg)
        workspace.train()
    elif cfg.mode == "eval":
        logger.info("Start evaluating advprompter, generating suffix using existing advprompter LLM")
        workspace = EvalWorkspace(cfg)
        workspace.eval()
    elif cfg.mode == "eval_suffix_dataset":
        logger.info("Start evaluating existing suffix datasets")
        workspace = EvalSuffixDatasetsWorkspace(cfg)
        workspace.eval_suffix_datasets(cfg.eval.suffix_dataset_pth_dct)
    else:
        raise ValueError(f"Mode {cfg.mode} not recognized.")
    logger.info("Finished!")


if __name__ == "__main__":
    main()
