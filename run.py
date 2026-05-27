import pprint
from parse_config import get_config
import logging
import os
from trainer.pda_trainer import PDATrainer
from trainer.al_trainer import ALTrainer
from trainer.from_saved_trainer import FromSavedTrainer
from trainer.uniform_trainer import UniformTrainer
from trainer.random_trainer import RandomTrainer
from trainer.daa_trainer import DAATrainer
from trainer.oracle_trainer import OracleTrainer
from trainer.val_trainer import ValTrainer
from dataset import DomainWiseData, SampleWiseData
import random
import numpy as np
import torch
import json
def set_logger(config):
    logger = logging.getLogger()

    logger.handlers = []

    log_dir = config['log']['log_dir']
    config['acquisition']['save_dir'] = log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.info(config)
    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
DATA_CLASSES = {
    'domain_wise': DomainWiseData,
    'sample_wise': SampleWiseData,
}

TRAINER_CLASSES = {
    'pda': PDATrainer,
    'al': ALTrainer,
    'saved': FromSavedTrainer,
    'uniform': UniformTrainer,
    'random': RandomTrainer,
    'daa': DAATrainer,
    'oracle': OracleTrainer,
    'val': ValTrainer,
}
def main(config):
    logger = set_logger(config)
    logger.info("%s", pprint.pformat(config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config['seed'])
    data_class = DATA_CLASSES[config['acquisition']['granularity']]
    data_config = config['data']
    data = data_class(data_config)
    
    trainer_class = TRAINER_CLASSES[config['acquisition']['acquisition_function']['name']]
    if trainer_class == PDATrainer or trainer_class == UniformTrainer or trainer_class == RandomTrainer or trainer_class == DAATrainer or trainer_class == OracleTrainer or trainer_class == ValTrainer:
        trainer = trainer_class(
            acquisition_config=config['acquisition'],
            training_config=config['training'],
            finetuning_config=config['finetuning'],
            model_config=config['model'],
            data=data,
            device=device,
            logger=logger,
        )
    elif trainer_class == ALTrainer:
        trainer = trainer_class(
            acquisition_config=config['acquisition'],
            training_config=config['training'],
            seed = config['seed'],
            model_config=config['model'],
            sample_wise_data=data,
            device=device,
            logger=logger,
        )
    elif trainer_class == FromSavedTrainer:
        trainer = trainer_class(
            acquisition_config=config['acquisition'],
            training_config=config['training'],
            finetuning_config=config['finetuning'],
            model_config=config['model'],
            sample_wise_data=data,
            device=device,
            logger=logger,
        )
    return trainer.acquire_and_finetune()
if __name__ == "__main__":
    config = get_config()
    test_acc = []
    for seed in range(config['seed']['left'], config['seed']['right']):
        config['seed'] = seed
        test_acc.append(main(config))
    summary_dict = {}
    summary_dict['test_acc'] = test_acc
    summary_dict['n_samples_per_batch'] = config['acquisition']['n_samples_per_round']*config['acquisition']['n_rounds_per_batch']
    summary_dict['n_samples_used'] = [i*summary_dict['n_samples_per_batch'] for i in range(len(test_acc[0]))]
    json.dump(summary_dict, open(os.path.join(config['log']['log_dir'], 'summary.json'), 'w'))
    
            
    