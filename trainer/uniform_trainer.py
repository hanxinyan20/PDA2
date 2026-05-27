
from model.nn import MultiClassClassifier
from dataset import DomainWiseData
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from .trainer_utils import EarlyStopping
from strategy.base_strategy import UniformAcquisitionStrategy
import json
class UniformTrainer:
    def __init__(self, acquisition_config, training_config, finetuning_config, model_config, data:DomainWiseData,device,logger):
        self.acquisition_config = acquisition_config
        self.training_config = training_config
        self.finetuning_config = finetuning_config
        self.model_config = model_config
        self.data = data
        self.device = device
        self.logger = logger
        self.model = MultiClassClassifier(
            input_dim = self.data.n_features,
            layer_dims=model_config['layer_dims'],
            num_classes=self.data.n_labels
        ).to(device)
        if self.training_config['optimizer'] == 'adam':
            train_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_config['learning_rate'], weight_decay=self.training_config['weight_decay'])
        elif self.training_config['optimizer'] == 'sgd':
            train_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.training_config['learning_rate'], weight_decay=self.training_config['weight_decay'])
        else:
            raise ValueError(f"Unsupported optimizer: {self.training_config['optimizer']}")
        src_train_dataset = self.data.get_train_dataset()
        src_valid_dataset = self.data.get_src_valid_dataset()
        self.tgt_test_dataset = self.data.get_tgt_test_dataset()
        self.tgt_valid_dataset = self.data.get_tgt_valid_dataset()
        self.criterion = torch.nn.CrossEntropyLoss()
        steps, valid_loss_record = self.train(
            model=self.model,
            optimizer=train_optimizer,
            train_dataset=src_train_dataset,
            valid_dataset=src_valid_dataset,
            early_stop_config=training_config['early_stop'],
            max_steps=training_config['max_steps'],
            batch_size=training_config['batch_size']
        )
        pretrain_log = {
            'src_region': {
                'region': self.data.src_region,
                'train_size': len(src_train_dataset),
                'valid_size': len(src_valid_dataset)
            },
            'optimizer': self.training_config['optimizer'],
            'training_steps': steps,
            'valid_loss_record': valid_loss_record
        }
        self.logger.info("pretraining log: %s", json.dumps(pretrain_log, indent=4))
    def acquire_and_finetune(self):
        if self.finetuning_config['optimizer'] == 'adam':
            self.finetune_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.finetuning_config['learning_rate'], weight_decay=self.finetuning_config['weight_decay'])
        elif self.finetuning_config['optimizer'] == 'sgd':
            self.finetune_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.finetuning_config['learning_rate'], weight_decay=self.finetuning_config['weight_decay'])
        else:
            raise ValueError(f"Unsupported optimizer: {self.finetuning_config['optimizer']}")
        
        summary_dict = {}
        _, test_acc = self.evaluate(self.model, self.tgt_test_dataset)
        summary_loss_dict = {'valid_loss': [], 'test_loss': [], 'test_acc_from_0': [test_acc]}
        
        
        
        for b in range(self.acquisition_config['n_batches']):
            self.logger.info(f"Finetuning batch {b+1}")
            ft_log = self.acquire_and_finetune_batch()
            summary_loss_dict['valid_loss'].append(ft_log['tgt_region']['valid']['loss'])
            summary_loss_dict['test_loss'].append(ft_log['tgt_region']['test']['loss'])
            summary_loss_dict['test_acc_from_0'].append(ft_log['tgt_region']['test']['acc'])

        self.logger.info("%s", json.dumps(summary_loss_dict, indent=4))
        return summary_loss_dict['test_acc_from_0']   
    
    def acquire_and_finetune_batch(self):
        '''
        acquire data (n_rounds_per_batch * n_samples_per_round)
        finetune the model on the acquired data
        '''
        strategy = UniformAcquisitionStrategy(
            acquisition_config=self.acquisition_config,
            domain_wise_data=self.data
        )
        x_batch, y_batch = [], []
        
        for r in range(self.acquisition_config['n_rounds_per_batch']):
            n_sampls = self.acquisition_config['n_samples_per_round']
            x_acquired, y_acquired, _ = strategy.acquire(n_samples=n_sampls)
            x_acquired = x_acquired.to(self.device)
            y_acquired = y_acquired.to(self.device)
            x_batch.append(x_acquired)
            y_batch.append(y_acquired)
        x_batch = torch.cat(x_batch, dim=0).to(self.device)
        y_batch = torch.cat(y_batch, dim=0).to(self.device)
        finetune_dataset = TensorDataset(x_batch, y_batch)
        steps, val_loss_record = self.train(
            model=self.model,
            optimizer=self.finetune_optimizer,
            train_dataset=finetune_dataset,
            valid_dataset=self.tgt_valid_dataset,
            early_stop_config=self.finetuning_config['early_stop'],
            max_steps=self.finetuning_config['max_steps'],
            batch_size=self.finetuning_config['batch_size']
        )
        test_loss, test_acc = self.evaluate(self.model, self.tgt_test_dataset)
        ft_log = {
            'optimizer': self.finetuning_config['optimizer'],
            'tgt_region':{
                'region': self.data.tgt_region,
                'valid': {
                    'size': len(self.tgt_valid_dataset),
                    'loss': val_loss_record
                },
                'test': {
                    'size': len(self.tgt_test_dataset),
                    'loss': test_loss,
                    'acc': test_acc
                }

            },
            'finetuning_dataset_size': len(finetune_dataset),
            'finetuning_steps': steps,
            'finetuning_batch_size': self.finetuning_config['batch_size'],
            'region_acquired_times': self.data.get_region_acquired_times()
        }
        self.logger.info("finetuning log: %s", json.dumps(ft_log, indent=4))
        return ft_log
        
    
        
    def train(self, model:nn.Module, optimizer:torch.optim.Optimizer, train_dataset:Dataset, valid_dataset:Dataset, early_stop_config, max_steps:int, batch_size:int):
        if early_stop_config['enabled']:
            early_stopping = EarlyStopping(
                patience=early_stop_config['patience'],
                delta=early_stop_config['delta'],
                verbose=True,
            )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        valid_loss_record = []
        valid_loss, valid_acc = self.evaluate(model, valid_dataset)
        valid_loss_record.append(valid_loss)
        for step in range(max_steps):
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            valid_loss, valid_acc = self.evaluate(model, valid_dataset)
            valid_loss_record.append(valid_loss)
            if early_stop_config['enabled']:
                early_stopping(valid_loss, model)
                if early_stopping.early_stop:
                    break
        return step+1, valid_loss_record
    def evaluate(self, model, valid_dataset):
        valid_loader = DataLoader(valid_dataset, batch_size=self.training_config['batch_size'], shuffle=False)
        model.eval()
        valid_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in valid_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)
                
                loss = self.criterion(outputs, labels)
                valid_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)  
                correct_predictions += (predicted == labels).sum().item() 
                total_predictions += labels.size(0)  

        avg_loss = valid_loss / len(valid_loader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy
            
            
            
                
            
            
