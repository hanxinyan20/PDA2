from model.nn import MultiClassClassifier
from dataset import SampleWiseData
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from .trainer_utils import EarlyStopping
import json

class FromSavedTrainer:
    def __init__(self, acquisition_config, training_config, finetuning_config, model_config, sample_wise_data:SampleWiseData,device,logger):
        self.acquisition_config = acquisition_config
        self.training_config = training_config
        self.finetuning_config = finetuning_config
        self.model_config = model_config
        self.sample_wise_data = sample_wise_data
        self.device = device
        self.logger = logger
        self.model = MultiClassClassifier(
            input_dim = self.sample_wise_data.n_features,
            layer_dims=model_config['layer_dims'],
            num_classes=self.sample_wise_data.n_classes
        ).to(device)
        if self.training_config['optimizer'] == 'adam':
            train_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_config['learning_rate'], weight_decay=self.training_config['weight_decay'])
        elif self.training_config['optimizer'] == 'sgd':
            train_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.training_config['learning_rate'], weight_decay=self.training_config['weight_decay'])
        else:
            raise ValueError(f"Unsupported optimizer: {self.training_config['optimizer']}")
        x_src_train = torch.load(self.acquisition_config['load_dir'] + "/x_train.pt").to(self.device)
        y_src_train = torch.load(self.acquisition_config['load_dir'] + "/y_train.pt").to(self.device)
        x_src_valid = torch.load(self.acquisition_config['load_dir'] + "/x_src_valid.pt").to(self.device)
        y_src_valid = torch.load(self.acquisition_config['load_dir'] + "/y_src_valid.pt").to(self.device)
        x_tgt_valid = torch.load(self.acquisition_config['load_dir'] + "/x_tgt_valid.pt").to(self.device)
        y_tgt_valid = torch.load(self.acquisition_config['load_dir'] + "/y_tgt_valid.pt").to(self.device)
        x_tgt_test = torch.load(self.acquisition_config['load_dir'] + "/x_tgt_test.pt").to(self.device)
        y_tgt_test = torch.load(self.acquisition_config['load_dir'] + "/y_tgt_test.pt").to(self.device)
        src_train_dataset = TensorDataset(x_src_train, y_src_train)
        src_valid_dataset = TensorDataset(x_src_valid, y_src_valid)
        self.tgt_valid_dataset = TensorDataset(x_tgt_valid, y_tgt_valid)
        self.tgt_test_dataset = TensorDataset(x_tgt_test, y_tgt_test)
        
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
                'region': self.sample_wise_data.src_region,
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
        _, test_acc = self.evaluate(self.model, self.tgt_test_dataset)
        summary_loss_dict = {'valid_loss': [], 'test_loss': [], 'test_acc_from_0': [test_acc]}
        for i in range(1,int(self.acquisition_config['n_rounds']/self.acquisition_config['save_rounds'])+1):
            r = i * self.acquisition_config['save_rounds'] -1
            x_acquired = torch.load(self.acquisition_config['load_dir'] + f"/x_acquired_{r}.pt")
            y_acquired = torch.load(self.acquisition_config['load_dir'] + f"/y_acquired_{r}.pt")
            x_acquired = x_acquired.to(self.device)
            y_acquired = y_acquired.to(self.device)
            finetune_dataset = TensorDataset(x_acquired, y_acquired)
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
            summary_loss_dict['valid_loss'].append(val_loss_record)
            summary_loss_dict['test_loss'].append(test_loss)
            summary_loss_dict['test_acc_from_0'].append(test_acc)
        self.logger.info("%s", json.dumps(summary_loss_dict, indent=4))
        return summary_loss_dict['test_acc_from_0']
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
            
            
            
                
            
            
