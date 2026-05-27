from model.nn import MultiClassClassifier
from dataset import DomainWiseData
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from .trainer_utils import EarlyStopping
from strategy.pda_strategy import PDADomainWiseAcquisitionStrategy
from copy import deepcopy
import math
import json
from prettytable import PrettyTable
import pandas as pd
class PDATrainer:
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
        self.current_batch = 0
    def acquire_and_finetune(self):
        if self.finetuning_config['optimizer'] == 'adam':
            self.finetune_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.finetuning_config['learning_rate'], weight_decay=self.finetuning_config['weight_decay'])
        elif self.finetuning_config['optimizer'] == 'sgd':
            self.finetune_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.finetuning_config['learning_rate'], weight_decay=self.finetuning_config['weight_decay'])
        else:
            raise ValueError(f"Unsupported optimizer: {self.finetuning_config['optimizer']}")
        
        acquire_summary = PrettyTable()
        acquire_summary.field_names = ["Region", "Counts", "Avg Mean Reward", "Avg Exploration Bonus"]
        summary_dict = {}
        _, test_acc = self.evaluate(self.model, self.tgt_test_dataset)
        summary_loss_dict = {'valid_loss': [], 'test_loss': [], 'test_acc_from_0': [test_acc]}
        
        
        
        for b in range(self.acquisition_config['n_batches']):
            self.logger.info(f"Finetuning batch {b+1}")
            self.current_batch = b
            acquire_table, ft_log = self.acquire_and_finetune_batch()
            acquire_table = pd.DataFrame(acquire_table.rows, columns=acquire_table.field_names)
            
            for _, row in acquire_table.iterrows():
                region = row["Region"]
                if region not in summary_dict:
                    summary_dict[region] = {"Counts": [], "Mean Reward": [], "Exploration Bonus": []}
                summary_dict[region]["Counts"].append(row["Counts"])
                summary_dict[region]["Mean Reward"].append(row["Mean Reward"])
                summary_dict[region]["Exploration Bonus"].append(row["Exploration Bonus"])
            summary_loss_dict['valid_loss'].append(ft_log['tgt_region']['valid']['loss'])
            summary_loss_dict['test_loss'].append(ft_log['tgt_region']['test']['loss'])
            summary_loss_dict['test_acc_from_0'].append(ft_log['tgt_region']['test']['acc'])
            
            
        self.logger.info("Acquisition summary:")
        for region, stats in summary_dict.items():
            avg_counts = sum(stats["Counts"])
            avg_reward = sum(stats["Mean Reward"]) / len(stats["Mean Reward"])
            avg_bonus = sum(stats["Exploration Bonus"]) / len(stats["Exploration Bonus"])
            acquire_summary.add_row([region, f"{avg_counts:.2f}", f"{avg_reward:.4f}", f"{avg_bonus:.4f}"])
            
        self.logger.info("\n" + str(acquire_summary))
        self.logger.info("%s", json.dumps(summary_loss_dict, indent=4))
        return summary_loss_dict['test_acc_from_0']   
    
    def acquire_and_finetune_batch(self):
        '''
        acquire data (n_rounds_per_batch * n_samples_per_round)
        finetune the model on the acquired data
        '''
        strategy = PDADomainWiseAcquisitionStrategy(
            acquisition_config=self.acquisition_config,
            domain_wise_data=self.data
        )
        x_batch, y_batch = [], []
        other_region_names = self.data.other_regions
        for r in range(self.acquisition_config['n_rounds_per_batch']):
            n_sampls = self.acquisition_config['n_samples_per_round']
            x_acquired, y_acquired, region_idx = strategy.acquire(n_samples=n_sampls)
            x_acquired = x_acquired.to(self.device)
            y_acquired = y_acquired.to(self.device)
            reward = self.compute_reward(x_acquired, y_acquired)
            strategy.update(region_idx, reward)
            if self.acquisition_config['acquisition_function']['abandon_neg_reward_samples'] == False or reward >= 0:
                x_batch.append(x_acquired)
                y_batch.append(y_acquired)
            
        other_region_counts = strategy.counts
        other_region_values, other_region_bouns = strategy.values_and_bouns()
        acquire_log_table = PrettyTable()
        acquire_log_table.field_names = ["Region", "Counts", "Mean Reward", "Exploration Bonus"]
        for i, region_name in enumerate(other_region_names):
            acquire_log_table.add_row([region_name, other_region_counts[i], other_region_values[i], other_region_bouns[i]])
        self.logger.info("\n" + str(acquire_log_table))
        if self.acquisition_config['use_val_last_batch'] and self.current_batch == self.acquisition_config['n_batches'] - 1:
            x_batch.append(self.tgt_valid_dataset.tensors[0].to(self.device))
            y_batch.append(self.tgt_valid_dataset.tensors[1].to(self.device))
        if len(x_batch) == 0:
            self.logger.info("All regions have negative mean reward.")
            x_batch = self.tgt_valid_dataset.tensors[0].to(self.device)
            y_batch = self.tgt_valid_dataset.tensors[1].to(self.device)
        else:
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
            
        }
        self.logger.info("finetuning log: %s", json.dumps(ft_log, indent=4))
        return acquire_log_table, ft_log
        
    def compute_reward(self, x_acquired, y_acquired):
        if self.acquisition_config['acquisition_function']['reward'] == 'adam_score':
            reward = self.compute_adam_score(x_acquired, y_acquired)
        elif self.acquisition_config['acquisition_function']['reward'] == 'sgd_score':
            reward = self.compute_sgd_score(x_acquired, y_acquired)
        elif self.acquisition_config['acquisition_function']['reward'] == 'loss_reduction_score':
            reward = self.compute_loss_reduction_score(x_acquired, y_acquired)
        else:
            raise ValueError(f"Unsupported reward function: {self.acquisition_config['acquisition_function']['reward']}")
        return reward
    
    def compute_adam_score(self, x_acquired, y_acquired):
        '''
        score = delta_params(x_acquired, y_acquired)^T*grad(x_val, y_val)
        '''
        state_dict_backup = deepcopy(self.finetune_optimizer.state_dict())
        self.model.eval()
        self.model.zero_grad()
        y_acquired_pred = self.model(x_acquired)
        loss = self.criterion(y_acquired_pred, y_acquired)
        loss.backward()
        

        delta_params_groups = []
        for group in self.finetune_optimizer.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            delta_params = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)
                    state= self.finetune_optimizer.state[p]
                    
                    if len(state) == 0:
                        state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                            if group['capturable'] else torch.tensor(0.)
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    
                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                    state_steps.append(state['step'])
            
            lr=group['lr']
            
            eps=group['eps']
            maximize=group['maximize']
            capturable=group['capturable']
            amsgrad=group['amsgrad']
            
            for i, param in enumerate(params_with_grad):
                grad = grads[i] if not maximize else -grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step_t = state_steps[i]       
                if capturable:
                    assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."
                else:
                    assert not step_t.is_cuda, "If capturable=False, state_steps should not be CUDA tensors."
            
                step_t += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(param, alpha=group['weight_decay'])
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                if capturable:
                    step = step_t
                    bias_correction1 = 1 - torch.pow(beta1, step)
                    bias_correction2 = 1 - torch.pow(beta2, step)

                    step_size = lr / bias_correction1
                    step_size_neg = step_size.neg()

                    bias_correction2_sqrt = bias_correction2.sqrt()

                    if amsgrad:
                        torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                        denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
                    else:
                        denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

                    delta_params.append(-exp_avg / denom)
                else:
                    step = step_t.item()
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    step_size = lr / bias_correction1
                    bias_correction2_sqrt = math.sqrt(bias_correction2)
                    if amsgrad:
                        torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                        denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
                    else:
                        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                    delta_params.append(step_size * exp_avg / denom )
            # delta_params = torch.stack(delta_params, dim=0)                    
            delta_params_groups.append(delta_params)
            
        
        # self.finetune_optimizer.load_state_dict(state_dict)
        self.model.zero_grad()
        y_tgt_valid_pred = self.model(self.tgt_valid_dataset.tensors[0].to(self.device))
        loss = self.criterion(y_tgt_valid_pred, self.tgt_valid_dataset.tensors[1].to(self.device))
        loss.backward()
        
        
        grads_groups = []
        for group in self.finetune_optimizer.param_groups:
            grads = []
            maximize = group['maximize']
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad if not maximize else -p.grad
                    grads.append(grad)
                    
            grads_groups.append(grads)
        
        reward = 0
        for i in range(len(grads_groups)):
            for j in range(len(grads_groups[i])):
                reward += torch.sum(delta_params_groups[i][j] * grads_groups[i][j])
        
        self.finetune_optimizer.load_state_dict(state_dict_backup)
        return reward.item()
    
    def compute_loss_reduction_score(self, x_acquired, y_acquired):
        def clone_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Adam):
            """
            Clone the model and optimizer with full optimizer state (e.g., exp_avg, exp_avg_sq, step).
            
            Returns:
                cloned_model: A deepcopy of the original model.
                cloned_optimizer: A new optimizer that shares no parameters or states with the original one,
                                but keeps the same state_dict mapped to the new model.
            """

            # 1. Deepcopy the model
            cloned_model = deepcopy(model)

            # 2. Create a new optimizer with the same hyperparameters
            optimizer_class = type(optimizer)
            optimizer_kwargs = optimizer.defaults
            cloned_optimizer = optimizer_class(cloned_model.parameters(), **optimizer_kwargs)

            # 3. Map original parameters to new cloned parameters
            orig_params = list(model.parameters())
            cloned_params = list(cloned_model.parameters())
            param_id_map = {id(p): q for p, q in zip(orig_params, cloned_params)}

            # 4. Copy optimizer state
            new_state = {}
            for old_param, state in optimizer.state.items():
                if id(old_param) not in param_id_map:
                    continue
                new_param = param_id_map[id(old_param)]
                new_state[new_param] = deepcopy(state)

            # 5. Copy param_groups, replacing parameters
            new_param_groups = deepcopy(optimizer.param_groups)
            for group in new_param_groups:
                group['params'] = [param_id_map[id(p)] for p in group['params']]

            # 6. Assign state and param_groups to cloned optimizer
            cloned_optimizer.__setstate__({'state': new_state, 'param_groups': new_param_groups})

            return cloned_model, cloned_optimizer
        
        new_model, new_optimizer = clone_model_and_optimizer(self.model, self.finetune_optimizer)
        new_model.eval()
        y_tgt_valid_pred = new_model(self.tgt_valid_x)
        old_loss = self.criterion(y_tgt_valid_pred, self.tgt_valid_y).item()
        
        
        new_model.train()
        new_optimizer.zero_grad()
        y_acquired_pred = new_model(x_acquired)
        loss = self.criterion(y_acquired_pred, y_acquired)
        loss.backward()
        new_optimizer.step()
        
        
        new_model.eval()
        y_tgt_valid_pred = new_model(self.tgt_valid_x)
        new_loss = self.criterion(y_tgt_valid_pred, self.tgt_valid_y).item()

        return old_loss - new_loss
        
    def compute_sgd_score(self, x_acquired, y_acquired):
        '''
        score = lr*grad(x_acquired, y_acquired)^T*grad(x_val, y_val)
        '''
        self.model.eval()
        self.model.zero_grad()
        
        y_acquired_pred = self.model(x_acquired)
        loss = self.criterion(y_acquired_pred, y_acquired)
        grad_acquired = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        
        y_tgt_valid_pred = self.model(self.tgt_valid_dataset.tensors[0].to(self.device))
        loss = self.criterion(y_tgt_valid_pred, self.tgt_valid_dataset.tensors[1].to(self.device))
        grad_valid = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        
        reward = sum(torch.sum(g1 * g2) for g1, g2 in zip(grad_acquired, grad_valid)) * self.finetuning_config['learning_rate']
        return reward.item() 
        
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
            
            
            
                
            
            
