import time
import os
import torch
import csv
import metric
from torch.utils.tensorboard import SummaryWriter

class Train:
    def __init__(self, **kwargs):

        self.result_path = kwargs['result_path']
        self.tensorboard_path = os.path.join(self.result_path, 'tensorboard')
        
        self.log_path = os.path.join(self.result_path, 'log_%d.csv'%kwargs['fold_idx'])
        if os.path.isfile(self.log_path):
            os.remove(self.log_path)
        self.tb_writer = SummaryWriter(log_dir=self.tensorboard_path)

        self.device = kwargs['device']
        self.model = kwargs['model']
        self.optimizer = kwargs['optimizer']
        self.criterion = kwargs['criterion']
        self.task = kwargs['task']
        self.fold_idx = kwargs['fold_idx']
        self.epochs = kwargs['epochs']
        self.batch_size = kwargs['batch_size']
        self.period = kwargs['period']
        self.train_data_loader = kwargs['train_data_loader']
        self.val_data_loader = kwargs['val_data_loader']
        self.scheduler = kwargs['scheduler']
        self.is_plateau = kwargs['is_plateau']

        self.loss = {}

        self.rmse = {}
        self.accuracy = {}
        self.precision = {}
        self.recall = {}
        self.specificity = {}
        self.npv = {}
        self.f1score = {}

    def log_regression(self, output, target, _mode):
        _rmse = metric.gait_phase_rmse(output=output, target=target)
        self.rmse[_mode] += _rmse

    def log_classification(self, output, target, _mode):
        _output = torch.sigmoid(output)
        _output = (_output > 0.5).float() 
        metrics = metric.classification_metrics(_output, target)
        self.accuracy[_mode] += metrics['accuracy']
        self.precision[_mode] += metrics['precision']
        self.recall[_mode] += metrics['recall']
        self.specificity[_mode] += metrics['specificity']
        self.npv[_mode] += metrics['npv']
        self.f1score[_mode] += metrics['f1score']

    def log(self, output, target, _mode):
        if self.task == 'regression':
           self.log_regression(output, target, _mode)
        elif self.task == 'classification':
            self.log_classification(output, target, _mode)
        elif self.task == 'multi':
            out_r, out_c = output
            self.log_regression(out_r, target[:, :-1], _mode)
            self.log_classification(out_c, target[:, -1], _mode)
        
    def write_regression(self, epoch):
        self.tb_writer.add_scalars('rmse_%d'%self.fold_idx, self.rmse, epoch)
    
    def write_classification(self, epoch):
        self.tb_writer.add_scalars('accuracy_%d'%self.fold_idx, self.accuracy, epoch)
        self.tb_writer.add_scalars('precision_%d'%self.fold_idx, self.precision, epoch)
        self.tb_writer.add_scalars('recall_%d'%self.fold_idx, self.recall, epoch)
        self.tb_writer.add_scalars('specificity_%d'%self.fold_idx, self.specificity, epoch)
        self.tb_writer.add_scalars('npv_%d'%self.fold_idx, self.npv, epoch)
        self.tb_writer.add_scalars('f1score_%d'%self.fold_idx, self.f1score, epoch)

    def write(self, epoch, end_time):
        self.tb_writer.add_scalars('loss_%d'%self.fold_idx, self.loss, epoch)
        if self.task == 'regression':
            print('Epoch: %d | Loss: %1.5f | RMSE: %1.5f | Elapsed Time: %0f'%(epoch, self.loss['val'], self.rmse['val'], end_time))
            with open(self.log_path, 'a', newline='') as f:
                csv_writer = csv.DictWriter(f, fieldnames=['loss', 'val_loss', 'rmse', 'val_rmse'])
                csv_writer.writerow({'loss': self.loss['train'], 'rmse': self.rmse['train'],
                                    'val_loss': self.loss['val'], 'val_rmse': self.rmse['val']})
            self.write_regression(epoch)
        elif self.task == 'classification':
            print('Epoch: %d | Loss: %1.5f | Accuracy: %1.5f | Elapsed Time: %0f'%(epoch, self.loss['val'], self.accuracy['val'], end_time))
            with open(self.log_path, 'a', newline='') as f:
                csv_writer = csv.DictWriter(f, fieldnames=['loss', 'val_loss', 'acc', 'val_acc'])
                csv_writer.writerow({'loss': self.loss['train'], 'acc': self.accuracy['train'],
                                    'val_loss': self.loss['val'], 'val_acc': self.accuracy['val']})
            self.write_classification(epoch)
        elif self.task == 'multi':
            print('Epoch: %d | Loss: %1.5f | RMSE: %1.5f | Accuracy: %1.5f | Elapsed Time: %0f'%(epoch, self.loss['val'], self.rmse['val'], self.accuracy['val'], end_time))
            self.write_regression(epoch)
            self.write_classification(epoch)

    def on_epoch_validation(self):
        self.model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(self.val_data_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                if x.size(0) == self.batch_size:
                    output = self.model(x)
                    self.log(output, y, 'val')
                    
                    if isinstance(self.criterion, dict) and isinstance(output, tuple):
                    
                        out_r, out_c = output

                        regression_criterion = self.criterion['regression']
                        regression_loss = regression_criterion(out_r, y[:, :-1])
                        classification_criterion = self.criterion['classification']
                        classification_loss = classification_criterion(out_c.squeeze(), y[:, -1])

                        _loss = regression_loss + classification_loss
                        self.loss['val'] += _loss.item()
                    
                    else:
                        _loss = self.criterion(output, y)
                        self.loss['val'] += _loss.item()

    def on_epoch_train(self):
        self.model.train()
        self.optimizer.zero_grad()
        for _, (x, y) in enumerate(self.train_data_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            if x.size(0) == self.batch_size:
                output = self.model(x)
                self.log(output, y, 'train')

                if isinstance(self.criterion, dict) and isinstance(output, tuple):
                    
                    out_r, out_c = output

                    #########################################################
                    # preventing the model from training swing phase
                    # mask = torch.stack([y[:, -1], y[:, -1]], dim=1)
                    # out_r.register_hook(lambda grad: grad * mask)
                    #########################################################

                    regression_criterion = self.criterion['regression']
                    regression_loss = regression_criterion(out_r, y[:, :-1])
                    
                    classification_criterion = self.criterion['classification']
                    classification_loss = classification_criterion(out_c.squeeze(), y[:, -1])
                    
                    self.optimizer.zero_grad()
                    regression_loss.backward(retain_graph=True)
                    classification_loss.backward(retain_graph=True)

                    self.optimizer.step()
                    self.loss['train'] += regression_loss.item() + classification_loss.item()

                else:
                    _loss = self.criterion(output, y)
                    self.optimizer.zero_grad()
                    _loss.backward()
                    self.optimizer.step()
                    self.loss['train'] += _loss.item()

    def do(self):
        print('K-fold: %d'%self.fold_idx)
        tb_writer = SummaryWriter(log_dir=self.tensorboard_path)
        
        best_val_loss = float('inf')
        for epoch in range(1, self.epochs + 1):
            
            for metric in [self.loss, self.rmse, self.accuracy, self.precision, self.recall, self.specificity, self.npv, self.f1score]:
                metric['train'] = 0
                metric['val'] = 0
            
            start_time = time.time()
            self.on_epoch_train()
            self.on_epoch_validation()
            end_time = time.time() - start_time

            for metric in [self.loss, self.rmse, self.accuracy, self.precision, self.recall, self.specificity, self.npv, self.f1score]:
                metric['train'] = metric['train'] / len(self.train_data_loader)
                metric['val'] = metric['val'] / len(self.val_data_loader)

            self.write(epoch, end_time)
            
            # saving the model
            if epoch % self.period == 0:
                model_path = os.path.join(self.result_path, 'model_%d_%d.pt'%(self.fold_idx, epoch))
                torch.save(self.model, model_path)
            if  self.loss['val'] < best_val_loss:
                model_path = os.path.join(self.result_path, 'model_%d_best.pt'%(self.fold_idx))
                torch.save(self.model, model_path)
                print('Current Validation Loss: %1.5f | New validation Loss: %1.5f'%(best_val_loss, self.loss['val']))
                best_val_loss = self.loss['val']
        
            if self.is_plateau:
                self.scheduler.step(best_val_loss)
            else:
                self.scheduler.step()

        tb_writer.close()