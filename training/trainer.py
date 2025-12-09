import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from utils.iq_extractor import extract_iq_data

class IQCMAE_Trainer:
    """
    Trainer class for IQ-CMAE models.
    Handles training loop, validation, and checkpointing.
    """
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 data_loader_train: torch.utils.data.DataLoader, 
                 data_loader_val: Optional[torch.utils.data.DataLoader], 
                 device: torch.device, 
                 output_dir: str, 
                 epochs: int, 
                 start_epoch: int = 0, 
                 writer: Optional[Any] = None, 
                 print_freq: int = 20, 
                 args: Optional[Any] = None):
        
        self.model = model
        self.optimizer = optimizer
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.device = device
        self.output_dir = output_dir
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.print_freq = print_freq
        self.args = args
        self.scaler = torch.cuda.amp.GradScaler()
        self.writer = writer

    def train(self):
        """Run full training loop."""
        for epoch in range(self.start_epoch, self.epochs):
            train_stats = self.train_one_epoch(epoch)
            
            val_stats = {}
            if self.data_loader_val is not None:
                val_stats = self.evaluate(epoch)
                print(f"Epoch {epoch}: Train Loss {train_stats['loss']:.4f} | Val Loss {val_stats['loss']:.4f}")
            else:
                print(f"Epoch {epoch}: Train Loss {train_stats['loss']:.4f}")
            
            # Save checkpoint
            if self.output_dir:
                self._save_checkpoint(epoch, train_stats, val_stats)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.data_loader_train)
        
        for i, samples in enumerate(self.data_loader_train):
            # Handle dictionary or tuple inputs
            clean_imgs, noisy_imgs = self._unpack_samples(samples)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # Model forward pass
                # Returns: loss, mae_loss, contrastive_loss, pred, mask
                loss, mae_loss, contrastive_loss, _, _ = self.model(clean_imgs, noisy_imgs)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            if i % self.print_freq == 0:
                print(f"Epoch: [{epoch}][{i}/{num_batches}] "
                      f"Loss: {loss.item():.4f} "
                      f"MAE: {mae_loss.item():.4f} "
                      f"Contrastive: {contrastive_loss.item():.4f}")
        
        if num_batches == 0:
            return {'loss': 0.0}
        return {'loss': total_loss / num_batches}

    def evaluate(self, epoch: int) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.data_loader_val)
        
        with torch.no_grad():
            for i, samples in enumerate(self.data_loader_val):
                clean_imgs, noisy_imgs = self._unpack_samples(samples)

                with torch.cuda.amp.autocast():
                    loss, _, _, _, _ = self.model(clean_imgs, noisy_imgs)
                
                total_loss += loss.item()
        
        return {'loss': total_loss / num_batches}

    def _unpack_samples(self, samples: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Unpack samples from dataset into (clean, noisy)."""
        if isinstance(samples, dict):
            # Handle NEDataRawDataset keys
            if 'image' in samples:
                clean_imgs = samples['image'].to(self.device, non_blocking=True)
                noisy_imgs = samples.get('teacher_image', None)
                if noisy_imgs is not None:
                    noisy_imgs = noisy_imgs.to(self.device, non_blocking=True)
            # Handle legacy keys
            elif 'clean' in samples:
                clean_imgs = samples['clean'].to(self.device, non_blocking=True)
                noisy_imgs = samples.get('noisy', None)
                if noisy_imgs is not None:
                    noisy_imgs = noisy_imgs.to(self.device, non_blocking=True)
            else:
                raise KeyError("Dataset dictionary must contain 'image' or 'clean' keys")
        else:
            # Fallback for tuple/list
            clean_imgs = samples[0].to(self.device, non_blocking=True)
            noisy_imgs = None
        return clean_imgs, noisy_imgs

    def _save_checkpoint(self, epoch: int, train_stats: Dict[str, float], val_stats: Dict[str, float]):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{epoch}.pth")
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'train_stats': train_stats,
            'val_stats': val_stats,
        }
        if self.args:
            save_dict['args'] = self.args
            
        torch.save(save_dict, checkpoint_path)
        
        # Save last checkpoint pointer
        torch.save(save_dict, os.path.join(self.output_dir, "checkpoint-last.pth"))
