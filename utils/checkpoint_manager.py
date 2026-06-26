"""
Checkpoint Manager for v2 Models

This module provides a unified interface for saving and loading checkpoints
with a hierarchical directory structure:
ckpt/{dataset}/{modality}/{model_name}/{frame_length}_fold{fold}_step{step}/
"""

import os
import torch
from typing import Dict, Any, Optional, Tuple


class CheckpointManager:
    """Manages checkpoint saving and loading with hierarchical directory structure"""
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
    
    def _get_checkpoint_dir(self, dataset: str, modality: str, model_name: str, 
                           frame_length: int, fold: int, step: int) -> str:
        """Generate checkpoint directory path"""
        dir_name = f"frame{frame_length}_fold{fold}_step{step}"
        return os.path.join(self.base_dir, dataset, modality, model_name, dir_name)
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, loss: float, dataset: str, modality: str, 
                       model_name: str, frame_length: int, fold: int, step: int,
                       is_best: bool = False, additional_info: Optional[Dict[str, Any]] = None) -> str:
        """Save model checkpoint"""
        checkpoint_dir = self._get_checkpoint_dir(dataset, modality, model_name, 
                                                frame_length, fold, step)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'dataset': dataset,
            'modality': modality,
            'model_name': model_name,
            'frame_length': frame_length,
            'fold': fold,
            'step': step
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, 'latest_model.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if applicable
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            return best_path
        
        return latest_path
    
    def save_fusion_checkpoints(self, generator: torch.nn.Module, discriminator: torch.nn.Module,
                               gen_optimizer: torch.optim.Optimizer, disc_optimizer: torch.optim.Optimizer,
                               epoch: int, gen_loss: float, disc_loss: float, dataset: str,
                               frame_length: int, fold: int, step: int, is_best: bool = False,
                               additional_info: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Save fusion model checkpoints (generator and discriminator)"""
        checkpoint_dir = self._get_checkpoint_dir(dataset, 'fusion', 'simpleUMAP',
                                                frame_length, fold, step)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Generator checkpoint
        gen_checkpoint = {
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': gen_optimizer.state_dict(),
            'loss': gen_loss,
            'dataset': dataset,
            'modality': 'fusion',
            'model_name': 'FusionModel',
            'model_type': 'generator',
            'frame_length': frame_length,
            'fold': fold,
            'step': step
        }
        
        # Discriminator checkpoint
        disc_checkpoint = {
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': disc_optimizer.state_dict(),
            'loss': disc_loss,
            'dataset': dataset,
            'modality': 'fusion',
            'model_name': 'FusionModel',
            'model_type': 'discriminator',
            'frame_length': frame_length,
            'fold': fold,
            'step': step
        }
        
        if additional_info:
            gen_checkpoint.update(additional_info)
            disc_checkpoint.update(additional_info)
        
        # Save latest checkpoints
        gen_latest_path = os.path.join(checkpoint_dir, 'generator_latest_model.pth')
        disc_latest_path = os.path.join(checkpoint_dir, 'discriminator_latest_model.pth')
        torch.save(gen_checkpoint, gen_latest_path)
        torch.save(disc_checkpoint, disc_latest_path)
        
        # Save best checkpoints if applicable
        if is_best:
            gen_best_path = os.path.join(checkpoint_dir, 'generator_best_model.pth')
            disc_best_path = os.path.join(checkpoint_dir, 'discriminator_best_model.pth')
            torch.save(gen_checkpoint, gen_best_path)
            torch.save(disc_checkpoint, disc_best_path)
            return gen_best_path, disc_best_path
        
        return gen_latest_path, disc_latest_path
    
    def load_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       dataset: str, modality: str, model_name: str, frame_length: int,
                       fold: int, step: int, load_best: bool = True) -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint_dir = self._get_checkpoint_dir(dataset, modality, model_name, 
                                                frame_length, fold, step)
        
        checkpoint_file = 'best_model.pth' if load_best else 'latest_model.pth'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def load_fusion_checkpoints(self, generator: torch.nn.Module, discriminator: torch.nn.Module,
                               gen_optimizer: torch.optim.Optimizer, disc_optimizer: torch.optim.Optimizer,
                               dataset: str, frame_length: int, fold: int, step: int,
                               load_best: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load fusion model checkpoints"""
        checkpoint_dir = self._get_checkpoint_dir(dataset, 'fusion', 'FusionModel', 
                                                frame_length, fold, step)
        
        checkpoint_file = 'best_model.pth' if load_best else 'latest_model.pth'
        gen_checkpoint_path = os.path.join(checkpoint_dir, f'generator_{checkpoint_file}')
        disc_checkpoint_path = os.path.join(checkpoint_dir, f'discriminator_{checkpoint_file}')
        
        if not os.path.exists(gen_checkpoint_path) or not os.path.exists(disc_checkpoint_path):
            raise FileNotFoundError(f"Fusion checkpoints not found in: {checkpoint_dir}")
        
        gen_checkpoint = torch.load(gen_checkpoint_path, map_location='cpu')
        disc_checkpoint = torch.load(disc_checkpoint_path, map_location='cpu')
        
        generator.load_state_dict(gen_checkpoint['model_state_dict'])
        discriminator.load_state_dict(disc_checkpoint['model_state_dict'])
        gen_optimizer.load_state_dict(gen_checkpoint['optimizer_state_dict'])
        disc_optimizer.load_state_dict(disc_checkpoint['optimizer_state_dict'])
        
        return gen_checkpoint, disc_checkpoint
    
    def get_checkpoint_path(self, dataset: str, modality: str, model_name: str,
                           frame_length: int, fold: int, step: int, load_best: bool = True) -> str:
        """Get checkpoint file path"""
        checkpoint_dir = self._get_checkpoint_dir(dataset, modality, model_name, 
                                                frame_length, fold, step)
        checkpoint_file = 'best_model.pth' if load_best else 'latest_model.pth'
        return os.path.join(checkpoint_dir, checkpoint_file)
    
    def get_fusion_checkpoint_paths(self, dataset: str, frame_length: int, fold: int, step: int,
                                   load_best: bool = True) -> Tuple[str, str]:
        """Get fusion checkpoint file paths"""
        checkpoint_dir = self._get_checkpoint_dir(dataset, 'fusion', 'FusionModel', 
                                                frame_length, fold, step)
        checkpoint_file = 'best_model.pth' if load_best else 'latest_model.pth'
        gen_path = os.path.join(checkpoint_dir, f'generator_{checkpoint_file}')
        disc_path = os.path.join(checkpoint_dir, f'discriminator_{checkpoint_file}')
        return gen_path, disc_path
    
    def list_checkpoints(self, dataset: str, modality: str, model_name: str) -> list:
        """List available checkpoints for a given dataset/modality/model"""
        model_dir = os.path.join(self.base_dir, dataset, modality, model_name)
        if not os.path.exists(model_dir):
            return []
        
        checkpoints = []
        for dir_name in os.listdir(model_dir):
            if dir_name.startswith('frame') and '_fold' in dir_name and '_step' in dir_name:
                checkpoint_dir = os.path.join(model_dir, dir_name)
                if os.path.isdir(checkpoint_dir):
                    checkpoints.append(dir_name)
        
        return sorted(checkpoints)
