import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

EPS = 1E-20

class AdvWeightPerturb:
    def __init__(self, model, delta=0.1, eps=1e-6, use_mixed_precision=False, scaler=None, cfg=None):
        self.model = model
        self.cfg = cfg
        self.delta = delta
        self.eps = eps
        self.use_mixed_precision = use_mixed_precision
        self.scaler = scaler if scaler is not None else GradScaler()
        self.params = list(model.parameters())  # Cache parameters list for iteration

    def calc_awp_and_apply(self, batch):
        self.model.zero_grad()

        if self.use_mixed_precision:
            with autocast(device_type=self.cfg.device):
                output_dict = self.model(batch)
                loss = output_dict["loss"]
            self.scaler.scale(loss).backward()
        else:
            output_dict = self.model(batch)
            loss = output_dict["loss"]
            loss.backward()

        # Compute and apply perturbations, save them for restore
        perturbations = []
        with torch.no_grad():
            for param in self.params:
                pert = None
                if param.grad is not None:
                    grad = param.grad.data
                    norm = torch.norm(grad)
                    if norm > 0:
                        pert = self.delta * grad / (norm + self.eps)
                        param.data.add_(pert)
                perturbations.append(pert)

        return loss, perturbations

    def restore_weights(self, perturbations):
        with torch.no_grad():
            for i, param in enumerate(self.params):
                pert = perturbations[i]
                if pert is not None:
                    param.data.sub_(pert)

    def train_step(self, optimizer, batch):
        # First pass: compute loss, backward, perturb weights, get perturbations
        _, perturbations = self.calc_awp_and_apply(batch)

        # Second pass: compute loss on perturbed model
        self.model.zero_grad()
        if self.use_mixed_precision:
            with autocast(device_type=self.cfg.device):
                output_dict = self.model(batch)
                loss = output_dict["loss"]
            self.scaler.scale(loss).backward()
        else:
            output_dict = self.model(batch)
            loss = output_dict["loss"]
            loss.backward()

        # Restore original weights using saved perturbations
        self.restore_weights(perturbations)

        # Optimizer step
        if self.use_mixed_precision:
            if self.cfg.clip_grad > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if self.cfg.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
            optimizer.step()

        optimizer.zero_grad()

        return loss