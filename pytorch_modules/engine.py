import torch
import msgspec
from typing import Literal
from torch.utils.data import DataLoader
from torch import nn, optim
from torchmetrics import Accuracy
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

def train_step(dataloader : DataLoader, 
               model : nn.Module, 
               optimizer : optim.Optimizer, 
               loss_fn : nn.Module, 
               device : torch.device, accuracy : Accuracy, 
               ) -> tuple[float, float]:
    model.train()
    total_loss = 0
    data : torch.Tensor
    target : torch.Tensor
    for data, target in tqdm(dataloader,desc="Training", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss : torch.Tensor = loss_fn(output, target)
        accuracy.update(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    average_accuracy = accuracy.compute()
    return average_loss, average_accuracy


def test_step(model : nn.Module,
              dataloader : DataLoader,
              loss_fn : nn.Module,
              device : torch.device,
              accuracy : Accuracy) -> tuple[float, float]:
    model.eval()
    total_loss = 0
    data : torch.Tensor
    target : torch.Tensor
    with torch.inference_mode():
        for data, target in tqdm(dataloader, desc="Testing", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss : torch.Tensor = loss_fn(output, target)
            accuracy.update(output, target)
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    average_accuracy = accuracy.compute()
    return average_loss, average_accuracy


def train(model: nn.Module, 
          train_dataloader: DataLoader, 
          test_dataloader: DataLoader, 
          loss_fn: nn.Module, 
          optimizer: optim.Optimizer, 
          device: torch.device, 
          epochs: int,
          writer : SummaryWriter) -> nn.Module:
    
    accuracy = Accuracy(task='multiclass', num_classes=len(test_dataloader.dataset.classes)).to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss, train_accuracy = train_step(train_dataloader, model, optimizer, loss_fn, device, accuracy)
        test_loss, test_accuracy = test_step(model, test_dataloader, loss_fn, device, accuracy)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/test', test_accuracy, epoch)
    print("Training finished")
    return model
    