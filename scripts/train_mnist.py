import torch
import os
from torch import nn
from torchvision import datasets, transforms
import msgspec
from torch.utils.tensorboard import SummaryWriter
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from pytorch_modules import engine, utils

import scripts.models as models


class HyperParameters(msgspec.Struct):
    input_shape : int
    hidden_units : int
    output_shape : int
    epochs : int
    batch_size : int
    learning_rate : float
    optimizer : torch.optim.Optimizer
    loss : nn.Module
    
def main():
    model_class = models.MNISTConvModelv0
    SAVE_MODEL = True

    # Set hyperparameters
    hyperparameters = HyperParameters(
        input_shape = 28 * 28,
        hidden_units = 128,
        output_shape = 10,
        epochs = 20,
        batch_size = 32,
        learning_rate = 0.01,
        optimizer = "SGD",
        loss = "CrossEntropyLoss"
    )

    runs_dir = os.path.join(os.path.dirname(__file__), 'runs')
    experiment_name = f"{model_class.__name__}_{utils.get_next_experiment_number(runs_dir)}"
    run_save_path = os.path.join(runs_dir, experiment_name)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load MNIST data
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=hyperparameters.batch_size, shuffle=False)

    # Initialize model
    model = model_class(hyperparameters.input_shape, hyperparameters.hidden_units, hyperparameters.output_shape).to(device)

    # Initialize optimizer

    optimizer_class : torch.optim.Optimizer = getattr(torch.optim, hyperparameters.optimizer)
    optimizer = optimizer_class(model.parameters(), lr=hyperparameters.learning_rate)

    # Initialize loss function
    loss_fn : nn.Module = getattr(nn, hyperparameters.loss)()

    # Initialize tensorboard writer

    with SummaryWriter(run_save_path) as writer:
        writer.add_hparams(msgspec.structs.asdict(hyperparameters), {}, run_name=run_save_path)
        trained_model = engine.train(model, 
                                     train_dataloader, 
                                     test_dataloader, 
                                     loss_fn, 
                                     optimizer, 
                                     device, 
                                     hyperparameters.epochs, 
                                     writer)

    # Save model
    if SAVE_MODEL:
        utils.save_model(trained_model, experiment_name)

if __name__ == '__main__':
    main()