import wandb

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from tqdm import tqdm
import random

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
random.seed(42)

def get_dataloader(is_train, batch_size, slice=5):
    """
    Get a training dataloader
    """
    full_dataset = torchvision.datasets.MNIST(
        root='.', 
        train=is_train, 
        transform=T.ToTensor(), 
        download=True
    )
    sub_dataset = torch.utils.data.Subset(
        full_dataset, 
        indices=range(0, len(full_dataset), 
        slice)
    )
    loader = torch.utils.data.DataLoader(
        dataset=sub_dataset, 
        batch_size=batch_size, 
        shuffle=True if is_train else False, 
        pin_memory=True, 
        num_workers=2
    )

    return loader

def get_model(dropout):
    """
    A simple model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256,10)
    ).to(device)

    return model

def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    """
    Compute performance of the model on the validation dataset and log a wandb.Table
    """
    model.eval()
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in tqdm(enumerate(valid_dl), leave=False):
            images, labels = images.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels)*labels.size(0)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Log one batch of images to the dashboard, always same batch_idx.
            if i==batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))

    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

def log_image_table(images, predicted, labels, probs):
    """
    Log a wandb.Table with (img, pred, target, scores)
    """
    table = wandb.Table(columns=['image', 'pred', 'target']+[f'score_{i}' for i in range(10)])
    for img, pred, targ, prob in zip(images.to('cpu'), predicted.to('cpu'), labels.to('cpu'), probs.to('cpu')):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
        
    wandb.log({'predictions_table':table}, commit=False)

# Launch 10 experiments, trying different dropout rates
for i in range(10):
    with wandb.init(
        project='my-test-project',
        entity='davidguzmanr',
        group='PyTorch',
        name=f'run-{i}',
        config={
            'epochs': 10,
            'batch_size': 128,
            'lr': 1e-3,
            'dropout': random.uniform(0.01, 0.80),
            }
    ):
        config = wandb.config

        # Get the data
        train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
        valid_dl = get_dataloader(is_train=False, batch_size=2*config.batch_size)

        # A simple MLP model
        model = get_model(config.dropout)

        # Make the loss and optimizer
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # Training
        example_ct = 0
        for epoch in tqdm(range(config.epochs)):
            model.train()
            for images, labels in tqdm(train_dl, leave=False):
                images, labels = images.to(device), labels.to(device)
        
                # Forward pass
                outputs = model(images)
                train_loss = loss_func(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                train_loss.backward()
                
                # Step with optimizer
                optimizer.step()

                # Log training loss and epoch count
                example_ct += len(images)
                wandb.log({
                    'train_loss': train_loss, 
                    'epoch': example_ct/len(train_dl.dataset)
                    }, 
                    step=example_ct
                )
            
            val_loss, accuracy = validate_model(model, valid_dl, loss_func, log_images=(epoch==(config.epochs-1)))
            
            # Log validation metrics
            wandb.log({
                'val_loss': val_loss, 
                'val_accuracy': accuracy
                }, 
                step=example_ct
            )
            print(f'Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, accuracy: {accuracy:.2f}')

wandb.finish()