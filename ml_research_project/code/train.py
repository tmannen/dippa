import os
import torch

def train(model, criterion, optimizer, scheduler, dataloader, epochs=25):
    """Training process."""
    model.train()
    for epoch in range(epochs):

        # Training
        running_loss = 0.0

        for i, data in enumerate(dataloader):
            # Model computations
            inputs, angles = data
            inputs = inputs.float().cuda()
            # output and target should be same shape (unsqueeze)
            angles = angles.float().unsqueeze(1).cuda()
            optimizer.zero_grad()
            # print("training image: ", imgs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, angles)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        
        print("Epoch loss: ", running_loss)
        running_loss = 0.0

        scheduler.step()
