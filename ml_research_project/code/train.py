import os
import torch

def train(model, criterion, optimizer, scheduler, dataloader, epochs=5):
    """Training process."""
    model.train()
    for epoch in range(epochs):

        # Training
        train_loss = 0.0

        for i, data in enumerate(dataloader):
            # Model computations
            inputs, angles = data
            inputs = inputs.float().cuda()
            angles = angles.float().cuda()
            optimizer.zero_grad()
            # print("training image: ", imgs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, angles)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        scheduler.step()
