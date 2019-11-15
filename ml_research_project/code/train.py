import os
import torch

def train(model, criterion, optimizer, scheduler, train_loader, val_loader, save_path, epochs=25):
    """Training process."""
    for epoch in range(epochs):
        model.train()
        # Training
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            # Model computations
            inputs, angles = data
            inputs = inputs.float().cuda()
            # output and target should be same shape (unsqueeze)
            angles = angles.float().cuda()
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

        if epoch % 5 == 0:
            # print val loss and save model every 5 epochs
            if save_path is not None:
                torch.save(model, save_path)

            model.eval()
            running_val_loss = 0
            for i, data in enumerate(val_loader):
                inputs, angles = data
                inputs = inputs.float().cuda()
                # output and target should be same shape (unsqueeze)
                angles = angles.float().cuda()
                outputs = model(inputs)
                loss = criterion(outputs, angles)
                running_val_loss += loss.item()

            print("Validation Loss (average over validation set): {}".format(running_val_loss / len(val_loader.dataset)))
