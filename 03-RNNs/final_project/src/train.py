import random
import torch
import torch.nn as nn
import torch.optim as optim
from src.vocab import get_tensors_from_pair
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def after_subplot(ax: plt.Axes, group_name: str, x_label: str):
    """Add title xlabel and legend to single chart"""
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc="center right")

    if group_name.lower() == "loss":
        ax.set_ylim([None, 4.5])


def train(
    source_data, 
    target_data, 
    model, 
    epochs, 
    batch_size, 
    print_every, 
    learning_rate, 
    interactive_tracking=False
):
    # Initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None
    
    total_training_loss = 0
    total_valid_loss = 0
    valid_loss_min = None
    loss = 0
    logs = {}

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.NLLLoss()

    # Use cross validation
    kf = KFold(n_splits=epochs, shuffle=True)
    for e, (train_index, test_index) in enumerate(kf.split(source_data), 1):

        # Training
        total_training_loss = 0  
        model.train()
        for i in tqdm(
            range(0, len(train_index)), 
            desc="{}/{} - Training".format(e, epochs), 
            total=len(train_index), 
            leave=False, 
            ncols=80
        ):
            optimizer.zero_grad()
            
            src = source_data[i]
            trg = target_data[i]

            output = model(src, trg)

            current_train_loss = 0
            for (s, t) in zip(output["decoder_output"], trg): 
                current_train_loss += criterion(s, t)

            loss += current_train_loss
            total_training_loss += (current_train_loss.item() / trg.size(0)) # add the iteration loss

            current_train_loss.backward()
            optimizer.step()

        # Validating 
        total_valid_loss = 0 
        model.eval()
        for i in tqdm(
            range(0, len(test_index)), 
            desc="{}/{} - Validating".format(e, epochs), 
            total=len(test_index), 
            leave=False, 
            ncols=80
        ):
            src = source_data[i]
            trg = target_data[i]

            output = model(src, trg)

            current_val_loss = 0
            for (s, t) in zip(output["decoder_output"], trg): 
                current_val_loss += criterion(s, t)

            total_valid_loss += (current_val_loss.item() / trg.size(0)) # add the iteration loss

        avg_train_loss = total_training_loss / len(train_index)
        avg_valid_loss = total_valid_loss / len(test_index)
        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or (
                (valid_loss_min - avg_valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {avg_valid_loss:.6f}. Saving model ...")

            # Save the weights to save_path
            torch.save(model.state_dict(), "checkpoints/" + model.model_name + "_best_loss.pt")

            valid_loss_min = avg_valid_loss

        if e % print_every == 0:
            print("{}/{} Epoch  -  Training Loss = {:.4f}  -  Validation Loss = {:.4f}".format(e, epochs, avg_train_loss, avg_valid_loss))

        # Update learning rate, i.e., make a step in the learning rate scheduler
        scheduler.step()
        
        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = avg_train_loss
            logs["val_loss"] = avg_valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()
