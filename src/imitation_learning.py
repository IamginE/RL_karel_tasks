import torch
import numpy as np
import random
import os

from data_loading import Dataset_Supervision
from trainer import FF_Trainer
from plot import plot

from networks import Policy_Network
import torch.optim as optim
from torch import nn


def pretrain(data_path:str, save_path:str, batch_size:int, epochs:int, 
    loss, model, optimizer, device, vec_size=4*4*3+6) -> None:
    r"""Pretrains a model on the given data and saves the model parameter to a file.
    :param data_path (str): Path to .csv-file with pretraining data.
    :param save_path (str): Path to the file, where the model is saved.
    :param batch_size (int): Batch size to use for training.
    :param epochs (int): Number of epochs to train.
    :param loss: Loss function used in training.
    :param model: Network to train.
    :param optimizer: Optimizer used for training.
    :param device: Device used in training (cpu or gpu).
    :param vec_size (int): Size of a vector representing a state.
    """

    train_batch_size = batch_size
    train_kwargs = {'batch_size': train_batch_size}

    if device.type == 'cuda':
        use_cuda = True
    else:
        use_cuda = False
  
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(cuda_kwargs)
  
    training_data = Dataset_Supervision(csv_file=data_path, vec_size=vec_size)
    train_loader = torch.utils.data.DataLoader(training_data, **train_kwargs)

    trainer = FF_Trainer(model, train_loader, None, optimizer, loss, device)

    for _ in range(epochs):
        trainer.train(10)
    torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, save_path)

def test_params(data_path:str, test_data_path:str, plot_path:str, logging_path:str, id:str, lrs:list, seed:int, epochs:int, x_ticks:list,
    device, vec_size=4*4*3+6, batch_size:int=64, loss = nn.CrossEntropyLoss()) -> None:
    r"""Trains the a model for various learning rates and plots performance measures for comparison in a .png file.
    :param data_path (str): Path to .csv-file with training data.
    :param test_data_path (str): Path to .csv-file with testing data.
    :param plot_path (str): Path to the folder, where plots are saved
    :param logging_path (str): Path to folder, where logs are written.
    :param id (str): Identifier for the plots and logging, prefix of the respective file names.
    :param lrs (list): List of learning rates.
    :param seed (int): Seed used for fixing random seeds.
    :param epochs (int): Number of epochs to train.
    :param x_ticks (list): List of x-ticks for the plot.
    :param device: Device used in training (cpu or gpu).
    :param vec_size (int): Size of a vector representing a state.
    :param batch_size (int): Batch size used in training.
    :param loss: Loss function used in training.
    """

    # fix random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_batch_size = batch_size
    test_batch_size = batch_size
    train_kwargs = {'batch_size': train_batch_size}
    test_kwargs = {'batch_size': test_batch_size}

    if device.type == 'cuda':
        use_cuda = True
    else:
        use_cuda = False
  
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(cuda_kwargs)
  
    training_data = Dataset_Supervision(csv_file=data_path, vec_size=vec_size)
    train_loader = torch.utils.data.DataLoader(training_data, **train_kwargs)
    test_data = Dataset_Supervision(csv_file=test_data_path, vec_size=vec_size)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    training_losses = []
    training_accuracies = []
    test_losses = []
    test_accuracies = []

    for lr in lrs:
        logging = logging_path + "/" + id +  "_log_lr_" + str(lr).replace(".", "_") + ".csv"
        if os.path.isfile(logging):
            raise RuntimeError("CSV with the same name exists: {}".format(logging))

        training_loss = []
        training_accuracy = []
        test_loss = []
        test_accuracy = []

        model = Policy_Network(54, 6, False)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr = lr)
        trainer = FF_Trainer(model, train_loader, test_loader, optimizer, loss, device)

        with open(logging, 'w') as fp:
            fp.write("Epoch, Training Loss, Training Accuracy, Test Loss, Test Accuracy\n")
    
            for e in range(epochs):
                epoch_loss, epoch_accuracy = trainer.train(100)
                training_loss.append(epoch_loss)
                training_accuracy.append(epoch_accuracy)
                epoch_test_loss, epoch_test_accuracy = trainer.evaluate()
                test_loss.append(epoch_test_loss)
                test_accuracy.append(epoch_test_accuracy)

                out_str = "{}, {:.4f}, {:.2f}, {:.4f}, {:.2f} \n".format(
                    e, epoch_loss, epoch_accuracy, epoch_test_loss, epoch_test_accuracy)
                fp.write(out_str)
                
            training_losses.append(training_loss)
            training_accuracies.append(training_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

    names = ["SGD, lr = " + str(lr) for lr in lrs]
    plot(training_losses, (0,2), x_ticks, names, plot_path + "/" + id + "_training_losses" + ".png", "Training Loss with SGD", "epochs", "cross entropy loss")
    plot(test_losses, (0,2), x_ticks, names, plot_path + "/" + id + "_test_losses" + ".png", "Test Loss with SGD", "epochs", "cross entropy loss")
    plot(training_accuracies, (0,100),  x_ticks, names, plot_path + "/" + id + "_training_accuracies" + ".png", "Training Accuracy with SGD", "epochs", "accuracy")
    plot(test_accuracies, (0,100), x_ticks, names, plot_path + "/" + id + "_test_accuracies" + ".png", "Test Accuracy with SGD", "epochs", "accuracy")

 


# fix random seed
#  seed = 1337
#  random.seed(seed)
#  np.random.seed(seed)
#  torch.manual_seed(seed)
#  torch.cuda.manual_seed(seed)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = False
#  
#  model = Policy_Network(54, 6, False)
#  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#  model.to(device)
#  
#  pretrain("./data/supervised_1000_2.csv",
#      "./saved_models/actor_pretrained.pt",
#      1337,
#      64,
#      30,
#      nn.CrossEntropyLoss(),
#      model,
#      optim.SGD(model.parameters(), lr=0.5),
#      device
#      )