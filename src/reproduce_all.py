from create_supervised_data import generate_supervised_data
from imitation_learning import test_params, pretrain
from networks import Policy_Network
from data_loading import Dataset_Supervision
from trainer import FF_Trainer
from evaluation import eval_policy

import torch
import numpy as np
import random
from torch import optim
from torch import nn

# random seed used in all experiments
SEED =  1337

# Uncomment the following lines to generate the data for supervised learning
"""
generate_supervised_data(0, 50, "data/train", "data/supervised_first_50.csv") 
generate_supervised_data(0, 100, "data/train", "data/supervised_first_100.csv") 
generate_supervised_data(0, 4000, "data/train", "data/supervised_first_4000.csv") 
generate_supervised_data(0, 24000, "data/train", "data/supervised_full.csv") 
generate_supervised_data(100000, 102400, "data/val", "data/supervised_val.csv")
"""

# Test supervised learning on the full training data set to verify model capacity and to get a baseline (I.2.c))
# Uncomment the following lines to run the tests
"""
test_params("data/supervised_full.csv",
            "data/supervised_val.csv",
            "plots",
            "logs",
            "capacity",
            [0.3, 0.1, 0.05, 0.03],
            SEED,
            20,
            [2*i for i in range(1, 11)],
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
"""

# Produce and save baseline model
"""
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = Policy_Network(54, 6, False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pretrain("./data/supervised_full.csv",
    "./saved_models/actor_pretrained_full.pt",
    64,
    25,
    nn.CrossEntropyLoss(),
    model,
    optim.SGD(model.parameters(), lr=0.1),
    device,
    100
    )
"""

# evaluate the baseline model
# only prints results to the console
"""
train_batch_size = 64
test_batch_size = 64
train_kwargs = {'batch_size': train_batch_size}
test_kwargs = {'batch_size': test_batch_size}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
  
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
  
training_data = Dataset_Supervision(csv_file="data/supervised_full.csv", vec_size=4*4*3+6)
test_data = Dataset_Supervision(csv_file="data/supervised_val.csv", vec_size=4*4*3+6)
train_loader = torch.utils.data.DataLoader(training_data, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

actor = Policy_Network(54, 6, False)
checkpoint_actor = torch.load("./saved_models/actor_pretrained_full.pt")
actor.load_state_dict(checkpoint_actor['model_state_dict'])
actor.set_softmax(False)

trainer_training_data = FF_Trainer(actor, None, train_loader, None, nn.CrossEntropyLoss(), device)
trainer_test_data = FF_Trainer(actor, None, test_loader, None, nn.CrossEntropyLoss(), device)

trainer_training_data.evaluate()
trainer_test_data.evaluate()

gamma = 0.99

actor.set_softmax(True)
eval_policy(actor, "./data/train", 0, 24000, 30, 4, 4, gamma, device)
eval_policy(actor, "./data/val", 100000, 102400, 30, 4, 4, gamma, device)
"""