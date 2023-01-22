from create_supervised_data import generate_supervised_data
from imitation_learning import test_params

import torch

# Uncomment the following lines to generate the data for supervised learning
"""
generate_supervised_data(0, 50, "data/train", "data/supervised_first_50.csv") 
generate_supervised_data(0, 100, "data/train", "data/supervised_first_100.csv") 
generate_supervised_data(0, 4000, "data/train", "data/supervised_first_4000.csv") 
generate_supervised_data(0, 24000, "data/train", "data/supervised_full.csv") 
generate_supervised_data(100000, 102400, "data/val", "data/supervised_val.csv")
"""

test_params("data/supervised_full.csv",
            "data/supervised_val.csv",
            "plots",
            "logs",
            "capacity",
            [0.3, 0.1, 0.05, 0.03],
            1337,
            20,
            [2*i for i in range(1, 11)],
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
