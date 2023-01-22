test_params(
    "./data/supervised_full_2.csv",
    "./data/val_full_2.csv",
    "./plots",
    "./logs",
    "cap_check",
    [0.3, 0.1, 0.05, 0.03],
    1337,
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)   