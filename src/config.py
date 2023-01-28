import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', default="", help='Path to the .json file', type=str)
    parser.add_argument('--pretrained', help='Boolean that decides if the PPO-trained model or the model trained only via supervised learning on all training tasks should be used.', default=False, type=str2bool)
    parser.add_argument('--vec_size', help='Size of a vectorized state.', default=54, type=int)
    
    args = parser.parse_args() 
    return args

