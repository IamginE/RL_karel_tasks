from config import get_configs
from execute_policy import print_Karel_policy
from networks import Policy_Network
import torch


def main():
    args = get_configs()
    actor = Policy_Network(args.vec_size, 6, True)
    
    if args.pretrained:
        checkpoint_actor = torch.load("./saved_models/actor_pretrained_full.pt")
    else:
        checkpoint_actor = torch.load("./saved_models/actor_first_100_11000.pt")
    actor.load_state_dict(checkpoint_actor['model_state_dict'])
    actor.set_softmax(True)

    print_Karel_policy(actor, args.path, 30)
main()