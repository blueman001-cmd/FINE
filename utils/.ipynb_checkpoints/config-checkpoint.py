import argparse


def build_args():
    
    parser = argparse.ArgumentParser(description="MAGIC")
    
    # parser.add_argument("--dataset", type=str, default="streamspot")
    
    # parser.add_argument("--dataset", type=str, default="wget")

    parser.add_argument("--dataset", type=str, default="trace")

    # parser.add_argument("--dataset", type=str, default="theia")

    # parser.add_argument("--dataset", type=str, default="cadets")
    
    parser.add_argument("--max_epoch", type=str, default=8)
    parser.add_argument("--num_hidden", type=str, default=64)
    parser.add_argument("--num_layer", type=str, default=3)
    parser.add_argument("--n", type=str, default=0)
    parser.add_argument("--p", type=str, default=1)

    
    # parser.add_argument("--pooling", type=str, default="sum")
    
    parser.add_argument("--pooling", type=str, default="mean")
    
    args = parser.parse_args()
    
    return args
