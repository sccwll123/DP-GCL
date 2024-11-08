import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='movielens-100k')
    parser.add_argument("--implcit_bottom", type=int, default=3)
    parser.add_argument("--loadFilename", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--ssl_temp", type=float, default=0.2)
    parser.add_argument("--choosing_tmp", type=float, default=0.2)
    parser.add_argument("--rec_loss_reg", type=float, default=1)
    parser.add_argument("--ssl_loss_reg", type=float, default=0.02)
    parser.add_argument("--walk_length", type=int, default=10)
    parser.add_argument("--sparse_reg", type=int, default=0.02)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr_decay", type=float, default=1.)
    parser.add_argument("--lr_decay_every_step", type=int, default=5)
    parser.add_argument("--K_list", type=int, nargs='+', default=[10, 20])
    parser.add_argument("--K_neg_ratio", type=float, default=0.125)
    parser.add_argument("--K_pos_ratio", type=float, default=0.0625)
    parser.add_argument("--tao_sample", type=float, default=0.7)

    opt = parser.parse_args()
    return opt
