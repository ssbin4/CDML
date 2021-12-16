import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from cdml import evaluate
from cdml import dataset
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test hdml with triplet loss.')
    parser.add_argument('-b', '--batch_size', type=int, default=30, help="Batch size.")
    parser.add_argument('-s', '--image_size', type=int, default=227, help="The size of input images.")
    parser.add_argument('-m', '--max_steps', type=int, default=1000, help="The maximum step number.")
    parser.add_argument('-c', '--n_class', type=int, default=99, help="Number of classes.")
    parser.add_argument('-lo', '--loss', type=str, default='triplet', choices=['triplet', 'npair'],
                        help='Choose loss function.')
    parser.add_argument('-d', '--dataset', type=str, default='cars196', choices=['cars196', 'cub200_2011'], help='Choose a dataset.')
    parser.add_argument('-n', '--no_cdml', action='store_true', default=False, help='No use cdml.')
    parser.add_argument('-p', '--pretrained', action='store_true', default=False, help='Use pretrained weight.')
    parser.add_argument('-nc', '--n_cand', type=int, default=4, help="Candidate number.")
    parser.add_argument('-a', '--alpha', type=float, default=0.1, help="Threshold value for candidates.")
    parser.add_argument('-e', '--epoch', type=int, default=30000, help="Epoch to be tested.")
    args = parser.parse_args()
    if args.dataset == 'cub200_2011':
        n_class = 101
        if args.loss == 'triplet':
            streams = dataset.get_streams('data/CARS196/cars196.hdf5', args.batch_size, 'triplet',
                                          crop_size=args.image_size)
        elif args.loss == 'npair':
            streams = dataset.get_streams('data/CARS196/cars196.hdf5', args.batch_size, 'n_pairs',
                                          crop_size=args.image_size)
    elif args.dataset == 'cars196':
        n_class = 99
        if args.loss == 'triplet':
            streams = dataset.get_streams('data/cub200_2011/cub200_2011.hdf5', args.batch_size, 'triplet',
                                          crop_size=args.image_size)
        elif args.loss == 'npair':
            streams = dataset.get_streams('data/cub200_2011/cub200_2011.hdf5', args.batch_size, 'n_pairs',
                                          crop_size=args.image_size)
    else:
        raise ValueError("`dataset` must be 'cars196' or 'cub200_2011'.")
    writer = SummaryWriter()
    model_path = os.path.join('model', args.loss)
    model_path = os.path.join(model_path, args.dataset)
    if args.no_cdml:
        evaluate.evaluate_triplet(streams, writer, args.max_steps, args.n_class,
                                  model_path=model_path,
                                  pretrained=args.pretrained,
                                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                  epoch=args.epoch)
    else:
        evaluate.evaluate_cdml_triplet(streams, writer, args.max_steps, n_class,
                                       model_path=model_path,
                                       pretrained=args.pretrained,
                                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                       epoch=args.epoch, n_cand=args.n_cand, alpha=args.alpha)
    writer.close()