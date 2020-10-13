from argparse import ArgumentParser
import yaml
from solver import Solver

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config.yaml')
    parser.add_argument('--data_dir', '-d', type=str, default='vctk_data')
    parser.add_argument('--train_set', type=str, default='train')
    parser.add_argument('--train_index_file', type=str, default='train_samples_64.json')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default='vctk_model')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--summary_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=5000)
    parser.add_argument('--tag', '-t', type=str, default='init')
    parser.add_argument('--iters', type=int, default=500000)

    args = parser.parse_args()

    # load config file
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    solver = Solver(config=config, args=args)

    if args.iters > 0:
        solver.train(n_iterations=args.iters)
