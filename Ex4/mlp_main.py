from mlp_mnist import KerasMnist

from matplotlib import pyplot as plt
import argparse


def train_eval_single(hidden_layers, skips, epochs, batch_size, output_curve=None):
    km = KerasMnist(hidden_layers, skips, epochs, batch_size)
    km.load_data()
    km.build_model()
    history, score_train, score_test = km.train_eval_model()

    if output_curve:
        km.plot_curves(history, output_curve)

    return score_train, score_test


def train_eval_series(hidden_layers, skips, epochs, batch_size, output_curve):
    train_loss_values, test_loss_values = [], []
    train_acc_values, test_acc_values = [], []
    confs = []
    for conf in hidden_layers:
        conf_str = str(conf)
        print('training network: ', conf_str)
        score_train, score_test = train_eval_single(conf, skips, epochs, batch_size)
        print('train loss: {}, train accuracy: {}'.format(score_train[0], score_train[1]))
        print('test loss: {}, test accuracy: {}\n'.format(score_test[0], score_test[1]))
        train_loss_values.append(score_train[0])
        test_loss_values.append(score_test[0])
        train_acc_values.append(score_train[1])
        test_acc_values.append(score_test[1])
        confs.append(conf_str)

    plt.clf()
    x = range(len(hidden_layers))
    plt.xticks(x, confs)
    plt.plot(x, train_loss_values, 'bo')
    plt.plot(x, test_loss_values, 'b+')
    plt.xlabel('hidden layers')
    plt.ylabel('loss')
    plt.savefig(output_curve + '.png')


def get_args():
    parser = argparse.ArgumentParser(description='Trains and tests a neural classification model.')
    parser.add_argument('-m', '--mode', type=str, choices=['single', 'series'],
                        help="""single will train a single model and plot the learning curves,
                         series will train multiple models and plot the final loss values.""")
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-hl', '--hidden_layers', nargs='+', type=int, default=(32, 16),
                        help="""A list of integers that defines the network architecture. 
                                Every integer x in the list adds another layer of size x to the network.
                                For series mode, use -1 as a separator between model configurations.
                                For example, '16 16 -1 32 16' is a series of two models of two hidden layers each.""")
    parser.add_argument('-s', '--skips', type=int, default=1)
    parser.add_argument('-o', '--output_name', type=str, default="figure",
                        help="Base name for the output files. They will be created in the directory from which the code was executed.")

    return parser.parse_args()


def check_args(args):
    if -1 in args.hidden_layers:
        if args.mode == 'single':
            raise argparse.ArgumentTypeError(
                "using -1 in the hidden layers argument is allowed only in 'series' mode.")

    hidden_layers = []
    i = 0
    curr_net = []
    while i < len(args.hidden_layers):
        if args.hidden_layers[i] == -1:
            hidden_layers.append(curr_net)
            curr_net = []
        else:
            curr_net.append(args.hidden_layers[i])
        i += 1
    hidden_layers.append(curr_net)

    print("parsed network configurations: \n", hidden_layers)
    if len([curr_net for curr_net in hidden_layers
            if len(curr_net) == 0]) > 0:
        raise argparse.ArgumentTypeError(
            "all network configurations should have at least one hidden layer.")

    return hidden_layers


def main():
    args = get_args()
    hidden_layers = check_args(args)

    if args.mode == 'single':
        score_train, score_test = train_eval_single(
            hidden_layers[0], args.skips, args.epochs,
            args.batch_size, args.output_name)
        print('train loss: {}, train accuracy: {}'.format(score_train[0], score_train[1]))
        print('test loss: {}, test accuracy: {}\n'.format(score_test[0], score_test[1]))
    else:
        train_eval_series(hidden_layers, args.skips, args.epochs,
                          args.batch_size, args.output_name)


if __name__ == "__main__":
    main()

