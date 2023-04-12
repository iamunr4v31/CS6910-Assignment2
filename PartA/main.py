from model import NeuralNetwork
from argparse import ArgumentParser
import pytorch_lightning as pl

def main(args):
    nn = NeuralNetwork(
        args.in_channels, args.out_channels, args.kernel_size, 
        args.stride, args.padding, batch_norm=args.batch_norm, activation=args.activation, 
        kernel_strategy=args.kernel_strategy, dropout=args.dropout, num_classes=args.num_classes, 
        hidden_size=args.hidden_size, dataset_path=args.dataset_path, num_workers=args.num_workers, 
        batch_size=args.batch_size, augmentation=args.augmentation
        )
    logger = pl.loggers.WandbLogger(project="CS6910-Assignment2")
    logger.watch(nn)
    callbacks = [pl.callbacks.ModelCheckpoint(monitor='val/acc', mode='max', save_top_k=1, save_last=True)]
    trainer = pl.Trainer(max_epochs=args.n_epochs, devices=-1, precision='16-mixed', logger=logger, callbacks=callbacks)
    trainer.fit(nn)
    if args.test:
        trainer.test(nn)

if __name__ == '__main__':
    parser = ArgumentParser(description='Neural Network Hyperparameters')

    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=64, help='Number of output channels')
    parser.add_argument('--kernel_size', type=int, default=3, help='Size of convolution kernel')
    parser.add_argument('--stride', type=int, default=1, help='Stride of convolution')
    parser.add_argument('--padding', type=int, default=1, help='Padding of convolution')
    parser.add_argument('--batch_norm', type=bool, default=True, help='Use batch normalization')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu', 'silu', 'mish'], help='Activation function')
    parser.add_argument('--kernel_strategy', type=str, default='same', choices=['same', 'double', 'half'], help='Kernel strategy')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in output')
    parser.add_argument('--hidden_size', type=int, default=64, help='Size of hidden layer')
    parser.add_argument('--dataset_path', type=str, default='../inaturalist_12K/', help='Path to dataset')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for dataloader')
    parser.add_argument('--augmentation', type=bool, default=True, help='Use data augmentation')
    parser.add_argument('--test', type=bool, default=False, help='Test mode')

    args = parser.parse_args()
    main(args)