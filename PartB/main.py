from model import FineTuneModel
from argparse import ArgumentParser
import pytorch_lightning as pl

def main(args):
    nn = FineTuneModel(
        learning_rate=args.learning_rate, num_classes=args.num_classes, augmentation=args.augmentation,
        dataset_path=args.dataset_path, num_workers=args.num_workers, batch_size=args.batch_size
    )
    logger = pl.loggers.WandbLogger(project="CS6910-Assignment2")
    logger.watch(nn)
    callbacks = [pl.callbacks.ModelCheckpoint(monitor='val/acc', mode='max', save_top_k=1, save_last=True)]
    trainer = pl.Trainer(max_epochs=args.n_epochs, devices=-1, precision='16-mixed', logger=logger, callbacks=callbacks)
    trainer.fit(nn)
    if args.test:
        trainer.test(nn)

if __name__ == "__main__":
    parser = ArgumentParser(description='Fine-tune Model Hyperparameters')

    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--augmentation', type=bool, default=True, help='Use data augmentation')
    parser.add_argument('--dataset_path', type=str, default='../inaturalist_12K/', help='Path to dataset')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for dataloader')
    parser.add_argument('--test', type=bool, default=False, help='Test mode')

    args = parser.parse_args()