"""
SouthPark Chatbot
"""

import os
import argparse
import torch

# import config
from models import MobileHairNet
from trainer import Trainer
from evaluate import evalTest, evaluate, evaluateOne
from dataset import HairDataset, ImgTransformer

from utils import CheckpointManager

DIR_PATH = os.path.dirname(__file__)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def build_model(checkpoint):
    model = MobileHairNet()

    if checkpoint:
        model.load_state_dict(checkpoint["model"])

    # Use appropriate device
    model = model.to(device)

    return model


def train(args, model, checkpoint, checkpoint_mng):
    trainer = Trainer(args, model, checkpoint_mng)

    if checkpoint:
        trainer.resume(checkpoint)

    trainfile = os.path.join(DIR_PATH, args.train_corpus)
    devfile = os.path.join(DIR_PATH, args.test_corpus)

    print("Reading training data from %s..." % trainfile)

    train_datasets = HairDataset(trainfile, args.imsize, color_aug=True)

    print(f"Read {len(train_datasets)} training images")

    print("Reading development data from %s..." % devfile)

    dev_datasets = HairDataset(devfile, args.imsize)

    print(f"Read {len(dev_datasets)} development images")

    # Ensure dropout layers are in train mode
    model.train()

    # trainer.train(train_datasets, n_epochs=args.ep, batch_size=args.bs, stage=args.mode, dev_data=dev_datasets)
    trainer.train(args, train_datasets, dev_data=dev_datasets)


def test(args, model, checkpoint):
    model.eval()

    testfile = os.path.join(DIR_PATH, args.test_corpus)
    print("Reading Testing data from %s..." % testfile)

    test_datasets = HairDataset(testfile, args.imsize)

    print(f"Read {len(test_datasets)} testing images")

    evalTest(test_datasets, model, args)


def run(args, model, checkpoint):
    dset = args.set
    num = args.num
    img_path = args.image
    model.eval()

    if not img_path:
        if dset == "train":
            path = args.train_corpus
        else:
            path = args.test_corpus

        testfile = os.path.join(DIR_PATH, path)
        print("Reading Testing data from %s..." % testfile)

        test_datasets = HairDataset(testfile, args.imsize)

        print(f"Read {len(test_datasets)} testing images")

        evaluate(test_datasets, model, num, absolute=False)
    else:
        transformer = ImgTransformer(args.imsize, color_aug=False)
        img = transformer.load(img_path)
        evaluateOne(img, model, absolute=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices={"train", "test", "run"}, help="mode to run the network")
    parser.add_argument("-cp", "--checkpoint")
    parser.add_argument("-st", "--set", choices={"train", "test"}, default="test")
    parser.add_argument("-im", "--image")
    parser.add_argument("-n", "--num", type=int, default=4)
    parser.add_argument("--train_corpus", type=str, default="data/dataset_celeba/train")
    parser.add_argument("--test_corpus", type=str, default="data/dataset_celeba/val")
    parser.add_argument("--ep", type=int, default=20, help="number of epochs")
    parser.add_argument("--save_freq", type=int, default=1, help="save checkpoint every x epochs")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="folder for models")
    parser.add_argument("--model_name", type=str, default="default", help="model name")
    parser.add_argument("--imsize", type=int, default=448, help="training image size")
    parser.add_argument("--workers", type=int, default=8, help="number of workers")
    parser.add_argument("--bs", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer (adam; sgd)")
    parser.add_argument("--wup", type=int, default=0, help="number of warm up epochs")
    parser.add_argument(
        "--lr_schedule", type=str, default="", help="learning rate schedule (multi_step_lr; plateau; cosine)"
    )
    parser.add_argument("--print_freq", type=int, default=100, help="print stats every x iterations")
    parser.add_argument("--grad_lambda", type=float, default=0.5, help="gradient loss lambda")
    args = parser.parse_args()

    print("args: ", args)

    SAVE_PATH = os.path.join(DIR_PATH, args.save_dir, args.model_name)
    print("Saving path:", SAVE_PATH)
    checkpoint_mng = CheckpointManager(SAVE_PATH)

    checkpoint = None
    if args.checkpoint:
        print("Load checkpoint:", args.checkpoint)
        checkpoint = checkpoint_mng.load(args.checkpoint, device)

    model = build_model(checkpoint)

    if args.mode == "train":
        train(args, model, checkpoint, checkpoint_mng)

    elif args.mode == "test":
        test(args, model, checkpoint)

    elif args.mode == "run":
        # run(model, checkpoint, dset=args.set, num=args.num, img_path=args.image)
        run(args, model, checkpoint)


# def init():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-cp', '--checkpoint')
#     args = parser.parse_args()

#     checkpoint_mng = CheckpointManager(SAVE_PATH)
#     checkpoint = None if not args.checkpoint else checkpoint_mng.load(args.checkpoint, device)

#     model = build_model(checkpoint)
#     # Set dropout layers to eval mode
#     model.eval()

#     return model


if __name__ == "__main__":
    main()
