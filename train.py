import os
import torch
from argparse import ArgumentParser
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from dataset.cub import TrainSet as cub
from dataset.cub_three import TrainSet as cub_three
from dataset.flowers import TrainSet as flowers
from dataset.h36m import TrainSet as h36m
from dataset.h36m_wobg import TrainSet as h36m_wobg
from dataset.hands import TrainSet as hands
from dataset.horse import TrainSet as horse
from dataset.taichi import TrainSet as taichi
from torch.utils.tensorboard import SummaryWriter
from models.pose_prior_learner import PosePriorLearner
from utils.utils import show_images
import time

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--num_parts', type=int, default=16)
    parser.add_argument('--thick', type=float, default=1e-3)
    parser.add_argument('--sklr', type=float, default=512)
    parser.add_argument('--dataset', default='h36m')
    parser.add_argument('--subject', default=None)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=20000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--boundary_loss_weight', type=float, default=1)
    parser.add_argument('--edge_reg_loss_weight', type=float, default=1)
    parser.add_argument('--vector_quantized', type=bool, default=True)
    parser.add_argument('--use_alpha', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--block', type=int, default=16)
    parser.add_argument('--missing', type=float, default=0.8)
    parser.add_argument('--log_dir', default='./logs')
    args = parser.parse_args()
    return args

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # define model
    model = PosePriorLearner(args).cuda()

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    # preprocessing of input images
    resize = transforms.Resize((args.img_size, args.img_size))
    to_tensor = transforms.ToTensor()
    flip = transforms.RandomHorizontalFlip(p=0.5)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # inverse
    inverse1 = transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.5, 1 / 0.5, 1 / 0.5])
    inverse2 = transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1])
    inverse = transforms.Compose([inverse2, inverse1])

    # define dataset and dataloader
    if args.dataset == 'h36m':
        transform = transforms.Compose([resize, to_tensor, flip, normalize])
        dataset = h36m(transform=transform)
    elif args.dataset == 'h36m_wobg':
        transform = transforms.Compose([resize, to_tensor, flip, normalize])
        dataset = h36m_wobg(transform=transform)
    elif args.dataset == 'taichi':
        transform = transforms.Compose([resize, to_tensor, flip, normalize])
        dataset = taichi(transform=transform)
    elif args.dataset == 'horse':
        transform = transforms.Compose([resize, to_tensor, flip, normalize])
        dataset = horse(transform=transform)
    elif args.dataset == 'flowers':
        transform = transforms.Compose([resize, to_tensor, flip, normalize])
        dataset = flowers(transform=transform)
    elif args.dataset == 'hands':
        transform = transforms.Compose([resize, to_tensor, flip, normalize])
        dataset = hands(transform=transform)
    elif args.dataset == 'cub':
        transform = transforms.Compose([resize, to_tensor, normalize])
        dataset = cub(transform=transform)
    elif args.dataset == 'cub_three':
        transform = transforms.Compose([resize, to_tensor, normalize])
        dataset = cub_three(transform=transform)
    else:
        raise NotImplementedError
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            drop_last=True)

    log_dir = os.path.join(args.log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print("log dir created")

    return_imgs = False
    # training loop
    i = 0
    for epoch in range(0, args.num_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        model.train()
        for batch in dataloader:
            i += 1
            optimizer.zero_grad()
            frame = batch['img'].to(device)
            if i % 500 == 0 and i != 0:
                print('Training steps {}:'.format(i))
                return_imgs = True
            loss, values = model(frame, return_imgs)
            loss.backward()
            optimizer.step()
            # tensorboard logs, show images every 500 iterations
            if not return_imgs:
                writer.add_scalars(main_tag='losses', tag_scalar_dict=values, global_step=i)
            else:
                values_ = {k: v for k, v in values.items() if len(v.shape) == 0}
                writer.add_scalars(main_tag='losses', tag_scalar_dict=values_, global_step=i)
                for k, v in values.items():
                    if len(v.shape) > 1:
                        if k == 'transformed_template' \
                                or k == 'template' \
                                or k == 'memory':
                            grid = show_images(v, renorm=None)
                        else:
                            grid = show_images(v, renorm=inverse)
                        writer.add_image(k, grid, global_step=i)
            if i >= args.max_steps:
                break
            return_imgs = False
        # save checkpoint
        torch.save(model, os.path.join(log_dir, 'checkpoint_latest.pth'))

def main():
    args = parse_args()
    run(args)

if __name__ == '__main__':
    main()