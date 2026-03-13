import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torchvision.io import read_image
import torchvision.transforms as transforms
from dataset.h36m import TestSet as h36m
from dataset.taichi import TestSet as taichi
from dataset.cub import TestSet as cub
from dataset.cub_three import TestSet as cub_three
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import networkx as nx
import time

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--checkpoint_path', default="./checkpoint_latest.pth")
    args = parser.parse_args()
    return args

def test_epoch_end(batch_list):
    X = torch.cat([batch['det_keypoints'] for batch in batch_list]) * 0.5 + 0.5
    y = torch.cat([batch['keypoints'] for batch in batch_list])
    n_det_kp = X.shape[1]
    n_gt_kp = y.shape[1]
    batch_size = X.shape[0]
    X = X.reshape(batch_size, n_det_kp, 2)  # (b, n, 2)
    y = y.reshape(batch_size, n_gt_kp, 2)  # (b, m, 2)
    cost = torch.sum((X.unsqueeze(2) - y.unsqueeze(1)) ** 2, dim=-1).mean(dim=0).detach().cpu().numpy()  # (n, m)

    G = nx.DiGraph()
    source = 'S'
    sink = 'T'
    for i in range(n_det_kp):
        G.add_node(f'X_{i}')
    for j in range(n_gt_kp):
        G.add_node(f'Y_{j}', demand=0)
    G.add_node(source, demand=-n_gt_kp)
    G.add_node(sink, demand=n_gt_kp)
    for i in range(n_det_kp):
        G.add_edge(source, f'X_{i}', capacity=n_gt_kp // n_det_kp, weight=0)
    for i in range(n_det_kp):
        for j in range(n_gt_kp):
            weight = cost[i, j]
            G.add_edge(f'X_{i}', f'Y_{j}', capacity=1, weight=weight)
    for j in range(n_gt_kp):
        G.add_edge(f'Y_{j}', sink, capacity=1, weight=0)
    flowCost, flowDict = nx.network_simplex(G)

    c = 0
    n = 0
    for i in range(n_det_kp):
        for j in range(n_gt_kp):
            if flowDict[f'X_{i}'].get(f'Y_{j}', 0) > 0:
                print(i, j, cost[i, j])
                c = c + cost[i, j]
                n = n + 1
    print(n)
    return {'val_loss': c / n}

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if args.dataset == 'h36m':
        dataset = h36m(transform=transform)
    elif args.dataset == 'taichi':
        dataset = taichi(transform=transform)
    elif args.dataset == 'cub':
        dataset = cub(transform=transform)
    elif args.dataset == 'cub_three':
        dataset = cub_three(transform=transform)
    else:
        raise NotImplementedError
    test_dataloader = DataLoader(dataset=dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)
    test_output_list = []
    model = torch.load(args.checkpoint_path, weights_only=False).cuda().eval()
    template_points_ = model.memory.get_template().unsqueeze(0)
    for batch in test_dataloader:
        frame = batch['img'].to(device)
        keypoints = batch['keypoints'].to(device)
        template_points = template_points_.repeat(frame.size(0), 1, 1)
        with torch.no_grad():
            estimated_params = model.regressor(frame, template_points)
            aug = model.aug.unsqueeze(0).repeat(frame.size(0), 1, 1)
            transformed_template_points = torch.matmul(estimated_params,torch.cat([template_points, aug], dim=-1).unsqueeze(-1)).squeeze(-1)
            test_output_list.append({'keypoints': keypoints.cpu(), 'det_keypoints': transformed_template_points.cpu()})
    print(test_epoch_end(test_output_list)['val_loss'])

def main():
    args = parse_args()
    run(args)

if __name__ == '__main__':
    main()