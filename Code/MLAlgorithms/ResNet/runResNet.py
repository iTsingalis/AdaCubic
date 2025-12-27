import os.path as path
import os, sys


import os

import random
import argparse
from tqdm import tqdm
from pathlib import Path
from Utils.timer import Timer
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as pydsets
from torchvision import datasets, transforms

from Code.Optimizers.AdaCubic import AdaCubic
from Code.Optimizers.AdaHessian import AdaHessian
from Code.MLAlgorithms.ResNet.ResNet import ResNet

epoch_timer = Timer()
total_timer = Timer()

NUM_FEATURES = {
    'cifar10': 32 * 32,
    'cifar100': 32 * 32,
    'mnist': 28 * 28
}


def check_folder(name, exp_folder):
    run = 1
    run_folder = os.path.join(exp_folder, '{}{}'.format(name, run))
    if not os.path.exists(run_folder):
        Path(run_folder).mkdir(parents=True, exist_ok=True)

        print("Path {} created".format(run_folder))
        return run_folder

    while os.path.exists(run_folder):
        run += 1
        run_folder = os.path.join(exp_folder, '{}{}'.format(name, run))
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    print("Path {} created".format(run_folder))
    return run_folder


def acc(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for samples, labels in data_loader:
            samples = samples.to(device)

            tst_outputs = model(samples)

            correct += (torch.argmax(tst_outputs.data, 1).cpu() == labels).float().sum()

            total += labels.size(0)

        accuracy = 100 * float(correct) / float(total)
        return accuracy


def initialize(seed=None):
    if seed:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def main():
    exp_folder = os.path.join(root, 'Code', 'MLAlgorithms', 'ResNet', 'Results', args.optimizer)
    Path(exp_folder).mkdir(parents=True, exist_ok=True)

    exp_folder_num = check_folder('exp', exp_folder)

    for r in range(args.n_runs):
        num_classes = 10
        model = ResNet(num_classes=num_classes, depth=args.depth)

        with open(os.path.join(exp_folder_num, 'model_params.txt'), 'w') as file_object:
            file_object.write('{}'.format({'num_classes': num_classes, 'depth': args.depth}))

        #######################
        #  USE GPU FOR MODEL  #
        #######################

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)

        # model = torch.nn.DataParallel(model)
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

        criterion = torch.nn.CrossEntropyLoss()

        # Optimizers
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        elif args.optimizer == 'AdaHessian':
            optimizer = AdaHessian(model.parameters(), lr=0.15, average_conv_kernel=False)
        elif args.optimizer == 'AdaCubic':
            eta1 = 0.05
            eta2 = 0.75
            alpha1 = 2.5  # ** (1/ 3)  # very successful
            alpha2 = 0.25  # ** (1 /3)  # unsuccessful

            optimizer = AdaCubic(model.parameters(), eta1=eta1, eta2=eta2, alpha1=alpha1, alpha2=alpha2,
                                 xi0=0.05, tol=1e-4, n_samples=1, average_conv_kernel=False, solver='exact',
                                 kappa_easy=0.01, gamma1=0.25)
        else:
            raise ValueError('Select an existing optimizer.')

        # learning rate schedule
        if not type(optimizer).__name__ == "AdaCubic":
            import torch.optim.lr_scheduler as lr_scheduler
            lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1, last_epoch=-1)

        print(optimizer)

        n_iter = 0
        best_acc = 0.0

        with open(os.path.join(exp_folder_num, 'opt_params.txt'), 'w') as file_object:
            file_object.write('{}'.format(optimizer.defaults))

        with open(os.path.join(exp_folder_num, 'seed_{}.txt'.format(args.seed)), 'w') as file_object:
            file_object.write('{}'.format(args.seed))

        tr_losses, tst_losses = [], []
        tr_accuracies, tst_accuracies = [], []

        run_folder = os.path.join(exp_folder_num, 'run{}'.format(r))
        Path(run_folder).mkdir(parents=True, exist_ok=True)

        total_timer.start()
        for n_epoch in range(1, int(args.n_epochs + 1)):
            epoch_timer.start()
            running_loss, running_samples = 0, 0
            model.train()

            with (tqdm(total=len(train_loader.dataset), disable=True) as progressbar):
                for i, (samples, labels) in enumerate(train_loader):
                    samples = samples.to(device)
                    labels = labels.to(device)

                    def closure(backward=True):
                        if backward:
                            optimizer.zero_grad()
                        model_outputs = model(samples)
                        cri_loss = criterion(model_outputs, labels)

                        create_graph = type(optimizer).__name__ == "AdaCubic" or type(
                            optimizer).__name__ == "AdaHessian"
                        if backward:
                            cri_loss.backward(create_graph=create_graph)
                        return cri_loss

                    tr_loss = optimizer.step(closure=closure)

                    batch_loss = tr_loss.detach().cpu()
                    running_loss += batch_loss * train_loader.batch_size
                    running_samples += train_loader.batch_size

                    n_iter += 1

                    progressbar.update(labels.size(0))

                    if type(optimizer).__name__ == "AdaCubic":
                        with open(os.path.join(run_folder, 'xi_iter.txt'), 'a+') as file_object:
                            file_object.write('{}\n'.format(optimizer.state['xi'].detach().cpu().item()))

            elapsed_epoch_time = epoch_timer.stop(verbose=False)
            with open(os.path.join(run_folder, 'epoch_time.txt'), 'a+') as file_object:
                file_object.write('{}\n'.format(elapsed_epoch_time))

            if not type(optimizer).__name__ == "AdaCubic":
                print('Update lr_scheduler lr: {}'.format(lr_scheduler.get_last_lr()))
                lr_scheduler.step()

            tst_acc = acc(model, test_loader, device)
            tr_acc = acc(model, train_loader, device)

            if type(optimizer).__name__ == "AdaCubic":
                with open(os.path.join(run_folder, 'xi_epoch.txt'), 'a+') as file_object:
                    file_object.write('{}\n'.format(optimizer.state['xi'].detach().cpu().item()))

            with open(os.path.join(run_folder, 'tr_acc.txt'), 'a+') as file_object:
                file_object.write('{}\n'.format(tr_acc))

            with open(os.path.join(run_folder, 'tst_acc.txt'), 'a+') as file_object:
                file_object.write('{}\n'.format(tst_acc))

            with open(os.path.join(run_folder, 'loss.txt'), 'a+') as file_object:
                file_object.write('{}\n'.format(running_loss / running_samples))

            with open(os.path.join(run_folder, 'model.txt'), 'a+') as file_object:
                print(model, file=file_object)

            print("n_iteration: {} - n_epoch {} - Tr. Loss: {:.5} - Tr. Accuracy: {:.4} - Tst. Accuracy: {:.4}".
                  format(n_iter, n_epoch, running_loss / running_samples, tr_acc, tst_acc))

            if tst_acc > best_acc:
                best_acc = tst_acc

            tr_losses.append(running_loss / len(train_loader.sampler))
            tr_accuracies.append(tr_acc)
            tst_accuracies.append(tst_acc)

        elapsed_time = total_timer.stop(tag='Run execution time', verbose=True)
        with open(os.path.join(run_folder, 'time.txt'), 'a+') as file_object:
            file_object.write('{}'.format(elapsed_time))

        print('Best Tst. Acc {}'.format(best_acc))
        # Save best tst acc
        with open(os.path.join(run_folder, 'best_acc.txt'), 'w') as file_object:
            file_object.write('{}'.format(best_acc))

        # Turn interactive plotting off
        plt.ioff()

        fig = plt.figure()
        plt.title(type(optimizer).__name__)
        # tr_losses = preprocess(tr_losses)
        plt.plot(tr_losses, 'o-', label='Train Loss')
        plt.legend()
        plt.yscale('log')
        plt.grid()
        plt.savefig(os.path.join(run_folder, 'loss.png'))
        plt.close(fig)

        fig = plt.figure()
        plt.title(type(optimizer).__name__)
        plt.plot(tr_accuracies, 'o-', label='Train acc.')
        plt.plot(tst_accuracies, 'o-', label='Test acc.')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(run_folder, 'acc.png'))
        plt.close(fig)

        # plt.show(block=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--optimizer', type=str, default='AdaCubic', help='choose optim')
    parser.add_argument('--batch_size', type=int, default=256, metavar='B',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--seed', type=int, default=43, metavar='S',
                        help='choose seed')
    parser.add_argument('--n_epochs', type=int, default=500, metavar='E',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--n_runs', type=int, default=1, metavar='R',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--depth', type=int, default=18, metavar='R',
                        help='depth of resnet (default: 18)')
    parser.add_argument('--task', type=str, default='cifar10', metavar='T',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()

    print(args)

    initialize(seed=args.seed)

    # # Place here the root of the project
    root = '/media/blue/tsingalis/AdaCubic/'

    if 'cifar10' == args.task:

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = pydsets.CIFAR10(root=os.path.join(root, 'Datasets'), train=True, download=True,
                                        transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=1)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        test_dataset = pydsets.CIFAR10(root=os.path.join(root, 'Datasets'), train=False, download=True,
                                       transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=1)
    else:
        raise Exception('Select an available dataset')

    num_features = NUM_FEATURES[args.task]

    main()
