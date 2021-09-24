import torch
from models import Nima
from losses import earth_movers_distance
from datasets import ImageDataset
from torch.utils.data import DataLoader
from utils import AverageMeter, ensure_dir_exists
import time
import os
import argparse
from torchvision import transforms


# Base models on Imagenet benchmark: https://rwightman.github.io/pytorch-image-models/results/
def train(base_model_name, batch_size, fine_tune, dropout, lr, n_epochs, checkpoint_dir, log_interval,
          checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 10
    train_dataset = ImageDataset("data/images", "data/test/train.json", n_classes, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    val_dataset = ImageDataset("data/images", "data/test/test.json", n_classes, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Nima(base_model_name, dropout, n_classes)
    model.set_parameter_requires_grad(fine_tune)
    model = model.to(device)

    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr)

    loss_fn = earth_movers_distance
    best_loss = 1e9
    epochs_since_improvement = 0
    start_epoch = 0
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint["loss"]
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for epoch in range(start_epoch, n_epochs):
        train_epoch(train_loader, model, optimizer, loss_fn, device, log_interval, epoch)
        loss = validate(val_loader, model, loss_fn, device, log_interval)
        is_best = loss < best_loss
        best_loss = min(best_loss, loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        save_checkpoint(checkpoint_dir, "ava_{}_{}".format(base_model_name, epoch), epoch, model, optimizer,
                        epochs_since_improvement, loss,
                        is_best)


def save_checkpoint(cp_dir, data_name, epoch, model, optimizer, epochs_since_improvement, loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'loss': loss}
    filename = 'checkpoint_' + data_name + '.pt'
    torch.save(state, os.path.join(cp_dir, filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join(cp_dir, 'BEST_' + filename))


def train_epoch(train_loader, model, optimizer, loss_fn, device, log_interval, epoch):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    start = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        data_time.update(time.time() - start)

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(labels, outputs)
        losses.update(loss.item(), inputs.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)

        start = time.time()
        if i % log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))


def validate(val_loader, model, loss_fn, device, log_interval):
    with torch.no_grad():
        model.eval()
        losses = AverageMeter()
        batch_time = AverageMeter()
        start = time.time()
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = loss_fn(labels, outputs)
            losses.update(loss.item(), inputs.size(0))
            batch_time.update(time.time() - start)
            start = time.time()
            if i % log_interval == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))
    return losses.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--epoch', help='Num epoch', required=True, type=int
    )
    parser.add_argument(
        '-lr', '--lr', help='Learning rate', required=True, type=float
    )
    parser.add_argument(
        '-cd', '--cp_dir', help='Checkpoint directory', required=False, default="cp", type=str
    )
    parser.add_argument(
        '-bs', '--batch_size', help='Batch size', required=False, default=16, type=int
    )
    parser.add_argument(
        '-m', '--model', help='Base model name', required=False, default="resnet50", type=str
    )
    parser.add_argument(
        '-d', '--dropout', help='Dropout rate', required=False, default=0.75, type=float
    )
    parser.add_argument(
        '-f', '--fine_tune', help='Fine tune', required=False, default=False, type=bool
    )
    parser.add_argument(
        '-l', '--log_interval', help=' Log frequency', required=False, default=100, type=int
    )
    parser.add_argument(
        '-cp', '--checkpoint', help='Checkpoint path', required=False, type=str
    )
    args = parser.parse_args()

    n_epoches = args.__dict__['epoch']
    cp_dir = args.__dict__['cp_dir']
    batch_size = args.__dict__['batch_size']
    base_model_name = args.__dict__['model']
    dropout = args.__dict__['dropout']
    fine_tune = args.__dict__['fine_tune']
    log_interval = args.__dict__['log_interval']
    checkpoint = args.__dict__['checkpoint']
    lr = args.__dict__['lr']

    train(base_model_name=base_model_name, batch_size=batch_size, fine_tune=fine_tune, dropout=dropout, lr=lr,
          n_epochs=n_epoches, checkpoint_dir=cp_dir, log_interval=log_interval,
          checkpoint=checkpoint)
