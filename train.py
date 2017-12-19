import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import shutil
from net.lenet import *

train_transform=transforms.Compose(
    [
     transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
     transforms.ToTensor(),
     transforms.Normalize((0.485,0.456,0.406),
                          (0.229,0.224,0.225))]
)

val_transform=transforms.Compose(
    [
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize((0.485,0.456,0.406),
                          (0.229,0.224,0.225))]
)

trainset=torchvision.datasets.ImageFolder(root='/home/zhangzhaoyu/pig_train',transform=train_transform)
trainloader=torch.utils.data.DataLoader(
    trainset,
    batch_size=16,
    shuffle=True,
    num_workers=12,
    pin_memory=True
)

valset=torchvision.datasets.ImageFolder(root='/home/zhangzhaoyu/pig_val',transform=val_transform)
valloader=torch.utils.data.DataLoader(
    valset,
    batch_size=16,
    shuffle=False,
    num_workers=12,
    pin_memory=True
)

net=torchvision.models.resnet50(num_classes=30)
net=net.cuda()

cudnn.benchmark=True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

criterion=nn.CrossEntropyLoss().cuda()
lr=0.001
optimizer=optim.SGD(net.parameters(),lr=lr,momentum=0.9)

def train(trainloader,net,criterion,optimizer,epoch):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    net=net.train()

    for i,data in enumerate(trainloader,0):
        inputs,labels=data
        inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())

        outputs=net(inputs)
        loss=criterion(outputs,labels)

        prec1, prec3 = accuracy(outputs.data, labels.data, topk=(1, 3))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top3.update(prec3[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   epoch, i, len(trainloader), loss=losses, top1=top1, top3=top3))

    print 'Finished Training'

def val(valloader,net,criterion):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    net=net.eval()

    for i,data in enumerate(valloader,0):
        inputs,labels=data
        inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())

        outputs=net(inputs)
        loss=criterion(outputs,labels)

        prec1, prec3 = accuracy(outputs.data, labels.data, topk=(1, 3))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top3.update(prec3[0], inputs.size(0))

        if i % 10 == 0:
            print('Val: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                i, len(valloader), loss=losses,
                top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'.format(top1=top1, top3=top3))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best.pth.tar')

def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

start_epoch=0
epochs=30

resume='checkpoint.pth.tar'
if os.path.isfile(resume):
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(resume, checkpoint['epoch']))

best_prec1=0

for epoch in range(start_epoch,epochs):

    adjust_learning_rate(lr, optimizer, epoch)

    train(trainloader,net,criterion,optimizer,epoch)
    prec1=val(valloader,net,criterion)

    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best)