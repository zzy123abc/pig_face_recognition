import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets,models,transforms
import os
import shutil

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

trainset=torchvision.datasets.ImageFolder(root='/home/zhangzhaoyu/train',
                                          transform=train_transform)
trainloader=torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True,
    num_workers=24,
    pin_memory=True
)

valset=torchvision.datasets.ImageFolder(root='/home/zhangzhaoyu/val',
                                        transform=val_transform)
valloader=torch.utils.data.DataLoader(
    valset,
    batch_size=4,
    shuffle=False,
    num_workers=24,
    pin_memory=True
)

model_conv=torchvision.models.densenet161(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad=False

num_ftrs=model_conv.classifier.in_features
model_conv.classifier=nn.Linear(num_ftrs,30)

model_conv=model_conv.cuda()
criterion=nn.CrossEntropyLoss().cuda()

cudnn.benchmark=True

learning_rate=0.001
optimizer_conv=optim.SGD(model_conv.classifier.parameters(),lr=learning_rate,momentum=0.9)

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

def train(trainloader,net,criterion,optimizer,epoch):

    net = nn.DataParallel(net, device_ids=[0, 1])

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
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\t'\
                  'lr {lr}'.format(
                   epoch, i, len(trainloader), loss=losses,
                   top1=top1, top3=top3,lr=optimizer.param_groups[0]['lr']))

    print 'Finished Training'

def val(valloader,net,criterion):

    net = nn.DataParallel(net,device_ids=[0,1])

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

start_epoch=0
epochs=30

resume='checkpoint.pth.tar'
if os.path.isfile(resume):
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model_conv.load_state_dict(checkpoint['state_dict'])
    optimizer_conv.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(resume, checkpoint['epoch']))

best_prec1=0

for epoch in range(start_epoch,epochs):

    adjust_learning_rate(optimizer_conv, epoch)

    train(trainloader,model_conv,criterion,optimizer_conv,epoch)
    prec1=val(valloader,model_conv,criterion)

    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model_conv.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer_conv.state_dict(),
    }, is_best)
