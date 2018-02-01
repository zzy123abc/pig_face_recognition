import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from net.lenet import *

net=torchvision.models.densenet161(num_classes=30)
net=net.cuda()

cudnn.benchmark = True

checkpoint_file = 'best.pth.tar'
checkpoint = torch.load(checkpoint_file)
net.load_state_dict(checkpoint['state_dict'])

test_transform=transforms.Compose(
    [
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize((0.485,0.456,0.406),
                          (0.229,0.224,0.225))]
)

testset=torchvision.datasets.ImageFolder(root='/home/zhangzhaoyu/pig_test',transform=test_transform)
testloader=torch.utils.data.DataLoader(
    testset,
    batch_size=16,
    shuffle=False,
    num_workers=24,
    pin_memory=True
)

imgs_test = []
imgs = testloader.dataset.imgs
for i in range(len(imgs)):
    img_path = imgs[i][0]
    img=img_path.split('/')[-1]
    img=img[:-4]
    imgs_test.append(img)

f=open('imgs_list.txt','w')
for line in imgs_test:
    f.write(line)
    f.write('\n')
f.close()

net=net.eval()

for i,data in enumerate(testloader,0):

    inputs,labels=data
    inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())

    m=nn.Softmax()
    outputs=net(inputs)
    outputs=m(outputs)
    outputs=outputs.data.cpu().numpy()
    if i ==0:
        out=outputs
    if i > 0:
        out = np.concatenate((out, outputs), axis=0)

np.save('out.npy',out)

print 'test ok'
