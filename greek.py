"""
Aparna Krishnan and Suparna Srinivasan
CS 5330 Computer Vision
Task 3 - Greek letter and Embedding

"""


from operator import delitem, le
import torch
import torchvision
import sys
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import cv2

#network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

#trunc network
class Submodel_greek(Net):
      
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward( self, x ):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return x
        
def train(epoch, network, train_loader, optimizer, train_losses, train_counter, log_interval):
  network.train()

  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    greek_output = network(data)
    loss = F.nll_loss(greek_output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'C:/Users/aparn/OneDrive/Desktop/prcv/greek_model.pth')
      torch.save(optimizer.state_dict(), 'C:/Users/aparn/OneDrive/Desktop/prcv/greek_optimizer.pth')

def test(network, test_loader, test_losses):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      greek_output = network(data)
      test_loss += F.nll_loss(greek_output, target, size_average=False).item()
      pred = greek_output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def main(argv):
    if(len(argv)<2):
          print("Usage: python greek.py <directory_name>")
          exit()
    path = Path(argv[1])
    imgPaths = list(path.glob( '**/*.jpg' ))
    imgPathStrs = [ ]
    imgNames = []
    for p in imgPaths:
          pathStr = str(p)
          imgPathStrs.append(pathStr)
          imgNames.append(pathStr.split('/')[-1])
    imgPathStrs.sort()
    imgNames.sort()
    images = []
    i = 0
    for n in imgPathStrs:
      img = cv2.imread(n, cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      img = cv2.resize(img, (28, 28))
      img = img[ :, :, np.newaxis]
      images.append(img)
      i+=1
    images = np.asarray(images)
    labels = []
    for name in imgNames:
        labels.append(name.split('_')[0])
    unique, mapping = np.unique(np.array( labels ), return_inverse=True)
    d = np.zeros((len(imgNames), 784))
    for imgNum in range(images.shape[0]):
      data[imgNum] = images[imgNum].flatten()
    np.savetxt("letterData.csv", d, delimiter=",", header="image data")
    np.savetxt( "letterCats.csv", mapping, delimiter=",", header="ground truth cat" )

    n_epochs = 150
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    data = torchvision.datasets.ImageFolder(argv[1], 
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize([28,28]),
                                torchvision.transforms.RandomInvert(p=1),
                                torchvision.transforms.Grayscale(num_output_channels=1),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                                ]))
   
    train_loader = torch.utils.data.DataLoader(data,
    batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data,
    batch_size=batch_size_train, shuffle=True)
    
    examples = enumerate(test_loader)
    print(examples)
    batch_idx,(example_data,example_targets) = next(examples)
    print(example_targets)
    example_data[0][0].shape

    fig = plt.figure(1)
    print()
    for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Ground Truth: {}".format(example_targets[i]))
      plt.xticks([])
      plt.yticks([])
    plt.show()
    fig

    greek_network = Net()
    optimizer = optim.SGD(greek_network.parameters(), lr=learning_rate,
                          momentum=momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    test(greek_network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
      train(epoch, greek_network, train_loader, optimizer, train_losses, train_counter, log_interval)
      test(greek_network, test_loader, test_losses)

    with torch.no_grad():
      greek_output = greek_network(example_data)
    

    fig = plt.figure()
    for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Prediction: {}".format(
        greek_output.data.max(1, keepdim=True)[1][i].item()))
      plt.xticks([])
      plt.yticks([])
    plt.show()
    fig
    
    #The following SSD code does not run, so commenting out.

    ''' ssd = 0

    ssd = torch.cat((np.square(greek_output[:,None] - greek_output[3]).sum(axis=2), example_targets) , 1)

    print(ssd[ssd[:, 0].sort()[1]])

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(greek_output, example_targets)
    
    print("score = ",neigh.score(greek_output, example_targets))
 '''
    return

if __name__ == "__main__":
    main(sys.argv)