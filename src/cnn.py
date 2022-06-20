# source code inspireed by
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#model-training-and-validation-code

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models

CATEGORIES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}



class MyCNNNetwork3(nn.Module):
    def __init__(self, num_in_channels, num_classes):
        super(MyCNNNetwork3, self).__init__()
        padding = int((2*(num_classes-1) - num_in_channels - 5) * 1/2)
        self.conv1_1 = nn.Conv2d(in_channels=num_in_channels, out_channels=64, kernel_size=5, stride=2, padding=padding)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5,stride=2, padding=padding)
        padding2 = (2*(num_classes-1) - num_in_channels - 3) * 1
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=padding2)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=padding2)

  


        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)

        self.fc1 = nn.Linear(476288, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.dropout(x, 0.25)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = F.dropout(x, 0.25)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "MyCNNNetwork"
class MyCNNNetwork2(nn.Module):
    def __init__(self, num_in_channels, num_classes):
        super(MyCNNNetwork2, self).__init__()
        padding = (2*(num_classes-1) - num_in_channels - 3) * 1
        self.conv1_1 = nn.Conv2d(in_channels=num_in_channels, out_channels=32, kernel_size=3, padding=padding)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=padding)

        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=padding)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=padding)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=padding)


        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.fc1 = nn.Linear(2163200, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.dropout(x, 0.25)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = F.dropout(x, 0.25)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "MyCNNNetwork"
class MyCNNNetwork(nn.Module):
    def __init__(self, num_in_channels, num_classes):
        super(MyCNNNetwork, self).__init__()
        padding = (2*(num_classes-1) - num_in_channels - 3) * 1
        self.conv1_1 = nn.Conv2d(in_channels=num_in_channels, out_channels=32, kernel_size=3, padding=padding)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=padding)

        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=padding)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=padding)


        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.fc1 = nn.Linear(1081600, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.dropout(x, 0.25)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = F.dropout(x, 0.25)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "MyCNNNetwork"

def training(model, data_loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for batch_idx, (inputs, labels) in enumerate(data_loader):

        # zero the parameter gradients
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        # backward
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        if batch_idx % 10 == 0:
            writer.add_scalar(f"{model.name()}/Train_loss", running_loss / len(data_loader.dataset))
            writer.add_scalar(f"{model.name()}/Train_acc", running_corrects.double() / len(data_loader.dataset))
            print(f'Training Batch: {batch_idx:4} of {len(data_loader)}')

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print('-' * 10)
    print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')
    

    return epoch_loss, epoch_acc


def test(model, data_loader, criterion, writer, device):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    # do not compute gradients
    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if batch_idx % 10 == 0:
                writer.add_scalar(f"{model.name()}/Test_loss", running_loss / len(data_loader.dataset))
                writer.add_scalar(f"{model.name()}/Test_acc", running_corrects.double() / len(data_loader.dataset))
                print(f'Test Batch: {batch_idx:4} of {len(data_loader)}')

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print('-' * 10)
    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    return epoch_loss, epoch_acc


def plot(train_history, test_history, save_dir, metric, num_epochs):

    plt.title(f"Validation/Test {metric} vs. Number of Training Epochs")
    plt.xlabel(f"Training Epochs")
    plt.ylabel(f"Validation/Test {metric}")
    plt.plot(range(1, num_epochs + 1), train_history, label="Train")
    plt.plot(range(1, num_epochs + 1), test_history, label="Test")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.savefig(f"{save_dir}/{model_name}/{metric}.png")
    plt.show()

if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set seed for reproducability
    torch.manual_seed(0)

    # hyperparameter
    # TODO: find good hyperparameters
    # batch_size = ...
    # num_epochs = ...
    # learning_rate = ...
    # momentum = ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int, help="Size per batch")
    parser.add_argument("--num_epochs", default=5, type = int, help="Number of iterations")
    parser.add_argument("--learning_rate", type = float, default=1e-4)
    parser.add_argument("--momentum", type= float, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_name", default = "standard3")
    parser.add_argument("--save_dir")

    args = parser.parse_args()
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    momentum = args.momentum
    num_workers = args.num_workers
    model_name = args.model_name
    save_dir = args.save_dir

    num_classes = len(CATEGORIES.keys())
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    ## model setup

    if model_name == "standard":
        model = MyCNNNetwork(1, num_classes).to(device)
    elif model_name == "standard2":
        model = MyCNNNetwork2(1, num_classes).to(device)
    elif model_name == "standard3":
        model = MyCNNNetwork3(1, num_classes).to(device)
    else:
        print("No correct model name provided....\n Exiting....")
        exit()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter("/pvc/logs/cv03")

    # load train and test data
    root = './data'
    train_set = datasets.FashionMNIST(root=root,
                                    train=True,
                                    transform=transform,
                                    download=True)
    test_set = datasets.FashionMNIST(root=root,
                                    train=False,
                                    transform=transform,
                                    download=True)

    loader_params = {
        'batch_size': batch_size,
        'num_workers': num_workers  # increase this value to use multiprocess data loading
    }

    train_loader = DataLoader(dataset=train_set, shuffle=True, **loader_params)
    test_loader = DataLoader(dataset=test_set, shuffle=False, **loader_params)

    

    train_acc_history = []
    test_acc_history = []

    train_loss_history = []
    test_loss_history = []

    best_acc = 0.0
    since = time.time()

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # train
        training_loss, training_acc = training(model, train_loader, optimizer,
                                            criterion, device)
        train_loss_history.append(training_loss)
        train_acc_history.append(training_acc)

        # test
        test_loss, test_acc = test(model, test_loader, criterion, device)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        # overall best model
        if test_acc > best_acc:
            best_acc = test_acc
            #  best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(
        f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s'
    )
    print(f'Best val Acc: {best_acc:4f}')

    # plot loss and accuracy curves
    train_acc_history = [h.cpu().numpy() for h in train_acc_history]
    test_acc_history = [h.cpu().numpy() for h in test_acc_history]

    plot(train_acc_history, test_acc_history, save_dir, f'{model_name}_accuracy', num_epochs)
    plot(train_loss_history, test_loss_history, save_dir, f'{model_name}_loss', num_epochs)

    # plot examples
    example_data, _ = next(iter(test_loader))
    example_data = example_data.to(device)
    with torch.no_grad():
        output = model(example_data)

        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0].cpu(), cmap='gray', interpolation='none')
            plt.title("Pred: {}".format(CATEGORIES[output.data.max(
                1, keepdim=True)[1][i].item()]))
            plt.xticks([])
            plt.yticks([])
        plt.savefig( f"{save_dir}/{model_name}/examples.png")
        plt.show()
