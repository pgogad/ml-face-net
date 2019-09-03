import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from siamese_net import Config, SiameseNetworkDataset, SiameseNetwork

net = SiameseNetwork()
net.load_state_dict(torch.load(Config.model_path))
net.eval()


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(image_folder_dataset=folder_dataset_test,
                                        transform=transforms.Compose(
                                            [transforms.Resize((100, 100)), transforms.ToTensor()]),
                                        should_invert=False)

test_data_loader = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=True)
data_iter = iter(test_data_loader)
x0, _, _ = next(data_iter)

for i in range(10):
    _, x1, label2 = next(data_iter)
    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = net(Variable(x0), Variable(x1))
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
