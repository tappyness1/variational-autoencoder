import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, PILToTensor
import matplotlib.pyplot as plt


def get_load_data(root = "data", dataset = "FashionMNIST", download = False):

    if dataset == "FashionMNIST":
        training_data = datasets.FashionMNIST(
            root=root,
            train=True,
            download=download,
            transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root=root,
            train=False,
            download=download,
            transform=ToTensor()
        )
    
    elif dataset == "Flowers102":
        training_data = datasets.Flowers102(
            root=root,
            split="test",
            download=download,
            transform=Compose([Resize((572,572)), ToTensor()]) 
        )

        test_data = datasets.Flowers102(
            root=root,
            split = "train",
            download=download,
            transform=Compose([Resize((572,572)), ToTensor()])
        )
    elif dataset == "VOCDetection":
        training_data = datasets.VOCDetection(
            root=root,
            image_set ="train",
            download=download,
            transform=Compose([Resize((572, 572)),ToTensor()]),
            # target_transform = Compose([Resize((227, 227)),ToTensor()])
        )

        test_data = datasets.VOCDetection(
            root=root,
            image_set ="val",
            download=download,
            transform=Compose([Resize((572, 572)), ToTensor()]),
            # target_transform = Compose([Resize((227, 227)),ToTensor()])
        )

    elif dataset == "VOCSegmentation":
        training_data = datasets.VOCSegmentation(
            root=root,
            image_set ="train",
            download=download,
            transform=Compose([Resize((572, 572)),ToTensor()]),
            target_transform = Compose([Resize((388, 388)),ToTensor()])
        )

        test_data = datasets.VOCSegmentation(
            root=root,
            image_set ="val",
            download=download,
            transform=Compose([Resize((572, 572)), ToTensor()]),
            target_transform = Compose([Resize((388, 388)),ToTensor()])
        )
    return training_data, test_data


if __name__ == "__main__":
    # train, test = get_load_data(root = "../data")
    # img, label = train[1]
    # plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()

    # # for local testing
    # train, test = get_load_data(root = "../data", dataset = "VOCSegmentation", download = False)  
    # img, smnt = train[12] 
    # print (smnt)

    # # for gcp or whatever
    train, test = get_load_data(root = "../data", dataset = "FashionMNIST", download = False)
