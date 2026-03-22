import matplotlib.pyplot as plt
import numpy as np
import torch

#afficher quelques exemples
def imshow(images, labels, classes, n=5):
    images = images[:n]
    labels = labels[:n]
 
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images_denorm = images * std + mean
    images_denorm = images_denorm.clamp(0, 1)
    
    ncols = min(n, 8)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 3))
    axes = np.array(axes).reshape(-1) 
    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            npimg = images_denorm[i].numpy().transpose(1, 2, 0)
            ax.imshow(npimg)
            ax.set_title(classes[labels[i].item()], fontsize=8, wrap=True)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visu(loader, classes):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    imshow(images, labels, classes)

def classDistribution(full_dataset):
    class_counts = full_dataset.data["Class"].value_counts()
    plt.figure(figsize=(20, 5))
    plt.bar(range(len(class_counts)),
            class_counts.values, color="steelblue")
    plt.xticks(range(len(class_counts)),
            [c.split(",")[0] for c in class_counts.index],
            rotation=90, fontsize=6)
    plt.title("Distribution des classes")
    plt.ylabel("Nombre d'images")
    plt.tight_layout()
    plt.show()

def plotLossAcc(train_loss_list, val_loss_list, train_acc_list, val_acc_list):
    plt.subplot(1,2,1)
    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.plot(range(len(val_loss_list)), val_loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("loss curve")

    plt.subplot(1,2,2)
    plt.plot(range(len(train_acc_list)), train_acc_list)
    plt.plot(range(len(val_acc_list)), val_acc_list)

    plt.xlabel("Accuracy")
    plt.ylabel("Loss")
    plt.title("accuracy curve")
    plt.show()