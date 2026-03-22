from dataset import *
from model import *
from visu import *
from evaluation import *
from train import *
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


TRAIN_DIR="kaggle/train"
TRAIN_CSV="kaggle/train.csv"


if __name__=="__main__":
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #preparer et explorer les données
    full_train= CoinDataset(TRAIN_CSV, TRAIN_DIR, istrain=True)
    nb_classes= len(full_train.classes)
    classes= full_train.classes
    print(f"{nb_classes} classes en tout")
    print(f"Classes : {classes[:10]}")
    

    n_total= len(full_train)
    n_val = int(n_total * 0.20)
    n_test= int(n_total * 0.20)
    n_train= n_total - n_val - n_test
    trainset, validationset, testset = random_split(full_train, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(10))
    
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True,num_workers=2, collate_fn=collate)
    val_loader   = DataLoader(validationset,   batch_size=64, shuffle=False, num_workers=2, collate_fn=collate)
    test_loader = DataLoader(testset,batch_size=64, shuffle=False, num_workers=2, collate_fn=collate)
    
    visu(train_loader, classes)
    classDistribution(full_train)
    #initialiser le modèle
    model = build_alexnet(num_classes=nb_classes, pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.005)
    
    #entrainement+validation
    num_epochs = 15
    val_loss_list = [] 
    val_acc_list = []
    train_loss_list = []
    train_acc_list = []
    for epoch in range(num_epochs):
        train_loss, train_acc = trainModel(model, optimizer, criterion, train_loader, device)
        val_loss, val_acc = evalModel(model, criterion, device, val_loader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f},Train Accuracy: {train_acc:.6f}, Validation Accuracy : {val_acc:.6f}")
        
    
    
    plotLossAcc(train_loss_list, val_loss_list, train_acc_list, val_acc_list)
    #evaluation
    test_loss, test_acc = evalModel(model, criterion, device, test_loader)
    print(f"Test Loss     : {test_loss:.6f}")
    print(f"Test Accuracy : {test_acc*100:.6f}%")