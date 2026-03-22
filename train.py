def trainModel(model, optimizer, criterion, train_loader, device):
    
    model.train()
    running_loss= 0.0
    correct= 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds= outputs.argmax(dim=1)
        correct+= (preds == labels).sum().item()
    
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct/ len(train_loader.dataset)
    return train_loss, train_acc
    
    


        