import torch
def evalModel(model, criterion, device, eval_loader):
    
    model.eval()
    running_loss = 0.0
    correct= 0
    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds= outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
        
    eval_loss = running_loss / len(eval_loader.dataset)
    eval_acc= correct/ len(eval_loader.dataset)
        
        
    return eval_loss, eval_acc
    