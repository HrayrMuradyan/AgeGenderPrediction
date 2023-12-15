import torch
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calc_mse(net, x, y):
    if net.training:       
        net.eval()
    loss_f = torch.nn.MSELoss()
    with torch.no_grad():
        loss = loss_f(net(x), y).item() 
    return loss 


def evaluate(net, loader, loss_fn, classification):
    net.eval()
    running_loss = 0
    if classification:
        running_corrects = 0
    for idx, batch in enumerate(loader):
        if classification:
            inputs, _, labels= batch
        else:
            inputs, labels, _ = batch
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            prediction = net(inputs).squeeze(-1) 
        if classification:
            prediction = torch.sigmoid(prediction)
            running_corrects += (prediction.round() == labels).sum()
        loss_value = loss_fn(prediction, labels).item()
        running_loss += loss_value
    net.train()
    if classification:
        return running_loss, running_corrects
    else:
        return running_loss



def trainer(net, train_loader, val_loader, optimizer, loss_fn, task='Age', epochs=10, path_to_save='./weights/'):
    try:
        if task.lower() == 'age':
            classification = False
        elif task.lower() == 'gender':
            classification = True

        net.train()
        batch_count = len(train_loader)
        data_count = len(train_loader.dataset)
        train_list = []
        val_list = []
        print(f'\033[1mStarting the training of task - {task} prediction.\n\033[0m')

        best_val_loss = 10**6

        for epoch in range(1, epochs + 1): 

            running_loss = 0
            if classification:
                running_corrects = 0
            zeros_count = len(str(epochs)) - len(str(epoch))

            for batch_idx, batch in enumerate(train_loader):

                if classification:
                    inputs, _, labels = batch
                else:
                    inputs, labels, _ = batch
                inputs, labels = inputs.to(device), labels.to(device)
                prediction = net(inputs).squeeze(-1) 

                if classification:
                    prediction = torch.sigmoid(prediction)
                loss_value = loss_fn(prediction, labels)
                loss_value.backward()

                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss_value
                if classification:
                    running_corrects = running_corrects + (prediction.round()==labels).sum()
                percent = (batch_idx + 1) / batch_count * 100

                print(f'\rEPOCH {"0" * zeros_count}{epoch}/{epochs} |',f'batch train loss = {loss_value.item()/len(labels):.7f} |' ,f'{"=" * int(percent / 5)}> {percent:.2f}%', end='')
            if classification:
                val_loss, val_acc = evaluate(net=net, loader=val_loader, loss_fn=loss_fn, classification=classification)
                val_acc = val_acc / len(val_loader.dataset)
                
            else:
                val_loss = evaluate(net=net, loader=val_loader, loss_fn=loss_fn, classification=classification)

            val_loss = val_loss / len(val_loader.dataset)
            val_list.append(val_loss)

            if val_loss <= best_val_loss:
                file = f'{task}_Model_{net.__class__.__name__}.pt'
                torch.save(net.state_dict(), path_to_save + file)
                print('\n\nValidation loss was better than before, saving weights in:', file)
                best_val_loss = val_loss
            train_list.append(running_loss.cpu().detach().numpy() / data_count)
            if classification:
                train_acc = running_corrects/len(train_loader.dataset)

            print(f'\nAverage train loss = {running_loss / data_count:.6f}')
            print(f'\nValidation loss = {val_loss:.6f}')
            if classification:
                print(f'\nTrain accuracy = {train_acc:.4f}')
                print(f'\nValidation accuracy = {val_acc:.4f}')
            print('-----------------------------------\n')
            
    except KeyboardInterrupt:
        
        print('The training was interrupted manually by the user. Plotting the training results...\n')
        
    finally:    
        plt.figure(figsize=(9,9))
        plt.plot(train_list)
        plt.plot(val_list)
        plt.legend(["trainloss", "valloss"], loc ="upper right")
        plt.show()