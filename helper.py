import torch
from torchvision import transforms
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def age_pred_acc(net, image, label, error=5, verbose=True):
    image_tensor = transforms.ToTensor()(image)
    image_tensor = image_tensor.to(device)
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    label = torch.tensor(label).to(device)
    net.eval()
    with torch.no_grad():
        prediction = net(image_tensor).squeeze(-1)
    pred_error = np.abs((prediction-label).cpu().detach())
    res = 1 if pred_error <= error else 0
    if verbose:
        print('-'*50)
        print('Prediction of the age =', np.round(prediction.item(),2))
        print('The real age =', label.item())
        print('-'*50)
        print('The error =', np.round(pred_error.item(),2))
        print(f'The accuracy of the prediction based on the acceptable error({error}) is {res}')
        print('-'*50)
    return res

def predict(net, image):
    image_tensor = transforms.ToTensor()(image)
    image_tensor = image_tensor.to(device)
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    net.eval()
    with torch.no_grad():
        prediction = net(image_tensor).squeeze(-1)
    return prediction

def calc_preds(net, X_data, y_data, classification=False):
    preds = []
    for i, x in enumerate(X_data):
        prediction = predict(net, x)
        if classification:
            prediction = torch.sigmoid(prediction)
            pred_error = np.abs(y_data[i, 1]-prediction.cpu())
            preds.append((y_data[i, 1], prediction.item(), pred_error.item()))
        else:
            pred_error = np.abs(prediction.cpu()-y_data[i, 0])
            preds.append((y_data[i, 0], prediction.item(), pred_error.item()))
    return preds

def find_worst_preds(predictions):
    errors_list = [error[-1] for error in predictions]
    highest_errors = np.argsort(errors_list)[::-1]
    return highest_errors
