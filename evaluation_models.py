import torch.nn as nn
import torch
#import matplotlib.pyplot as plt
import numpy as np

from EvaluationDatasets import RegressionDataset
from torch.utils.data import DataLoader, Dataset, random_split
#from torcheval.metrics import MeanSquaredError, R2Score, MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix, BinaryF1Score, BinaryAccuracy, BinaryPrecision, BinaryRecall
#from torchmetrics.regression import MeanSquaredError
import torchmetrics as tm
import wandb
#from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


class LinearClassifierLayer(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        #self.linear.weight.data.normal_(mean=0.0, std=0.01)
        #self.linear.bias.data.zero_()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        if self.num_labels !=2: #cross-entropy loss takes logits
            return self.linear(x) 
        else:
            return self.sigmoid(self.linear(x))
    
class LogisticRegression(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, input_dim, output_dim, n_layers = 1, logits = False, dropout = None, bottleneck = None, last_layer = False):
        super().__init__()
        if bottleneck:
            intermed = bottleneck
        else:
            intermed = input_dim
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.n_layers = n_layers
        self.logits = logits
        self.first = nn.Linear(input_dim, intermed)
        self.linear = nn.Linear(intermed, intermed)
        self.active = nn.ReLU()
        self.last = nn.Linear(intermed, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = None
        self.last_layer = last_layer
        if dropout and dropout != 0.0:
            self.dropout = nn.Dropout(dropout) #probability

    def forward(self, x):
        # flatten
        if self.dropout:
            x = self.dropout(x)
        x = self.batchnorm(x)
        #print(x.shape)
        x = self.active(self.first(x))
        #print(x.shape)
        if self.n_layers > 1:
            for i in range(1, self.n_layers):
                x = self.active(self.linear(x))
                #print(x.shape)
        if self.last_layer:
            return x    
        x = self.last(x)
        #print(x.shape)
        if self.logits:
            return x
        else:
            return self.sigmoid(x) #need sigmoid... not softmax cuz binary
    
def get_last_layer(model, train_loader, device='cuda'):
    model.last_layer = True
    model.eval()
    out = []
    labs = []
    for x, y in train_loader:
        x = x.to(device)
        #y = y.to(device)
        #clear gradient 
        #optimizer.zero_grad()
        #make a prediction 
        z = model(x) 
        out.append(z)
        labs.append(y.cpu())
    outv = torch.concat(out)
    outl = torch.concat(labs).squeeze()
    return outv, outl

def train_and_eval_model(model, train_loader, test_loader, criterion, optimizer, eval_fn, n_epochs = 300, n_eval = 10, device='cuda', verbose=True, keep_best_acc = True, class_names = ['FRII','FRI'], confmat=True):
    """train & validate model (using accuracy)"""
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    loss_list, losses = [],[]
    af1,aac,apre,arec = [],[],[],[]

    model.train()
    #print(next(model.parameters()).device)
    for epoch in range(n_epochs):
        for x, y in train_loader:
            x = x.cuda() #.to(device)
            #print(x.shape)
            y = y.cuda() #.to(device)
            #clear gradient 
            optimizer.zero_grad()
            #make a prediction 
            z = model(x) #originaly just model(x)
            if criterion._get_name() == "CrossEntropyLoss":
                loss = criterion(z,y.long().squeeze())
            else:
                loss = criterion(z,y) #need long for crossentropy .long().squeeze()
            # calculate gradients of parameters 
            loss.backward()
            # update parameters 
            optimizer.step()
            
            loss_list.append(loss.data)
        losses.append(loss.item())

        # evaluate accuracy at end of every n_eval epochs
        if epoch % n_eval == 0:
            model.eval()
            acc, f1, precision, recall = eval_fn(test_loader, model, device=device, n_classes=len(class_names)) #formerly eval_binary
            if verbose:
                if type(acc) == torch.Tensor and (acc.ndim > 1 or len(acc) == len(class_names)):
                    #do one for each...
                    ldict = {"epoch":epoch,"loss": loss.item()}
                    for i,cn in enumerate(class_names):
                        ldict[f"accuracy_{cn}"] = acc[i]
                        ldict[f"F1 score_{cn}"] = f1[i]
                        ldict[f"precision_{cn}"] = precision[i]
                        ldict[f"recall_{cn}"] = recall[i]
                    ldict[f"accuracy"] = acc.mean().numpy() #do i need to average
                    ldict[f"F1 score"] = f1.mean().numpy()
                    ldict[f"precision"] = precision.mean().numpy()
                    ldict[f"recall"] = recall.mean().numpy()
                else:
                    ldict = {"epoch":epoch,"loss": loss.item(), "accuracy": acc, 'F1 score': f1,'precision':precision,'recall':recall}

                wandb.log(ldict)
            else: #keep them anyway and average last 10 epochs...
                af1.append(f1)
                aac.append(acc)
                apre.append(precision)
                arec.append(recall)

            # if keep_best_acc and f1 > best_acc:
            #     #print(f1)
            #     best_acc = f1
            #     best_weights = copy.deepcopy(model.state_dict())
            
        elif verbose:
            wandb.log({"loss": loss.item()})
    # restore model and return best accuracy
    if keep_best_acc == True:
        best_acc = losses[-1], acc, f1, precision, recall
    
    if verbose:
        metrics, y_pred, y_true = eval_fn(test_loader, model, device=device, return_vals=True, n_classes=len(class_names))
    return best_acc, model

def eval_binary(test_loader, model, device='cuda', metrics = [None], return_vals = False, n_classes=2): #BinaryAccuracy(), BinaryF1Score()
    model.eval()

    y_true = torch.tensor([], dtype=torch.long, device=device)
    all_outputs = torch.tensor([], device=device)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            y_true = torch.cat((y_true, targets), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)

    mevals = []
   
    acc = tm.Accuracy(task = "binary", average="micro") 
    f1 = tm.F1Score(task='binary', num_classes=2, average="none")
    precision = tm.Precision(task = 'binary',num_classes=2, average="none")
    recall = tm.Recall(task='binary', num_classes=2, average="none")
    
    mevals.append(acc(all_outputs.cpu().squeeze().round(),y_true.cpu().squeeze()))
    mevals.append(f1(all_outputs.cpu().squeeze().round(),y_true.cpu().squeeze()))
    mevals.append(precision(all_outputs.cpu().squeeze().round(),y_true.cpu().squeeze()))
    mevals.append(recall(all_outputs.cpu().squeeze().round(),y_true.cpu().squeeze()))
    if return_vals:
        y_true = y_true.cpu().numpy().squeeze()
        y_pred = all_outputs.cpu().numpy().squeeze()
        return mevals, y_pred, y_true
    return mevals

def eval_multiclass(test_loader, model, device='cuda', metrics = [None], return_vals = False, n_classes=4):

    y_true = torch.tensor([], dtype=torch.long, device=device)
    all_outputs = torch.tensor([], device=device)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            y_true = torch.cat((y_true, targets), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)

    mevals = []
    
    acc = tm.Accuracy(task='multiclass',average="none", num_classes = n_classes) #across all classes
    f1 = tm.F1Score(task='multiclass',num_classes=n_classes, average="none")
    precision = tm.Precision(task='multiclass',num_classes=n_classes, average="none")
    recall = tm.Recall(task='multiclass',num_classes=n_classes, average="none")
    
    y_pred = torch.argmax(all_outputs, dim= 1).cpu().round()

    mevals.append(acc(y_pred,y_true.cpu().squeeze()))
    mevals.append(f1(y_pred,y_true.cpu().squeeze()))
    mevals.append(precision(y_pred,y_true.cpu().squeeze()))
    mevals.append(recall(y_pred,y_true.cpu().squeeze()))
 
            
    if return_vals:
        y_true = y_true.cpu().numpy().squeeze()
        y_pred = y_pred.cpu().numpy().squeeze()
        return mevals, y_pred, y_true
    return mevals 


class LinearRegressionModel(nn.Module):
 
    def __init__(self, dim, num_labels=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(dim, num_labels)  # One in and one out
 
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
def train_linregress(model, train_loader, criterion, optimizer, epoch, num_epochs, device, verbose=False):
    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        train_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    avg_loss = train_loss/len(train_loader)
    if epoch % 10 == 0 and verbose:
        print(f'Epoch [{epoch + 1:03}/{num_epochs:03}] | Train Loss: {avg_loss:.4f}')
    return train_loss/len(train_loader)

def validate_linregress(model, val_loader, criterion, device, verbose=False):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_loss = val_loss / len(val_loader)
    if verbose:
        print(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss

def run_linregress(X,y, dataset_lengths = [0.8,0.2], num_epochs = 300, device = 'cuda', lr=0.001, verbose=False, weight_decay = 1e-5):
    if isinstance(X,np.ndarray): #else it's a dataset..
        dataset = RegressionDataset(X, y)
        train_dataset, test_dataset = random_split(dataset, lengths=dataset_lengths)
        embed_dim = X.shape[1]
    else:
        train_dataset = X
        test_dataset = y
        embed_dim = train_dataset.dataset.X[train_dataset.indices].numpy().shape[1]
    train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = 64, shuffle = True)

    model = LinearRegressionModel(embed_dim, num_labels=1).cuda()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr = lr, weight_decay = weight_decay)

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        train_losses.append(train_linregress(model, train_loader, criterion, optimizer, epoch, num_epochs, device, verbose=verbose))
        if epoch % 10 == 0:
            val_losses.append(validate_linregress(model, test_loader, criterion, device, verbose=verbose))
            if verbose:
                wandb.log({"epoch":epoch,"train_loss":train_losses[-1], "val_loss":val_losses[-1]})
        elif verbose:
            wandb.log({"epoch":epoch,"train_loss":train_losses[-1]})
    return train_losses, val_losses, model

def eval_linregress(X,y, model, device = 'cuda'):
    if isinstance(X,np.ndarray): #else it's a dataset..
        dataset = RegressionDataset(X, y)
    else:
        dataset = X
    alldata_loader = DataLoader(dataset = dataset, batch_size = 64, shuffle = False)
    y_pred, y_true = [], []

    with torch.no_grad():
        for inputs, targets in alldata_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            y_pred.extend(outputs.cpu().squeeze())
            y_true.extend(targets.cpu().squeeze())
    mevals = []

    mm = tm.MeanSquaredError()
    met = mm(torch.Tensor(y_pred), torch.Tensor(y_true))
    mevals.append(met)

    return mevals, y_pred, y_true

