import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from time import time
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from matplotlib import pyplot as plt


'''
This script contains several functions to train a Phosphorylation site prediction model:
1. extract_from_batch(): Only extract what's necessary from dataloader batches to run the model and loss function
2. fill_metric_dict(): Fills a dictionary with the performance metrics extracted from a confusion matrix
3. plot_curves(): plots the ROC and PR curves from a singular trained model
4. average_metrics(): Averages the performance metrics stored in a dictionary across all batches in a training epoch.
2. train_step(): Training the model for 1 epoch
3. val_step(): Model validation (overfitting, underfitting, loss reduction) for 1 epoch
4. test_model(): Test the model whenever a new optimum has been reached and calculates the AUROC, AUPRC and F1-score
5. train_model(): Combines the train, val and test functions and runs for a specified number of epochs
'''


def extract_from_batch(batch, dev: torch.device, model: nn.Module, loss_fun: nn.Module = None):
    """
    Extracts the data (X) and labels (y) from a batch in the dataloader and applies the model to the data,
    predicting the labels and the remaining loss, should a loss function be given.

    Returns the loss, true labels and predicted labels.
    """
    X, y, *_ = batch
    X, y = X.to(dev), y.to(dev)

    y_pred = model(X)
    loss = loss_fun(y_pred, y) if loss_fun is not None else None

    return loss, y, y_pred


def fill_metric_dict(matrix: np.ndarray, metric_dict):
    """
    Extracts the performance metrics (recall, precision, specificity and f1-score) from the confusion matrix and stores
    them in a dictionary. If a metric cannot be determined, defaults to 0 instead of throwing an error.

    Returns the filled dictionary.
    """
    tn, fp, fn, tp = matrix.ravel()
    metric_dict["prec"] += tp / (tp + fp) if tp > 0 or fp > 0 else 0
    metric_dict["recall"] += tp / (tp + fn)
    metric_dict["spec"] += tn / (tn + fp) if tn > 0 or fp > 0 else 0
    metric_dict["f1"] += 2 * tp / (2 * tp + fp + fn)
    metric_dict["acc"] += (tp + tn) / (tp + fp + tn + fn)

    return metric_dict


def average_metrics(metric_dict: dict, batch_nr, adjust: int):
    """
    Given a dict of the total accumulated accuracy, loss, specificity and sensitivity values from either a
    train or test step, print the averaged values for one epoch.
    Averages are taken over the entire dataloader (= nr of batches), with spec and sens adjusted for 1 class batches

    Returns the modified dictionary.
    """
    for k in metric_dict.keys():
        if "loss" in k:
            metric_dict[k] /= batch_nr  # Averages over 1 epoch by dividing by the number of batches
        else:
            metric_dict[k] /= (batch_nr - adjust)  # Since not all batches may contain both labels, omit those
    return metric_dict


def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fun: nn.Module,
               optim: torch.optim.Optimizer,
               dev: torch.device):
    """
    Performs a training step on a model trying to learn from a dataloader. Takes as inputs a model instance,
    the dataloader, the loss function, the optimizer algorithm and the device to run it on.

    Returns a dictionary with the performance metrics adjusted to the entire epoch.
    """
    model.train()

    train_dict = {"loss": 0, "recall": 0,
                  "spec": 0, "prec": 0,
                  "f1": 0, "acc": 0}
    neg_batches = 0

    # Loop through the dataloader's batches
    for batch in dataloader:
        loss, y, y_pred = extract_from_batch(batch, dev, model, loss_fun)

        train_dict["loss"] += loss.item()  # Calculate loss per batch
        y_pred_labels = torch.argmax(y_pred.softmax(dim=1), dim=1)  # Transformed into binary class designations
        matrix = confusion_matrix(y.cpu(), y_pred_labels.cpu(), labels=[0, 1])

        if any(matrix.sum(axis=1) == 0):
            neg_batches += 1  # Batches where only 1 of the classes is present is omitted from metric calculations
        else:
            train_dict = fill_metric_dict(matrix, train_dict)

        # Optimizer without gradient update
        optim.zero_grad()
        # Backpropagation
        loss.backward()
        # Optimizer step
        optim.step()

    return average_metrics(train_dict, len(dataloader), neg_batches)


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fun: torch.nn.Module,
             dev: torch.device):
    """
    Run the model on a validation dataset (as a dataloader) in inference mode and a loss function on a specified device.

    Returns a dictionary with the performance metrics adjusted to the entire epoch.
    """
    model.eval()

    val_dict = {"loss": 0, "recall": 0,
                "spec": 0, "prec": 0,
                "f1": 0, "acc": 0}

    neg_batches = 0

    with torch.inference_mode():
        # Loop through batch
        for batch in dataloader:
            loss, y, val_pred = extract_from_batch(batch, dev, model, loss_fun)

            val_dict["loss"] += loss.item()
            val_pred_labels = torch.argmax(val_pred, dim=1)
            matrix = confusion_matrix(y.cpu(), val_pred_labels.cpu(), labels=[0, 1])
            if any(matrix.sum(axis=1) == 0):
                neg_batches += 1  # Batches with only 1 of the classes is omitted from metric calculation
            else:
                val_dict = fill_metric_dict(matrix, val_dict)

    return average_metrics(val_dict, len(dataloader), neg_batches)


def test_model(model: nn.Module,
               test_dl: DataLoader,
               dev: torch.device,
               plot: int = 0):
    """
    Run a trained `model` on a test dataset (`test_dl`) and calculate the AUROC and AUPRC. If the optional argument plot
    equals True, it will also plot both AUC.

    Returns a dictionary with the AUROC, AUPRC, F1 and Accuracy scores.
    """
    y_probs, y_test, y_true = [], [], []
    results = {}

    model.eval()
    with torch.inference_mode():
        for batch in test_dl:
            _, y, y_pred = extract_from_batch(batch, dev, model)

            y_prob = nn.functional.softmax(y_pred, dim=1)
            y_class = torch.argmax(y_prob, dim=1)

            y_probs.append(y_prob[:, 1].cpu().numpy())
            y_test.append(y_class.cpu())
            y_true.append(y.cpu().numpy())

    y_probs = np.concatenate(y_probs)
    y_test = np.concatenate(y_test)
    y_true = np.concatenate(y_true)

    results["auroc"] = roc_auc_score(y_true, y_probs)
    results["auprc"] = average_precision_score(y_true, y_probs)
    results["acc"] = accuracy_score(y_true, y_test)
    results["f1"] = f1_score(y_true, y_test)

    if plot:
        plot_curves(y_true, y_probs, results)

    return results


def train_model(model: nn.Module,
                epochs: int,
                train_dl: DataLoader,
                val_dl: DataLoader,
                test_dl: DataLoader,
                loss_fun: nn.Module,
                optim: torch.optim.Optimizer,
                dev: torch.device = 'cpu',
                plot: int = 0):
    """
    This function will train a given `model` for a given number of `epochs` with given train, valid and test datasets
    (`train_dl`, `valid_dl` and `test_dl`), loss function (`loss_fun`) and optimizer (`optim`). A device `dev` can be
    specified to train on; default device is the CPU. If plot equals True, the ROC and PR curves are plotted and saved.
    The function will print the train and test accuracy/loss and the run time of the program.

    Returns a tuple with the dictionary with the test restuls of the latest test_step() and a dicitonary with the full
    progress of optimization of the performance metrics.
    """
    results = {}

    best_loss = float("inf")
    best_recall = 0
    best_spec = 0
    best_prec = 0
    best_f1 = 0
    best_acc = 0
    best_epoch = 0
    epochs_since_improvement = 0
    test_results = None

    start_time = time()
    for epoch in tqdm(range(epochs)):
        epoch = int(epoch)

        # Perform the training and validation step
        train_dict = train_step(model, train_dl, loss_fun, optim, dev)
        val_dict = val_step(model, val_dl, loss_fun, dev)

        # Append the average metric scores for 1 epoch to the results
        for key, value in train_dict.items():
            results.setdefault(f"train_{key}", []).append(value)
        for key, value in val_dict.items():
            results.setdefault(f"val_{key}", []).append(value)

        # Keep a record of rising/stagnating validation loss and decrease the learning rate accordingly
        if epoch > 0 and round(results["val_loss"][-1], 3) >= round(results["val_loss"][-2], 3):
            epochs_since_improvement += 1
        else:
            epochs_since_improvement = 0

        if epochs_since_improvement >= 3:
            for param_group in optim.param_groups:
                param_group['lr'] *= 0.5
                print(f"\nStagnating improvement at Epoch {epoch+1}: Reducing learning rate by a factor of 2.\n")

        # Keep a record of the best combined validation loss and validation sensitivity (not necessarily global best)
        if (round(val_dict["loss"], 4) <= round(best_loss, 4) and
                abs(val_dict["recall"] - train_dict["recall"]) <= 0.066):
            best_loss = val_dict["loss"]
            best_recall = val_dict["recall"]
            best_spec = val_dict["spec"]
            best_prec = val_dict["prec"]
            best_f1 = val_dict["f1"]
            best_acc = val_dict["acc"]
            best_epoch = epoch + 1

            # Determine the AUROC and AUPRC for these epochs
            test_results = test_model(model, test_dl, dev, plot)
            print(f'Optimum at Epoch {best_epoch}:\t'
                  f'AUROC = {test_results["auroc"]:.3f}\tAUPRC = {test_results["auprc"]:.3f}\t'
                  f'F1 = {test_results["f1"]:.3f}\tAccuracy = {test_results["acc"]*100:.3f}')

            path = '/data/home/arendvc/'
            torch.save(model.state_dict(), path + f'ModelV4_(STY)_(rep)_E{best_epoch}_{best_loss:.5f}.pth')
            print(f'Val loss: {best_loss:.5f}')

    end_time = time()
    total_time = end_time - start_time

    print(f"\nRuntime: {total_time/60:.2f} min on", str(next(model.parameters()).device))
    print(f"\nBest Epoch: {best_epoch}\nVal loss: {best_loss:.5f}\t| Val acc: {best_acc*100:.3f}\t"
          f"| Val recall: {best_recall*100:.2f}%\nVal spec: {best_spec*100:.2f}%\t"
          f"| Val precision: {best_prec*100:.2f}%\t| Val f1: {best_f1:.3f}\n")

    return test_results, results


def plot_curves(y_true: np.ndarray, y_probs: np.ndarray, metric_dict: dict):
    """
    Creates a plot with 2 subplots (the ROC and the PRC with the Area Under the Curve) of the test dataset.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    plt.figure(figsize=(17, 8))

    # Plot ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metric_dict["auroc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR curve (AUC = {metric_dict["auprc"]:.3f} | F1 = {metric_dict["f1"]:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")

    plt.savefig('AUCurves_ModelV4_(ds)_(rep).png')
    plt.close()
