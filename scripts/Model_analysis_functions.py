import torch
import random
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from Initiate_PSiteDataset import PSiteDataset, ReInitPSiteDataset
from PSite_Models import PSitePredictV4
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, f1_score, accuracy_score


def test_model(model: torch.nn.Module,
               test_dl: DataLoader,
               dev: torch.device = 'cpu'):
    """
    Tests the correctness/accuracy by which a given `model` can predict the data by calculating accuracy metric scores
    (AUROC, AUPRC and F1) for a given batch of data `test_dl`. Optional argument `dev` selects on which device to run
    the calculations.

    Returns the metric scores, as well as the true model outputs and calculated output probabilities.
    """
    y_probs, y_test, y_true = [], [], []

    model.eval()
    with torch.inference_mode():
        for batch in test_dl:
            X, y, *_ = batch
            X, y = X.to(dev), y.to(dev)
            y_pred = model(X)

            y_prob = torch.nn.functional.softmax(y_pred, dim=1)
            y_class = torch.argmax(y_prob, dim=1)

            y_probs.append(y_prob[:, 1].cpu().numpy())
            y_test.append(y_class.cpu().numpy())
            y_true.append(y.cpu().numpy())

    y_probs = np.concatenate(y_probs)
    y_test = np.concatenate(y_test)
    y_true = np.concatenate(y_true)

    return y_probs, y_test, y_true


def calculate_performance_metrics(y_probs, y_test, y_true):
    """
    Takes the output from the test_model() function and returns the performance metric scores such as AUROC, AUPRC, F1
    and the weighted average, alongside an array of FPR, TPR, precision and recall values to construct the ROC and PRC.
    """
    auroc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)
    f1 = f1_score(y_true, y_test)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg = accuracy_score(y_true, y_test)

    return auroc, auprc, f1, fpr, tpr, recall, precision, avg


def shuffle_dataset_labels(dataset: PSiteDataset):
    """
    Shuffles the binary labels randomly around and assigns them to new data. Input is the PSiteDataset itself
    and output is a ReInitPSiteDataset instance.
    """
    # Shuffle the labels randomly
    shuffled_labels = [x[1] for x in dataset]
    random.shuffle(shuffled_labels)

    # Create a new dataset by pairing the data of the original dataset with the shuffled labels
    shuffled_data = [(dataset[i][0], shuffled_labels[i], dataset[i][2]) for i in range(len(dataset))]
    data_points, shuffled_labels, sequences = zip(*shuffled_data)

    shuffled_dataset = ReInitPSiteDataset(data_points, shuffled_labels, sequences)

    return shuffled_dataset


def chance_performance_test(test_data: PSiteDataset, model_loc: str, n_dls: int = 50):
    """
    Determines whether the AUC scores (AUROC and AUPRC) calculated by the specified classifier for the test data
    were merely due to chance or statistically significant. Several randomly shuffled permutations of the test dataset
    are generated and significance of the actual test dataset in comparison to these is calculated using
    1-sample t-tests.
    Input:
        - test_data: PSiteDataset instance of the test dataset
        - model_loc: path/to/saved_model_state_dict
        - n_dls: number of random datasets/dataloaders to generate
    Output:
        dictionary with 2 keys: 'AUROC' and 'AUPRC', each containing a tuple with:
         - the actual test data AUC score,
         - a list with AUC scores of the random permutations
         - the t-statistic
         - the p-value
        Also prints 2 lines signifying the statistical (in)significance of the observed AUC score.
    """
    stat_dict = {}

    # Create test dataloader instances (from both the original and randomly shuffled datasets)
    test_dls = [DataLoader(dataset=test_data, batch_size=32, shuffle=False)]

    for n in range(n_dls):
        random.seed(n)
        new_ds = shuffle_dataset_labels(test_data)
        test_dls.append(DataLoader(dataset=new_ds, batch_size=32, shuffle=False))

    channel_nr = len(test_data[0][0])  # Needed later in the PSitePredict model initialization

    # Load in the saved model and run the test data on it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outputs = []
    for dl in test_dls:
        model = PSitePredictV4(input_shape=channel_nr,
                               hidden_units=40,
                               output_shape=2,
                               field_length=61,
                               kernel=3,
                               pad_idx=1,
                               dropout=0.3)

        model.load_state_dict(torch.load(model_loc))
        model.to(device)

        # Extract the AUROC and AUPRC scores from the output
        outputs.append(calculate_performance_metrics(*test_model(model, dl, device))[:2])

    # Perform a one-sample t-test to determine whether the observed AUROC/AUPRC scores are statistically different from
    # a distribution of randomly shuffled datasets
    obs_roc, obs_prc = outputs[0]
    sample_pop_roc, sample_pop_prc = zip(*outputs[1:])

    t_stat_roc, p_value_roc = stats.ttest_1samp(sample_pop_roc, obs_roc)
    t_stat_prc, p_value_prc = stats.ttest_1samp(sample_pop_prc, obs_prc)

    # Print some output
    if p_value_roc < 0.05:
        prefix = ''
    else:
        prefix = 'in'
    print(f"Observed AUROC score {obs_roc:.5f} is statistically {prefix}significant (p = {p_value_roc:.5f}).")

    if p_value_prc < 0.05:
        prefix = ''
    else:
        prefix = 'in'
    print(f"Observed AUPRC score {obs_prc:.5f} is statistically {prefix}significant (p = {p_value_prc:.5f}).")

    stat_dict['AUROC'] = sample_pop_roc, obs_roc, t_stat_roc, p_value_roc
    stat_dict['AURPC'] = sample_pop_prc, obs_prc, t_stat_prc, p_value_prc
    return stat_dict


def bootstrapping(probs_a: np.ndarray, probs_b: np.ndarray, true: np.ndarray,
                  n_samples: int = 100, random_state: int = 42):
    """
    Performs a bootstrapping analysis on the predictions of two separate models and returns the distribution of the
    AUC scores and confidence intervals.
    """
    assert len(probs_a) == len(probs_b) and len(probs_a) == len(true), "Classifier arrays not the same length."
    auroc_scores_a, auprc_scores_a = [], []
    auroc_scores_b, auprc_scores_b = [], []

    rng = np.random.RandomState(random_state)

    while len(auroc_scores_a) != n_samples:
        idc = rng.randint(len(probs_a), size=len(probs_a))
        probs_a, probs_b, true = probs_a[idc], probs_b[idc], true[idc]
        try:
            auroc_a, auroc_b = roc_auc_score(true, probs_a), roc_auc_score(true, probs_b)
            auprc_a, auprc_b = average_precision_score(true, probs_a), average_precision_score(true, probs_b)

            auroc_scores_a.append(auroc_a)
            auprc_scores_a.append(auprc_a)
            auroc_scores_b.append(auroc_b)
            auprc_scores_b.append(auprc_b)

        except ValueError:
            continue

    ci_auroc_a = np.around(np.percentile(auroc_scores_a, (2.5, 97.5)), decimals=3)
    ci_auprc_a = np.around(np.percentile(auprc_scores_a, (2.5, 97.5)), decimals=3)
    ci_auroc_b = np.around(np.percentile(auroc_scores_b, (2.5, 97.5)), decimals=3)
    ci_auprc_b = np.around(np.percentile(auprc_scores_b, (2.5, 97.5)), decimals=3)

    return ((auroc_scores_a, ci_auroc_a, auprc_scores_a, ci_auprc_a),
            (auroc_scores_b, ci_auroc_b, auprc_scores_b, ci_auprc_b))


def randomized_permutation_test(probs_a: np.ndarray, probs_b: np.ndarray, true: np.ndarray,
                                n_samples: int = 100, random_state: int = 42):
    """
    Randomly shuffles the probability scores between two models `probs_a` and `probs_b` and calculates the AUC score of
    this new distribution `n_samples` number of times. Returns the AUROC and AUPRC scores for both original models,
    plus two lists with the AUC scores for the ROC and precision-recall curves for the new, random permutation.

    Aside from the predicted probabilities, this function also needs the `true` labels of the data samples as input
    for AUC calculation. Optionally, you can change the seed for randomization (`random_state`) of the shuffled data.
    """
    assert len(probs_a) == len(probs_b), "Lists of probability scores are not the same size."

    auroc_a, auroc_b = roc_auc_score(true, probs_a), roc_auc_score(true, probs_b)
    auprc_a, auprc_b = average_precision_score(true, probs_a), average_precision_score(true, probs_b)

    new_auroc_scores, new_auprc_scores = [], []

    rng = np.random.RandomState(random_state)
    for _ in range(n_samples):
        random_swap_idcs = rng.choice([0, 1], size=len(probs_a))

        new_probs = np.where(random_swap_idcs, probs_a, probs_b)

        new_auroc_scores.append(roc_auc_score(true, new_probs))
        new_auprc_scores.append(average_precision_score(true, new_probs))

    return (auroc_a, auprc_a), (auroc_b, auprc_b), new_auroc_scores, new_auprc_scores


def average_metricdicts(dicts: list):
    """
    Function returns a dictionary of accuracy metrics where each key-value pair is the mean average of all the
    corresponding keys in the input list of metric-containing dictionaries. This is mainly used to iron out the
    variation in accuracy metrics due to batch size differences.
    """
    avg_metrics = {}
    n_dicts = 0

    for d in dicts:
        # Does not take into account metrics that did not improve, as they would skew the outcome
        if d['val_loss'] != float("inf"):
            n_dicts += 1
            for metric, value in d.items():
                if metric not in avg_metrics:
                    avg_metrics[metric] = value[-1]
                else:
                    avg_metrics[metric] += value[-1]

    for key in avg_metrics:
        avg_metrics[key] /= n_dicts

    return avg_metrics


def average_output_tuples(data: list):
    """
    The input is a list of output tuples from the test_model function. The output of this function is a single tuple
    that is the mean average of the values in the original tuples.
    """
    num_tpls = len(data)
    tuple_sums = [0] * len(data[0])  # Will correspond to test_model() outputs 0-2 and 7
    arrays = [np.array(0)] * 4  # Will correspond to test_model() outputs 3-6

    for tpl in data:
        for i, value in enumerate(tpl):
            if isinstance(value, np.ndarray) and arrays[i - 3].shape == ():
                arrays[i - 3] = value
            else:
                # If the value is an array, needs to be concatenated, else can just be summed up
                if isinstance(value, np.ndarray):
                    arrays[i - 3] = np.concatenate((arrays[i - 3], value))
                else:
                    tuple_sums[i] += value

    avg_outputs = [tuple_sum / num_tpls for tuple_sum in tuple_sums]

    # The combined arrays are reworked in a way that unique values in array X are linked with their counterpart in array
    # Y, but when multiple occurrences of a value are found in array X, the corresponding values in Y are averaged.
    for i in range(0, 3, 2):
        value_positions = {}

        for index, value in enumerate(arrays[i]):
            value_positions.setdefault(value, []).append(index)

        for value, pos in value_positions.items():
            if len(pos) > 1:
                avg = np.mean(arrays[i + 1][pos])
                arrays[i + 1][pos] = avg

        unique_idc = np.unique(arrays[i], return_index=True)[1]
        arrays[i] = arrays[i][unique_idc]
        arrays[i+1] = arrays[i+1][unique_idc]

    return avg_outputs[0], avg_outputs[1], avg_outputs[2], arrays[0], arrays[1], arrays[2], arrays[3], avg_outputs[7]


def plot_model_hyperpar_changes(results: list):
    """
    Generate a graph of the changes in prediction performance as a result of a change in a hyperparameter.

    Input is a list of tuples with first the hyperparameter value and second a list of dictionaries with the performance
    metrics.

    Output is four subplots charting out the hyperparameter change (x) against the change in either the loss, accuracy,
    F1-score or specificity/recall (y) of the predictions.
    """
    hyperparam, metrics = zip(*results)
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    train_f1, val_f1 = [], []
    train_rec, val_rec = [], []
    train_spec, val_spec = [], []

    for metric_dict in metrics:
        train_loss.append(metric_dict['train_loss'])
        val_loss.append(metric_dict['val_loss'])
        train_acc.append(metric_dict['train_acc'])
        val_acc.append(metric_dict['val_acc'])
        train_f1.append(metric_dict['train_f1'])
        val_f1.append(metric_dict['val_f1'])
        train_rec.append(metric_dict['train_recall'])
        val_rec.append(metric_dict['val_recall'])
        train_spec.append(metric_dict['train_spec'])
        val_spec.append(metric_dict['val_spec'])

    # Set up the plot
    plt.figure(figsize=(15, 7))
    hp = "<Hyperparameter>"  # Change name based on which Hyperparameter is investigated

    # Plot the loss
    plt.subplot(2, 2, 1)
    plt.plot(hyperparam, train_loss, label="Train loss")
    plt.plot(hyperparam, val_loss, label="Val loss")
    plt.title("Loss")
    plt.legend()

    # Plot the accuracy
    plt.subplot(2, 2, 2)
    plt.plot(hyperparam, train_acc, label="Train accuracy")
    plt.plot(hyperparam, val_acc, label="Val accuracy")
    plt.title("Accuracy")
    plt.legend()

    # Plot the F1-score
    plt.subplot(2, 2, 3)
    plt.plot(hyperparam, train_f1, label="Train F1")
    plt.plot(hyperparam, val_f1, label="Val F1")
    plt.title("F1-score")
    plt.xlabel(hp)
    plt.legend()

    # Plot the recall and precision
    plt.subplot(2, 2, 4)
    plt.plot(hyperparam, train_rec, label="Train sensitivity")
    plt.plot(hyperparam, val_rec, label="Val sensitivity")
    plt.plot(hyperparam, train_spec, label="Train specificity")
    plt.plot(hyperparam, val_spec, label="Val specificity")
    plt.title("Sensitivity vs Specificity")
    plt.xlabel(hp)
    plt.legend()

    plt.savefig("ParameterChange_<psite>_<hyperparam>_<epoch>.png")  # Fill in the image's name
