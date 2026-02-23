from sklearn import datasets

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot as plt

def load_datasets():
    qsar = datasets.fetch_openml(data_id=45077, as_frame=True)
    pc4 = datasets.fetch_openml(data_id=1049, as_frame=True)

    return qsar, pc4



def no_pruning(X, y):

    dtc = DecisionTreeClassifier(criterion="entropy")
    y_scores = cross_val_predict(dtc, X, y, method="predict_proba", cv=10)

    auc = round(roc_auc_score(y, y_scores[:, 1]),  2)
    return auc, y_scores


def pre_pruning(X, y):

    dtc = DecisionTreeClassifier(criterion="entropy")
    parameters = [{"min_samples_leaf": [2, 4, 6, 8, 10]}]
    tuned_dtc = GridSearchCV(dtc, parameters, scoring="roc_auc", cv=5)

    y_scores = cross_val_predict(tuned_dtc, X, y, method="predict_proba", cv=10)
    auc = round(roc_auc_score(y, y_scores[:, 1]),   2)
    return auc, y_scores

def post_pruning(X, y):
    dtc = DecisionTreeClassifier(criterion="entropy")
    parameters = [{"ccp_alpha": [0.001, 0.005, 0.01, 0.05, 0.1]}]
    tuned_dtc = GridSearchCV(dtc, parameters, scoring="roc_auc", cv=5)
    y_scores = cross_val_predict(tuned_dtc, X, y, method="predict_proba", cv=10)
    auc = round(roc_auc_score(y, y_scores[:, 1]), 2)
    return auc, y_scores

def print_table(dataset_name, auc_no_pruning, auc_pre_pruning, auc_post_pruning):
    print(f"\n--- {dataset_name} AUC Results ---")
    print(f"{'Method':<20} {'AUC'}")

    print(f"{'No Pruning':<20} {auc_no_pruning}")
    print(f"{'Pre-Pruning':<20} {auc_pre_pruning}")
    print(f"{'Post-Pruning':<20} {auc_post_pruning}")


def plot_roc_curves(dataset_name, y, y_scores_no, y_scores_pre, y_scores_post, pos_label):

    fpr_no, tpr_no, _ = roc_curve(y, y_scores_no[:, 1], pos_label=pos_label)
    fpr_pre, tpr_pre, _ = roc_curve(y, y_scores_pre[:, 1], pos_label=pos_label)
    fpr_post, tpr_post, _ = roc_curve(y, y_scores_post[:, 1], pos_label=pos_label)

    plt.figure()
    plt.plot(fpr_no, tpr_no, label="No Pruning")
    plt.plot(fpr_pre, tpr_pre, label="Pre-Pruning")
    plt.plot(fpr_post, tpr_post, label="Post-Pruning")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"ROC Curves - {dataset_name}")
    plt.legend()
    plt.show()   

def main():
    qsar, pc4 = load_datasets()

    # the QSAR dataset
    X_qsar = qsar.data
    y_qsar = qsar.target
    pos_label_qsar = qsar.target.unique()[0]

    auc_no_qsar, scores_no_qsar = no_pruning(X_qsar, y_qsar)
    auc_pre_qsar, scores_pre_qsar = pre_pruning(X_qsar, y_qsar)
    auc_post_qsar, scores_post_qsar = post_pruning(X_qsar, y_qsar)

    print_table("QSAR", auc_no_qsar, auc_pre_qsar, auc_post_qsar)
    plot_roc_curves("QSAR", y_qsar, scores_no_qsar, scores_pre_qsar, scores_post_qsar, pos_label_qsar)

    # the  PC4 dataset
    X_pc4 = pc4.data
    y_pc4 = pc4.target
    pos_label_pc4 = pc4.target.unique()[0]

    auc_no_pc4, scores_no_pc4 = no_pruning(X_pc4, y_pc4)
    auc_pre_pc4, scores_pre_pc4 = pre_pruning(X_pc4, y_pc4)
    auc_post_pc4, scores_post_pc4 = post_pruning(X_pc4, y_pc4)

    print_table("PC4", auc_no_pc4, auc_pre_pc4, auc_post_pc4)
    plot_roc_curves("PC4", y_pc4, scores_no_pc4, scores_pre_pc4, scores_post_pc4, pos_label_pc4)

main()


