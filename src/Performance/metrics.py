from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import joblib
import pandas as pd
import os
import warnings
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ============================
# Evaluation and Visualization
# ============================


def evaluate_and_save_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessor=None,
    class_names=("M20", "F20", "M50", "F50"),
    output_dir="./evaluation_results",
):
    os.makedirs(output_dir, exist_ok=True)

    print("Evaluating on training data...")
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)
    evaluate_model(
        y_train, y_train_pred, y_train_proba, class_names, output_dir, "train"
    )

    print("Evaluating on test data...")
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    evaluate_model(y_test, y_test_pred, y_test_proba, class_names, output_dir, "test")


def evaluate_model(
    y_true,
    y_pred,
    y_pred_proba=None,
    class_names=("M20", "F20", "M50", "F50"),
    output_dir="./evaluation_results",
    data_type="data",
):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, target_names=class_names)

    print(f"{data_type.capitalize()} Accuracy: {acc:.4f}")
    print(f"{data_type.capitalize()} Macro F1-score: {macro_f1:.4f}")
    print(f"{data_type.capitalize()} Weighted F1-score: {weighted_f1:.4f}")
    print(f"\n{data_type.capitalize()} Classification Report:\n{report}")

    with open(
        os.path.join(output_dir, f"{data_type}_classification_report.txt"), "w"
    ) as f:
        f.write(report)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    annotations = [[str(cell) for cell in row] for row in cm]
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            showscale=True,
            text=annotations,
            texttemplate="%{text}",
        )
    )
    fig_cm.update_layout(
        title=f"{data_type.capitalize()} Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
    )
    fig_cm.write_html(os.path.join(output_dir, f"{data_type}_confusion_matrix.html"))
    fig_cm.show()

    # ROC & PR Curves
    if y_pred_proba is not None:
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        fig_roc = go.Figure()
        fig_pr = go.Figure()

        for i in range(len(class_names)):
            # ROC
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            fig_roc.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC - {class_names[i]}")
            )

            # PR
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_pred_proba[:, i]
            )
            ap_score = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            fig_pr.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"PR - {class_names[i]} (AP={ap_score:.2f})",
                )
            )

        # Random ROC line
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")
            )
        )

        # Layouts
        fig_roc.update_layout(
            title=f"{data_type.capitalize()} ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        fig_pr.update_layout(
            title=f"{data_type.capitalize()} Precision-Recall Curves",
            xaxis_title="Recall",
            yaxis_title="Precision",
        )

        # Save and show
        fig_roc.write_html(os.path.join(output_dir, f"{data_type}_roc_curve.html"))
        fig_pr.write_html(os.path.join(output_dir, f"{data_type}_pr_curve.html"))
        fig_roc.show()
        fig_pr.show()

        # AUC Score
        auc_score = roc_auc_score(y_true_bin, y_pred_proba, average="macro")
        print(f"{data_type.capitalize()} Macro ROC AUC Score: {auc_score:.4f}")
        with open(os.path.join(output_dir, f"{data_type}_roc_auc.txt"), "w") as f:
            f.write(f"Macro ROC AUC Score: {auc_score:.4f}\n")


# ============================
# Main Execution
# ============================

if __name__ == "__main__":
    import sys

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    sys.path.append(project_root)

    # Load model and preprocessor
    model_path = os.path.abspath(
        os.path.join(script_dir, "..", "Modeling", "classifier92.joblib")
    )
    preprocessor_path = os.path.abspath(
        os.path.join(script_dir, "..", "Modeling", "preprocessor.joblib")
    )
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Load and prepare data
    df = pd.read_csv("merged_output_features.csv")
    X = df.drop(columns=["label"])
    y = df["label"]

    # Corrected train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply preprocessor
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Evaluate model
    evaluate_and_save_model(model, X_train, y_train, X_test, y_test, preprocessor)
