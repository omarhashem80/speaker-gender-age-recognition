from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, roc_curve, roc_auc_score
)
import joblib
import os
from sklearn.preprocessing import  label_binarize
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


# Function to evaluate the model and save the results
def evaluate_and_save_model(model, X_train, y_train, X_test, y_test, preprocessor=None, class_names=("M20", "F20", "M50", "F50"), output_dir='./evaluation_results'):
    # if preprocessor:
    #     X_test = preprocessor.transform(X_test)

    # Predict on training data
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)

    # Predict on test data
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate model on training data
    print("Evaluating on training data...")
    evaluate_model(y_train, y_train_pred, y_train_pred_proba, class_names, output_dir, "train")

    # Evaluate model on test data
    print("Evaluating on test data...")
    evaluate_model(y_test, y_test_pred, y_test_pred_proba, class_names, output_dir, "test")


def evaluate_model(y_true, y_pred, y_pred_proba=None, class_names=("M20", "F20", "M50", "F50"), output_dir='./evaluation_results', data_type="data"):
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred, target_names=class_names)

    print(f"{data_type.capitalize()} Accuracy: {acc:.4f}")
    print(f"{data_type.capitalize()} Macro F1-score: {macro_f1:.4f}")
    print(f"{data_type.capitalize()} Weighted F1-score: {weighted_f1:.4f}")
    print(f"\n{data_type.capitalize()} Classification Report:\n", report)

    # Save classification report to a text file
    with open(os.path.join(output_dir, f"{data_type}_classification_report.txt"), 'w') as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    annotations = [[str(cell) for cell in row] for row in cm]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        showscale=True,
        text=annotations,
        texttemplate="%{text}",
        hoverinfo="z"
    ))
    fig_cm.update_layout(
        title=f"{data_type.capitalize()} Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    fig_cm.write_html(os.path.join(output_dir, f"{data_type}_confusion_matrix.html"))
    fig_cm.show()

    # ROC Curve (multi-class)
    if y_pred_proba is not None:
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        fig_roc = go.Figure()
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC - {class_names[i]}'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig_roc.update_layout(
            title=f"{data_type.capitalize()} ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True
        )
        fig_roc.write_html(os.path.join(output_dir, f"{data_type}_roc_curve.html"))
        fig_roc.show()

        auc_score = roc_auc_score(y_true_bin, y_pred_proba, average='macro')
        print(f"{data_type.capitalize()} Macro ROC AUC Score: {auc_score:.4f}")
        with open(os.path.join(output_dir, f"{data_type}_roc_auc.txt"), 'w') as f:
            f.write(f"Macro ROC AUC Score: {auc_score:.4f}\n")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../Modeling/classifier92.joblib")
    model_path = os.path.abspath(model_path)
    # Load the model
    model = joblib.load(model_path)
    preprocessor_path = os.path.join(script_dir, "../Modeling/preprocessor.joblib")
    preprocessor_path = os.path.abspath(model_path)
    preprocessor = joblib.load(preprocessor_path)

    from src.Modeling.preprocessor import preprocessing
    import pandas as pd
    df = pd.read_csv("../data.csv")
    X_train_resampled, y_train_resampled, X_test_transformed, y_test, pipeline = preprocessing(df, 'label')
    evaluate_and_save_model(model, X_train_resampled, y_train_resampled, X_test_transformed, y_test, preprocessor)
