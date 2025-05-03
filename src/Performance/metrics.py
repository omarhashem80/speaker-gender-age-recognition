from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, roc_curve, roc_auc_score
)
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Function to evaluate the model and save the results
def evaluate_and_save_model(model, X_train, y_train, X_test, y_test, preprocessor=None, class_names=None, output_dir='./evaluation_results'):
    # Apply preprocessing if needed
    if preprocessor:
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)

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


def evaluate_model(y_true, y_pred, y_pred_proba=None, class_names=None, output_dir='./evaluation_results', data_type="data"):
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
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
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

# Example usage
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../Modeling/classifier.joblib")

    model_path = os.path.abspath(model_path)

    # Load the model
    model = joblib.load(model_path)
    
    def load_and_concatenate(folder_path):
        dataframes = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)
    # df = pd.read_csv("D:/College/Third Year/Second Term/Pattern/Project/speaker-gender-age-recognition/all_features.csv")
    script_directory = os.path.dirname(os.path.abspath(__file__))

    features_folder_path = os.path.join(script_directory, "../../features_200")

    folder_path = os.path.abspath(features_folder_path)
    target = 'label'
    df = load_and_concatenate(folder_path)

    # Split the data and preprocess
    target_column = 'label'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preprocessing: StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Evaluate and save the results
    evaluate_and_save_model(model, X_train_scaled, y_train, X_test_scaled, y_test, preprocessor=scaler, class_names=["M20", "F20", "M50", "F50"])
