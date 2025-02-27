import streamlit as st
import pickle

import plotly.express as px
import plotly.graph_objects as go

def load_evaluation_metrics(filepath: str):
    with open(filepath, 'rb') as f:
        evaluation_metrics = pickle.load(f)
    return (
        evaluation_metrics['confusion_matrices'],
        evaluation_metrics['roc_curves'],
        evaluation_metrics['auc_scores'],
        evaluation_metrics['f1_scores']
    )

def plot_roc_curve(selected_category, roc_curves, auc_scores):
    fig = go.Figure()
    roc_data = roc_curves[selected_category]
    fig.add_trace(go.Scatter(
        x=roc_data['fpr'],
        y=roc_data['tpr'],
        name=selected_category
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash'),
        name='Random'
    ))
    fig.update_layout(
        title=f"AUC-ROC Curve - {selected_category}, AUC={auc_scores[selected_category]:.3f}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=500,
        height=450,
        showlegend=False
    )
    return fig

def plot_confusion_matrix(selected_category, confusion_matrices, f1_scores):
    labels = ['OK', 'NOK']
    conf_matrix = confusion_matrices[selected_category]
    f1_score = f1_scores[selected_category]
    fig = px.imshow(
        conf_matrix,
        labels=dict(x="True Label", y="Predicted Label"),
        x=labels,
        y=labels,
        color_continuous_scale='Reds',
        width=500,
        height=500,
    )
    for i in range(len(labels)):
        for j in range(len(labels)):
            fig.add_annotation(
                x=j,
                y=i,
                text=str(conf_matrix[i, j]),
                showarrow=False,
                font=dict(size=14)
            )
    fig.update_layout(
        title=f"Confusion Matrix - {selected_category}, F1: {f1_score:.3f}",
        xaxis_title="True Label",
        yaxis_title="Predicted Label",
        coloraxis_showscale=False  # This line removes the vertical color scale
    )
    return fig

