import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Load results
def load_results(cancer_type: str) -> Dict:
    """Load results for a specific cancer type"""
    try:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', cancer_type)
        results_file = os.path.join(results_dir, 'results.json')
        
        logger.info(f"Loading results from: {results_file}")
        
        if not os.path.exists(results_file):
            logger.error(f"Results file not found: {results_file}")
            return None
            
        with open(results_file, 'r') as f:
            results = json.load(f)
            logger.info(f"Successfully loaded results for {cancer_type}")
            return results
    except Exception as e:
        logger.error(f"Error loading results for {cancer_type}: {str(e)}")
        return None

# Create layout
app.layout = html.Div([
    html.H1("Cancer Detection Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    # Cancer type selection
    html.Div([
        html.Label("Select Cancer Type:", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='cancer-type-dropdown',
            options=[
                {'label': 'Breast Cancer (BRCA)', 'value': 'BRCA'},
                {'label': 'Bladder Cancer (BLCA)', 'value': 'BLCA'},
                {'label': 'Liver Cancer (LIHC)', 'value': 'LIHC'},
                {'label': 'Prostate Cancer (PRAD)', 'value': 'PRAD'}
            ],
            value='BRCA',
            style={'width': '50%', 'margin': '0 auto'}
        )
    ], style={'margin': '20px'}),
    
    # Performance metrics
    html.Div([
        html.H2("Model Performance", style={'color': '#2c3e50'}),
        html.Div(id='performance-metrics', style={'display': 'flex', 'justifyContent': 'space-around'})
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
    
    # Main visualizations
    html.Div([
        html.Div([
            html.H2("Confusion Matrix", style={'color': '#2c3e50'}),
            dcc.Graph(id='confusion-matrix')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            html.H2("ROC Curve", style={'color': '#2c3e50'}),
            dcc.Graph(id='roc-curve')
        ], style={'width': '50%', 'display': 'inline-block'})
    ], style={'margin': '20px'}),
    
    # Training history
    html.Div([
        html.H2("Training History", style={'color': '#2c3e50'}),
        dcc.Graph(id='training-history')
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
    
    # Cross-validation results
    html.Div([
        html.H2("Cross-Validation Results", style={'color': '#2c3e50'}),
        dcc.Graph(id='cv-results')
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
])

# Callbacks
@app.callback(
    [Output('performance-metrics', 'children'),
     Output('confusion-matrix', 'figure'),
     Output('roc-curve', 'figure'),
     Output('training-history', 'figure'),
     Output('cv-results', 'figure')],
    [Input('cancer-type-dropdown', 'value')]
)
def update_dashboard(cancer_type):
    """Update dashboard components based on selected cancer type"""
    try:
        # Load results
        results = load_results(cancer_type)
        if results is None:
            return html.Div("Error loading results"), {}, {}, {}, {}
        
        # Performance metrics
        try:
            logger.info("Creating performance metrics section")
            logger.info(f"Final report: {results['final_report']}")
            logger.info(f"AUC: {results['auc']}")
            
            metrics = html.Div([
                html.Div([
                    # Accuracy
                    html.Div([
                        html.H3("Accuracy", style={'color': '#2c3e50', 'marginBottom': '10px', 'fontSize': '1.2em', 'textAlign': 'center'}),
                        html.P(f"{float(results['final_report']['accuracy']):.3f}", 
                              style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#2c3e50', 'margin': '0', 'textAlign': 'center'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'white', 
                             'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'margin': '0 10px'}),
                    
                    # AUC
                    html.Div([
                        html.H3("AUC", style={'color': '#2c3e50', 'marginBottom': '10px', 'fontSize': '1.2em', 'textAlign': 'center'}),
                        html.P(f"{float(results['auc']):.3f}", 
                              style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#2c3e50', 'margin': '0', 'textAlign': 'center'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'white', 
                             'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'margin': '0 10px'}),
                    
                    # Precision (weighted avg)
                    html.Div([
                        html.H3("Precision", style={'color': '#2c3e50', 'marginBottom': '10px', 'fontSize': '1.2em', 'textAlign': 'center'}),
                        html.P(f"{float(results['final_report']['weighted avg']['precision']):.3f}", 
                              style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#2c3e50', 'margin': '0', 'textAlign': 'center'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'white', 
                             'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'margin': '0 10px'}),
                    
                    # Recall (weighted avg)
                    html.Div([
                        html.H3("Recall", style={'color': '#2c3e50', 'marginBottom': '10px', 'fontSize': '1.2em', 'textAlign': 'center'}),
                        html.P(f"{float(results['final_report']['weighted avg']['recall']):.3f}", 
                              style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#2c3e50', 'margin': '0', 'textAlign': 'center'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'white', 
                             'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'margin': '0 10px'}),
                    
                    # F1-score (weighted avg)
                    html.Div([
                        html.H3("F1-score", style={'color': '#2c3e50', 'marginBottom': '10px', 'fontSize': '1.2em', 'textAlign': 'center'}),
                        html.P(f"{float(results['final_report']['weighted avg']['f1-score']):.3f}", 
                              style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#2c3e50', 'margin': '0', 'textAlign': 'center'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'white', 
                             'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'margin': '0 10px'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 
                         'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'flexWrap': 'wrap'})
            ], style={'width': '100%'})
            
            logger.info("Successfully created performance metrics section")
            
            # Confusion matrix
            cm = results['confusion_matrix']
            cm_fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="True", color="Count"),
                x=['Normal', 'Cancer'],
                y=['Normal', 'Cancer'],
                title=f"Confusion Matrix - {cancer_type}",
                color_continuous_scale='Blues'
            )
            cm_fig.update_layout(
                title_x=0.5,
                title_font_size=20,
                font=dict(size=14)
            )
            
            # ROC curve (using AUC value)
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name='Random',
                line=dict(dash='dash', color='gray')
            ))
            roc_fig.add_trace(go.Scatter(
                x=[0, 0.5, 1],
                y=[0, results['auc'], 1],
                name=f'AUC = {results["auc"]:.3f}',
                line=dict(color='#2c3e50', width=3)
            ))
            roc_fig.update_layout(
                title=f"ROC Curve - {cancer_type}",
                title_x=0.5,
                title_font_size=20,
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                font=dict(size=14),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Training history
            try:
                history = results['history']
                epochs = list(range(1, len(history['accuracy']) + 1))
                
                history_fig = go.Figure()
                history_fig.add_trace(go.Scatter(
                    x=epochs,
                    y=history['accuracy'],
                    name='Training Accuracy',
                    line=dict(color='#2c3e50', width=3)
                ))
                history_fig.add_trace(go.Scatter(
                    x=epochs,
                    y=history['val_accuracy'],
                    name='Validation Accuracy',
                    line=dict(color='#e74c3c', width=3)
                ))
                history_fig.update_layout(
                    title=f"Training History - {cancer_type}",
                    title_x=0.5,
                    title_font_size=20,
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy",
                    font=dict(size=14),
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
            except Exception as e:
                logger.error(f"Error creating training history plot: {str(e)}")
                history_fig = go.Figure()
            
            # Cross-validation results
            try:
                cv_results = results['cv_results']
                cv_df = pd.DataFrame(cv_results)
                
                cv_fig = go.Figure()
                metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1-score']
                colors = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#f1c40f']
                
                for metric, color in zip(metrics, colors):
                    cv_fig.add_trace(go.Box(
                        y=cv_df[metric],
                        name=metric.capitalize(),
                        marker_color=color
                    ))
                
                cv_fig.update_layout(
                    title=f"Cross-Validation Results - {cancer_type}",
                    title_x=0.5,
                    title_font_size=20,
                    yaxis_title="Score",
                    font=dict(size=14),
                    showlegend=False
                )
            except Exception as e:
                logger.error(f"Error creating CV results plot: {str(e)}")
                cv_fig = go.Figure()
            
            return metrics, cm_fig, roc_fig, history_fig, cv_fig
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {str(e)}")
            return html.Div(f"Error: {str(e)}"), {}, {}, {}, {}
            
    except Exception as e:
        logger.error(f"Error in update_dashboard: {str(e)}")
        return html.Div(f"Error: {str(e)}"), {}, {}, {}, {}

if __name__ == '__main__':
    app.run(debug=True, port=8051) 