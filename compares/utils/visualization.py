import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional

def plot_metrics_comparison(
    results: List[Dict[str, float]],
    metric_names: List[str],
    model_names: Optional[List[str]] = None,
    title: str = "Models Comparison"
) -> go.Figure:
    """
    Создание сравнительного графика метрик для разных моделей
    """
    df = pd.DataFrame(results)
    if model_names:
        df['Model'] = model_names
        
    fig = go.Figure()
    
    for metric in metric_names:
        fig.add_trace(go.Bar(
            name=metric,
            x=df['Model'] if model_names else df.index,
            y=df[metric],
            text=df[metric].round(3),
            textposition='auto',
        ))
        
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        showlegend=True
    )
    
    return fig

def plot_learning_curves(
    history: Dict[str, List[float]],
    metric_names: List[str],
    title: str = "Learning Curves"
) -> go.Figure:
    """
    Построение графиков обучения
    """
    df = pd.DataFrame(history)
    
    fig = go.Figure()
    
    for metric in metric_names:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[metric],
            mode='lines+markers',
            name=metric
        ))
        
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Score",
        showlegend=True
    )
    
    return fig 