import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_and_preprocess_data(data):
    """
    Load and preprocess the dataset
    """
    # Handle missing values
    df = data.copy()
    df = df.dropna()
    
    # Split features and target
    X = df.drop(['SMILES', 'Activity'], axis=1)
    y = df['Activity']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, scaler

def create_molecular_profile_plot(df):
    """
    Create an elegant circular visualization of molecular properties with enhanced design
    """
    features = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA']
    feature_names = ['Molecular Weight', 'Lipophilicity', 'H-Bond Donors', 'H-Bond Acceptors', 'Topological Surface Area']
    
    # Calculate statistics for active and inactive compounds
    active_means = df[df['Activity'] == 1][features].mean()
    inactive_means = df[df['Activity'] == 0][features].mean()
    active_stds = df[df['Activity'] == 1][features].std()
    inactive_stds = df[df['Activity'] == 0][features].std()
    
    # Create angles for the radar chart
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
    
    # Close the plot by appending the first value
    values_active = active_means.values.tolist()
    values_active.append(values_active[0])
    values_inactive = inactive_means.values.tolist()
    values_inactive.append(values_inactive[0])
    angles = list(angles)
    angles.append(angles[0])
    feature_names = feature_names + [feature_names[0]]
    
    # Create subplots: radar chart and statistics
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'polar'}, {'type': 'table'}]],
        column_widths=[0.7, 0.3]
    )
    
    # Add traces for active and inactive compounds
    fig.add_trace(
        go.Scatterpolar(
            r=values_active,
            theta=feature_names,
            name='Active Compounds',
            fill='toself',
            fillcolor='rgba(99, 161, 255, 0.3)',
            line=dict(color='#63a1ff', width=2),
            hovertemplate='<b>%{theta}</b><br>' +
                        'Value: %{r:.2f}<br>' +
                        '<extra>Active Compounds</extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatterpolar(
            r=values_inactive,
            theta=feature_names,
            name='Inactive Compounds',
            fill='toself',
            fillcolor='rgba(255, 99, 132, 0.3)',
            line=dict(color='#ff6384', width=2),
            hovertemplate='<b>%{theta}</b><br>' +
                        'Value: %{r:.2f}<br>' +
                        '<extra>Inactive Compounds</extra>'
        ),
        row=1, col=1
    )
    
    # Add statistics table
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Property', 'Active μ±σ', 'Inactive μ±σ'],
                fill_color='#f0f2f6',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[
                    feature_names[:-1],
                    [f'{m:.1f}±{s:.1f}' for m, s in zip(active_means, active_stds)],
                    [f'{m:.1f}±{s:.1f}' for m, s in zip(inactive_means, inactive_stds)]
                ],
                fill_color='white',
                align='left',
                font=dict(size=11)
            )
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showline=False,
                showticklabels=True,
                ticks='',
                gridcolor='rgba(0,0,0,0.1)',
                range=[0, max(max(values_active), max(values_inactive)) * 1.2]
            ),
            angularaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                linecolor='rgba(0,0,0,0.1)'
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        title={
            'text': 'Molecular Property Analysis',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#2c3e50')
        },
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Arial', size=12),
        height=500,
        margin=dict(l=0, r=0, t=100, b=0)
    )
    
    return fig

def create_evaluation_plots(y_true, y_pred, feature_importance, feature_names):
    """
    Create evaluation metric visualizations with enhanced design and interactivity
    """
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Confusion Matrix',
            'Feature Importance',
            'Performance Metrics',
            'Prediction Distribution'
        ),
        specs=[
            [{'type': 'heatmap'}, {'type': 'bar'}],
            [{'type': 'indicator'}, {'type': 'pie'}]
        ],
        vertical_spacing=0.2,
        horizontal_spacing=0.15
    )
    
    # Enhanced Confusion Matrix
    heatmap = go.Heatmap(
        z=cm,
        x=['Predicted Inactive', 'Predicted Active'],
        y=['Actually Inactive', 'Actually Active'],
        text=cm,
        texttemplate='%{text}',
        textfont={'size': 16, 'color': 'white'},
        colorscale=[
            [0, '#63a1ff'],
            [0.5, '#4a90e2'],
            [1, '#357abd']
        ],
        showscale=False,
        hovertemplate='<b>%{y}</b><br>' +
                    '%{x}<br>' +
                    'Count: %{text}<br>' +
                    '<extra></extra>'
    )
    fig.add_trace(heatmap, row=1, col=1)
    
    # Enhanced Feature Importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    bar = go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(
            color=importance_df['Importance'],
            colorscale=[
                [0, '#63a1ff'],
                [0.5, '#4a90e2'],
                [1, '#357abd']
            ],
            showscale=False
        ),
        hovertemplate='<b>%{y}</b><br>' +
                    'Importance: %{x:.3f}<br>' +
                    '<extra></extra>'
    )
    fig.add_trace(bar, row=1, col=2)
    
    # Performance Metrics Gauge
    fig.add_trace(
        go.Indicator(
            mode='gauge+number',
            value=f1 * 100,
            title={'text': 'F1 Score', 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#4a90e2'},
                'steps': [
                    {'range': [0, 50], 'color': '#ff6384'},
                    {'range': [50, 75], 'color': '#ffcd56'},
                    {'range': [75, 100], 'color': '#4bc0c0'}
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 4},
                    'thickness': 0.75,
                    'value': f1 * 100
                }
            }
        ),
        row=2, col=1
    )
    
    # Prediction Distribution Donut
    total = len(y_pred)
    correct = (y_true == y_pred).sum()
    incorrect = total - correct
    
    fig.add_trace(
        go.Pie(
            values=[correct, incorrect],
            labels=['Correct', 'Incorrect'],
            hole=0.7,
            marker=dict(
                colors=['#4bc0c0', '#ff6384']
            ),
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>' +
                        'Count: %{value}<br>' +
                        'Percentage: %{percent}<br>' +
                        '<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Model Evaluation Dashboard',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color='#2c3e50')
        },
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=800,
        font=dict(family='Arial', size=12),
        margin=dict(l=20, r=20, t=100, b=20)
    )
    
    # Update axes
    fig.update_xaxes(
        title_text='Importance Score',
        row=1, col=2,
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True
    )
    fig.update_yaxes(
        title_text='Features',
        row=1, col=2,
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True
    )
    
    return fig
