# config/styles.py
from dataclasses import dataclass, field
from typing import Dict

from dataclasses import dataclass, field

@dataclass
class ColorPalette:
    primary: str = "#3b82f6"
    success: str = "#22c55e" 
    warning: str = "#f59e0b"
    error: str = "#ef4444"
    background: str = "#f8fafc"
    text: Dict[str, str] = field(default_factory=lambda: {
        "primary": "#111827",
        "secondary": "#4b5563",
        "light": "#9ca3af"
    })

def get_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        .stApp {
            font-family: 'Inter', sans-serif !important;
            background-color: #f8fafc;
        }
        
        /* Header Styling */
        .app-header {
            background: linear-gradient(to right, #2563eb, #3b82f6);
            color: white;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }
        
        /* Card Components */
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 0.75rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            margin-bottom: 1rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Metrics Display */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
            margin: 1.5rem 0;
        }
        
        .metric-item {
            background: white;
            padding: 1.5rem;
            border-radius: 0.75rem;
            border-left: 4px solid #3b82f6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 600;
            color: #111827;
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: #6b7280;
            margin-top: 0.5rem;
        }
        
        /* Status Indicators */
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.375rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        .status-badge.success { 
            background: #dcfce7;
            color: #166534;
        }
        
        .status-badge.warning {
            background: #fef3c7;
            color: #92400e;
        }
        
        .status-badge.error {
            background: #fee2e2;
            color: #991b1b;
        }
        
        /* Table Improvements */
        .dataframe {
            border: none !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .dataframe thead th {
            background-color: #f8fafc !important;
            font-weight: 600 !important;
            padding: 0.75rem 1rem !important;
        }
        
        .dataframe tbody td {
            padding: 0.75rem 1rem !important;
        }
        
        /* Button Styling */
        .stButton > button {
            width: 100%;
            height: 2.75rem;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s;
            background-color: #3b82f6;
            color: white;
        }
        
        .stButton > button:hover {
            background-color: #2563eb;
            transform: translateY(-1px);
        }
        
        /* File Upload Area */
        .uploadedFile {
            border: 2px dashed #e2e8f0;
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            background: #f8fafc;
        }

        /* Tab Navigation */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background-color: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre;
            background-color: transparent;
            border-radius: 4px;
            color: #444;
            font-weight: 500;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(59, 130, 246, 0.1);
            color: #3b82f6;
        }

        /* Progress Bar */
        .stProgress > div > div {
            background-color: #3b82f6;
        }
    </style>
    """

def apply_style(component_type: str, **kwargs) -> str:
    """Generate HTML with appropriate styling for different components"""
    if component_type == "metric":
        return f"""
        <div class="metric-item">
            <div class="metric-value">{kwargs.get('value', '')}</div>
            <div class="metric-label">{kwargs.get('label', '')}</div>
        </div>
        """
    elif component_type == "status":
        return f"""
        <span class="status-badge {kwargs.get('status', 'success')}">
            {kwargs.get('text', '')}
        </span>
        """
    return ""