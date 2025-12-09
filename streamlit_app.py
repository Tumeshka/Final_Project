"""
Portfolio Environmental Impact Analyzer - Streamlit App
========================================================
Interactive web application for analyzing portfolio environmental impacts
using the ENCORE framework and Cross-Sector Impact Engine.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
import io

# Import the engine
from cross_sector_impact_engine import CrossSectorImpactEngine

# Page configuration
st.set_page_config(
    page_title="Portfolio Impact Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Fix white text on white background
st.markdown("""
<style>
    /* Fix dropdown/selectbox text color */
    .stSelectbox > div > div {
        color: #1f2937 !important;
        background-color: #f8fafc !important;
    }
    .stSelectbox label {
        color: #1f2937 !important;
    }
    .stSelectbox [data-baseweb="select"] {
        color: #1f2937 !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }
    
    /* Fix slider labels */
    .stSlider label {
        color: #1f2937 !important;
    }
    .stSlider [data-baseweb="slider"] {
        color: #1f2937 !important;
    }
    
    /* Fix radio button text */
    .stRadio label {
        color: #1f2937 !important;
    }
    .stRadio [data-baseweb="radio"] {
        color: #1f2937 !important;
    }
    
    /* Fix all form labels */
    label {
        color: #1f2937 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #1f2937 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1f2937 !important;
    }
    
    /* Main content headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    
    /* Waiting message */
    .waiting-message {
        text-align: center;
        padding: 4rem 2rem;
        color: #6b7280;
        font-size: 1.2rem;
    }
    
    /* Expander fix */
    .streamlit-expanderHeader {
        color: #1f2937 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_generated' not in st.session_state:
    st.session_state.analysis_generated = False
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None


@st.cache_resource
def load_engine():
    """Load and cache the CrossSectorImpactEngine"""
    engine = CrossSectorImpactEngine()
    engine.load_all_data()
    return engine


def create_network_graph_from_engine(engine, source_activity, min_intensity='M', 
                                      timescale='All', direct_indirect='All', 
                                      spatial='All', comp_svc_rating='All',
                                      svc_act_intensity='M', highlight_rank=1):
    """
    Create network graph using the same logic as engine.create_flow_visualization()
    but returning the figure instead of displaying it.
    """
    import networkx as nx
    import textwrap
    
    # Get flow data from ENCORE (same as engine)
    flow_data = engine.get_single_activity_flow(
        source_activity, 
        min_intensity=min_intensity,
        timescale=timescale,
        direct_indirect=direct_indirect,
        spatial=spatial,
        comp_svc_rating=comp_svc_rating,
        svc_act_intensity=svc_act_intensity
    )
    
    if not flow_data['pressures']:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f"No environmental impact data found for:\n{source_activity}\nat intensity {min_intensity}+", 
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.5))
        ax.axis('off')
        return fig, None
    
    # Create directed graph (same logic as engine)
    G = nx.DiGraph()
    
    def wrap_text(text, width=15):
        return '\n'.join(textwrap.wrap(str(text), width=width))
    
    # Add the target activity
    activity_node = flow_data['activity']
    activity_wrapped = wrap_text(activity_node)
    G.add_node(activity_wrapped, node_type='activity', layer=0)
    
    # Add pressure nodes and connect to activity
    pressures = flow_data['pressures']
    pressure_nodes = []
    for pressure in pressures:
        pressure_wrapped = wrap_text(pressure, 18)
        G.add_node(pressure_wrapped, node_type='pressure', layer=1)
        G.add_edge(activity_wrapped, pressure_wrapped)
        pressure_nodes.append((pressure, pressure_wrapped))
    
    # Add component nodes (unique)
    component_nodes = {}
    for component in flow_data['components']:
        component_wrapped = wrap_text(component, 14)
        G.add_node(component_wrapped, node_type='component', layer=2)
        component_nodes[component] = component_wrapped
    
    # Connect pressures to their actual components
    pressure_to_components = flow_data.get('pressure_to_components', {})
    for pressure, pressure_wrapped in pressure_nodes:
        if pressure in pressure_to_components:
            for component in pressure_to_components[pressure]:
                if component in component_nodes:
                    G.add_edge(pressure_wrapped, component_nodes[component])
    
    # Add service nodes (unique)
    services = flow_data['services']
    service_nodes = {}
    for service in services:
        service_short = str(service).replace(' services', '').replace('provisioning', 'prov.')
        service_wrapped = wrap_text(service_short, 12)
        G.add_node(service_wrapped, node_type='service', layer=3)
        service_nodes[service] = service_wrapped
    
    # Connect components to their actual services
    component_to_services = flow_data.get('component_to_services', {})
    for component, component_wrapped in component_nodes.items():
        if component in component_to_services:
            for service in component_to_services[component]:
                if service in service_nodes:
                    G.add_edge(component_wrapped, service_nodes[service])
    
    # Add affected activity nodes (unique)
    affected = flow_data['affected_activities']
    affected_nodes = {}
    activity_scores = flow_data.get('activity_scores', {})
    for affected_activity in affected:
        affected_wrapped = wrap_text(affected_activity, 25)
        G.add_node(affected_wrapped, node_type='affected_activity', layer=4)
        affected_nodes[affected_activity] = affected_wrapped
    
    # Connect services to their actual dependent activities
    service_to_activities = flow_data.get('service_to_activities', {})
    service_activity_weights = flow_data.get('service_activity_weights', {})
    edge_weights = {}
    
    for service, service_wrapped in service_nodes.items():
        if service in service_to_activities:
            for activity in service_to_activities[service]:
                if activity in affected_nodes:
                    edge = (service_wrapped, affected_nodes[activity])
                    G.add_edge(*edge)
                    weight = service_activity_weights.get((service, activity), 1)
                    edge_weights[edge] = weight
    
    # Create layered layout
    fig_width = max(20, len(pressures) * 2)
    fig_height = max(14, len(component_nodes) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    pos = {}
    layers = {
        0: [activity_wrapped],
        1: [p[1] for p in pressure_nodes],
        2: list(component_nodes.values()),
        3: list(service_nodes.values()),
        4: list(affected_nodes.values())
    }
    
    # Position nodes in vertical layers
    y_positions = [5, 4, 3, 2, 1]
    for layer, y_pos in enumerate(y_positions):
        if layer in layers:
            nodes_in_layer = layers[layer]
            if nodes_in_layer:
                x_spacing = (fig_width - 4) / max(len(nodes_in_layer), 1)
                x_start = -(fig_width - 4) / 2 + x_spacing / 2
                
                for i, node in enumerate(nodes_in_layer):
                    if node in G.nodes():
                        pos[node] = (x_start + i * x_spacing, y_pos)
    
    # Define colors for each node type (same as engine)
    node_colors = {
        'activity': '#e74c3c',      # Red
        'pressure': '#f39c12',       # Orange
        'component': '#2ecc71',      # Green
        'service': '#3498db',        # Blue
        'affected_activity': '#e91e63'  # Pink
    }
    
    # Find activities to highlight based on score
    tied_activities = []
    highlight_colors = ['#c0392b', '#8e44ad', '#16a085', '#d35400', '#2980b9']
    
    if activity_scores and highlight_rank > 0:
        sorted_by_score = sorted(activity_scores.items(), key=lambda x: x[1], reverse=True)
        if highlight_rank <= len(sorted_by_score):
            target_score = sorted_by_score[highlight_rank - 1][1]
            color_idx = 0
            for act_name, score in sorted_by_score:
                if score == target_score:
                    node = affected_nodes.get(act_name)
                    if node:
                        tied_activities.append((act_name, node, score, highlight_colors[color_idx % len(highlight_colors)]))
                        color_idx += 1
    
    # Find all edges in paths to highlighted activities
    highlight_edges = {}
    highlight_nodes = {}
    
    for act_name, act_node, count, color in tied_activities:
        highlight_nodes[act_node] = color
        
        # Find services that connect to this activity
        connected_services = set()
        for service, activities in service_to_activities.items():
            if act_name in activities and service in service_nodes:
                connected_services.add(service)
                edge = (service_nodes[service], act_node)
                if edge not in highlight_edges:
                    highlight_edges[edge] = color
                if service_nodes[service] not in highlight_nodes:
                    highlight_nodes[service_nodes[service]] = color
        
        # Find components that connect to those services
        connected_components = set()
        for component, svcs in component_to_services.items():
            for svc in svcs:
                if svc in connected_services and component in component_nodes:
                    connected_components.add(component)
                    edge = (component_nodes[component], service_nodes[svc])
                    if edge not in highlight_edges:
                        highlight_edges[edge] = color
                    if component_nodes[component] not in highlight_nodes:
                        highlight_nodes[component_nodes[component]] = color
        
        # Find pressures that connect to those components
        connected_pressures = set()
        for pressure, components in pressure_to_components.items():
            for comp in components:
                if comp in connected_components:
                    pressure_wrapped = dict(pressure_nodes).get(pressure)
                    if pressure_wrapped and comp in component_nodes:
                        connected_pressures.add(pressure)
                        edge = (pressure_wrapped, component_nodes[comp])
                        if edge not in highlight_edges:
                            highlight_edges[edge] = color
                        if pressure_wrapped not in highlight_nodes:
                            highlight_nodes[pressure_wrapped] = color
        
        # Add edges from source activity to connected pressures
        for pressure, pressure_wrapped in pressure_nodes:
            if pressure in connected_pressures:
                edge = (activity_wrapped, pressure_wrapped)
                if edge not in highlight_edges:
                    highlight_edges[edge] = color
                if activity_wrapped not in highlight_nodes:
                    highlight_nodes[activity_wrapped] = color
    
    # Draw nodes by type
    for node_type, color in node_colors.items():
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == node_type]
        if nodes:
            normal_nodes = [n for n in nodes if n not in highlight_nodes]
            highlighted = [n for n in nodes if n in highlight_nodes]
            
            if node_type == 'affected_activity':
                node_size = 3000
            elif node_type in ['pressure', 'component']:
                node_size = 2500
            else:
                node_size = 2000
            
            if highlight_rank == 0:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, 
                                      node_size=node_size, alpha=0.85, ax=ax)
            else:
                if normal_nodes:
                    nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color=color, 
                                          node_size=node_size, alpha=0.4, ax=ax)
                for h_node in highlighted:
                    h_color = highlight_nodes.get(h_node, '#c0392b')
                    nx.draw_networkx_nodes(G, pos, nodelist=[h_node], node_color=color, 
                                          node_size=node_size * 1.2, alpha=0.95,
                                          edgecolors=h_color, linewidths=3, ax=ax)
    
    # Draw edges
    if highlight_rank == 0:
        non_weighted_edges = [e for e in G.edges() if e not in edge_weights]
        if non_weighted_edges:
            nx.draw_networkx_edges(G, pos, edgelist=non_weighted_edges, edge_color='#7f8c8d', 
                                  alpha=0.7, arrows=True, arrowsize=15, arrowstyle='->', width=1.5, ax=ax)
        
        for edge, weight in edge_weights.items():
            if edge in G.edges():
                width = 0.5 + (weight - 2) * 0.875
                colors_map = {2: '#95a5a6', 3: '#7f8c8d', 4: '#e91e63', 5: '#c2185b', 6: '#880e4f'}
                edge_color = colors_map.get(weight, '#7f8c8d')
                nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=edge_color, 
                                      alpha=0.8, arrows=True, arrowsize=12 + weight*2, 
                                      arrowstyle='->', width=width, ax=ax)
    else:
        non_highlight_edges = [e for e in G.edges() if e not in highlight_edges]
        if non_highlight_edges:
            non_weighted = [e for e in non_highlight_edges if e not in edge_weights]
            if non_weighted:
                nx.draw_networkx_edges(G, pos, edgelist=non_weighted, edge_color='#bdc3c7', 
                                      alpha=0.3, arrows=True, arrowsize=12, arrowstyle='->', width=1, ax=ax)
        
        edges_by_color = {}
        for edge, edge_color in highlight_edges.items():
            if edge in G.edges():
                if edge_color not in edges_by_color:
                    edges_by_color[edge_color] = []
                edges_by_color[edge_color].append(edge)
        
        for edge_color, edge_list in edges_by_color.items():
            for edge in edge_list:
                if edge in edge_weights:
                    weight = edge_weights[edge]
                    width = 1 + (weight - 1) * 1.0
                    nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=edge_color, 
                                          alpha=0.9, arrows=True, arrowsize=14 + weight*2, 
                                          arrowstyle='->', width=width, ax=ax)
                else:
                    nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=edge_color, 
                                          alpha=0.9, arrows=True, arrowsize=18, arrowstyle='->', width=3, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', font_color='black', ax=ax)
    
    # Draw scores below affected activity nodes
    for act_name, act_node in affected_nodes.items():
        if act_node in pos:
            x, y = pos[act_node]
            score = activity_scores.get(act_name, 0)
            ax.text(x, y - 0.25, f"{score}", ha='center', va='top', 
                    fontsize=12, fontweight='bold', color='#880e4f',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                             edgecolor='#e91e63', alpha=0.9))
    
    # Add legend
    legend_labels = {
        'activity': 'Source Activity',
        'pressure': 'Environmental Pressures', 
        'component': 'Ecosystem Components',
        'service': 'Ecosystem Services',
        'affected_activity': 'Affected Activities'
    }
    
    legend_elements = [plt.scatter([], [], c=color, s=150, label=legend_labels[node_type], alpha=0.85) 
                      for node_type, color in node_colors.items()]
    
    from matplotlib.lines import Line2D
    if tied_activities:
        if len(tied_activities) == 1:
            act_name, _, score, color = tied_activities[0]
            short_name = act_name[:25] + '...' if len(act_name) > 25 else act_name
            rank_label = f"#{highlight_rank}" if highlight_rank > 1 else "Most"
            legend_elements.append(Line2D([0], [0], color=color, linewidth=3, 
                                         label=f'→ {rank_label} affected: {short_name} (score: {score})'))
        else:
            score = tied_activities[0][2]
            rank_label = f"#{highlight_rank}" if highlight_rank > 1 else "Most"
            legend_elements.append(Line2D([0], [0], color='gray', linewidth=1, linestyle='--',
                                         label=f'→ {rank_label} affected ({len(tied_activities)}-way tie, score: {score}):'))
            for act_name, _, _, color in tied_activities:
                short_name = act_name[:22] + '...' if len(act_name) > 22 else act_name
                legend_elements.append(Line2D([0], [0], color=color, linewidth=3, 
                                             label=f'    • {short_name}'))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize=9)
    
    ax.axis('off')
    plt.tight_layout()
    
    return fig, flow_data


def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Portfolio Environmental Impact Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze portfolio holdings using the ENCORE environmental impact framework</p>', unsafe_allow_html=True)
    
    # Load engine
    with st.spinner("Loading ENCORE data..."):
        engine = load_engine()
    
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Analysis Settings")
        
        # Portfolio selection
        st.subheader("Portfolio Selection")
        portfolio_options = {
            "STOXX Europe 600 Oil & Gas": {
                "path": "portfolio_exports/STOXX_Europe_600_Oil_Gas_Holdings.csv",
                "type": "oil_gas"
            },
            "STOXX Europe 600 Food & Beverage": {
                "path": "portfolio_exports/STOXX-Europe-600-Food--Beverage-UCITS-ETF-DE_fund.csv",
                "type": "food_beverage"
            }
        }
        
        selected_portfolio = st.selectbox(
            "Select ETF",
            options=list(portfolio_options.keys()),
            index=0,
            key="portfolio_select"
        )
        
        portfolio_config = portfolio_options[selected_portfolio]
        
        st.divider()
        
        # ENCORE Filters Section
        st.subheader("ENCORE Filters")
        
        # Intensity filters
        intensity_options = ['VL', 'L', 'M', 'H', 'VH']
        
        pressure_intensity = st.select_slider(
            "Pressure Intensity (min)",
            options=intensity_options,
            value='M',
            help="Minimum intensity for Activity → Pressure links (VL=Very Low to VH=Very High)"
        )
        
        dependency_intensity = st.select_slider(
            "Dependency Intensity (min)", 
            options=intensity_options,
            value='M',
            help="Minimum intensity for Service → Activity dependencies"
        )
        
        st.divider()
        
        # Additional ENCORE filters
        st.subheader("Advanced Filters")
        
        # Timescale filter
        timescale = st.selectbox(
            "Timescale",
            options=['All', 'Short term', 'Mid term', 'Long term'],
            index=0,
            help="Filter by when impacts occur"
        )
        
        # Impact type filter
        direct_indirect = st.selectbox(
            "Impact Type",
            options=['All', 'Direct', 'Indirect', 'Both'],
            index=0,
            help="Direct impacts vs cascading/indirect effects"
        )
        
        # Spatial filter
        spatial = st.selectbox(
            "Spatial Scale",
            options=['All', 'Local', 'Regional', 'Global'],
            index=0,
            help="Geographic scale of impacts"
        )
        
        # Sensitivity rating filter
        rating = st.selectbox(
            "Sensitivity Rating",
            options=['All', 'R', 'A', 'G'],
            format_func=lambda x: {'All': 'All', 'R': 'Red (High)', 'A': 'Amber (Medium)', 'G': 'Green (Low)'}[x],
            index=0,
            help="Ecosystem component sensitivity level"
        )
        
        st.divider()
        
        # Group by selection
        st.subheader("Chart Options")
        group_by = st.radio(
            "Group contributions by:",
            options=['Holdings', 'ISIC Category'],
            index=0,
            help="How to stack bar chart segments"
        )
        group_by_value = 'holding' if group_by == 'Holdings' else 'isic'
        
        # Highlight rank for network
        highlight_rank = st.slider(
            "Highlight Rank (Network)",
            min_value=0,
            max_value=5,
            value=1,
            help="Which affected activity to highlight (0=none, 1=most affected)"
        )
        
        # Generate button
        st.divider()
        generate_clicked = st.button("🔄 Generate Analysis", type="primary", use_container_width=True)
        
        if generate_clicked:
            st.session_state.analysis_generated = True
    
    # Main content area
    if generate_clicked:
        with st.spinner("Generating impact analysis..."):
            try:
                plt.close('all')
                
                # Generate affected activities chart
                fig_affected, result_affected = engine.plot_portfolio_impact_stacked(
                    csv_path=portfolio_config['path'],
                    preset='Baseline',
                    top_n=15,
                    figsize=(14, 8),
                    min_contribution_pct=2.0,
                    group_by=group_by_value,
                    portfolio_type=portfolio_config['type'],
                    intensity=pressure_intensity,
                    svc_act_intensity=dependency_intensity,
                    timescale=timescale,
                    direct_indirect=direct_indirect,
                    spatial=spatial,
                    rating=rating
                )
                plt.close(fig_affected)
                
                # Generate source activities chart
                fig_source, result_source = engine.plot_portfolio_source_impact_stacked(
                    csv_path=portfolio_config['path'],
                    preset='Baseline',
                    top_n=10,
                    figsize=(14, 8),
                    min_contribution_pct=2.0,
                    portfolio_type=portfolio_config['type'],
                    intensity=pressure_intensity,
                    svc_act_intensity=dependency_intensity,
                    timescale=timescale,
                    direct_indirect=direct_indirect,
                    spatial=spatial,
                    rating=rating
                )
                plt.close(fig_source)
                
                # Get list of source activities for network graph
                source_activities = list(result_source.get('source_activities', {}).keys())
                if not source_activities:
                    source_activities = engine.activities[:10]
                
                # Store results
                st.session_state.last_analysis = {
                    'fig_affected': fig_affected,
                    'result_affected': result_affected,
                    'fig_source': fig_source,
                    'result_source': result_source,
                    'portfolio': selected_portfolio,
                    'portfolio_config': portfolio_config,
                    'source_activities': source_activities,
                    'pressure_intensity': pressure_intensity,
                    'dependency_intensity': dependency_intensity,
                    'timescale': timescale,
                    'direct_indirect': direct_indirect,
                    'spatial': spatial,
                    'rating': rating,
                    'highlight_rank': highlight_rank,
                    'group_by_value': group_by_value
                }
                
            except Exception as e:
                st.error(f"Error generating analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # Display results or waiting message
    if not st.session_state.analysis_generated or st.session_state.last_analysis is None:
        st.markdown("""
        <div class="waiting-message">
            <span style="font-size: 4rem;">📊</span>
            <h3 style="color: #1f2937;">Ready to Analyze</h3>
            <p style="color: #6b7280;">Select your portfolio and adjust filters in the sidebar, then click <strong>Generate Analysis</strong>.</p>
            <br>
            <p style="font-size: 0.9rem; color: #9ca3af;">
                <strong>Filter Options:</strong><br>
                • <strong>Intensity</strong>: Pressure/dependency strength (VL→VH)<br>
                • <strong>Timescale</strong>: Short/Mid/Long term impacts<br>
                • <strong>Impact Type</strong>: Direct vs Indirect effects<br>
                • <strong>Spatial</strong>: Local/Regional/Global scale<br>
                • <strong>Sensitivity</strong>: Ecosystem component sensitivity
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        analysis = st.session_state.last_analysis
        
        # Summary metrics
        st.subheader(f"📈 {analysis['portfolio']} Summary")
        
        summary = analysis['result_affected']['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="Holdings", value=summary['total_holdings'])
        
        with col2:
            st.metric(label="Total Weight", value=f"{summary['total_weight']:.1f}%")
        
        with col3:
            st.metric(label="Impact Score", value=f"{summary['total_impact_score']:.1f}")
        
        with col4:
            filter_str = f"Int: {analysis['pressure_intensity']}+"
            st.metric(label="Filters Applied", value=filter_str)
        
        # Show active filters
        active_filters = []
        if analysis['timescale'] != 'All':
            active_filters.append(f"Timescale: {analysis['timescale']}")
        if analysis['direct_indirect'] != 'All':
            active_filters.append(f"Impact: {analysis['direct_indirect']}")
        if analysis['spatial'] != 'All':
            active_filters.append(f"Spatial: {analysis['spatial']}")
        if analysis['rating'] != 'All':
            rating_map = {'R': 'High', 'A': 'Medium', 'G': 'Low'}
            active_filters.append(f"Sensitivity: {rating_map.get(analysis['rating'], analysis['rating'])}")
        
        if active_filters:
            st.caption(f"Active filters: {' | '.join(active_filters)}")
        
        st.divider()
        
        # Charts in 3 tabs
        tab1, tab2, tab3 = st.tabs(["🔗 Network Graph", "📊 Affected Industries", "🏭 Source Industries"])
        
        with tab1:
            st.subheader("Environmental Impact Flow Network")
            st.caption("Shows how environmental pressures cascade through ecosystems to affect other industries (same as notebook visualization)")
            
            if analysis['source_activities']:
                selected_activity = st.selectbox(
                    "Select Source Activity to Visualize",
                    options=analysis['source_activities'],
                    index=0,
                    key="network_activity_select"
                )
                
                with st.spinner("Generating network graph..."):
                    fig_network, flow_data = create_network_graph_from_engine(
                        engine, 
                        selected_activity,
                        min_intensity=analysis['pressure_intensity'],
                        timescale=analysis['timescale'],
                        direct_indirect=analysis['direct_indirect'],
                        spatial=analysis['spatial'],
                        comp_svc_rating=analysis['rating'],
                        svc_act_intensity=analysis['dependency_intensity'],
                        highlight_rank=analysis['highlight_rank']
                    )
                    st.pyplot(fig_network)
                    plt.close(fig_network)
                    
                    if flow_data:
                        st.divider()
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Pressures", len(flow_data.get('pressures', [])))
                        with col2:
                            st.metric("Components", len(flow_data.get('components', [])))
                        with col3:
                            st.metric("Services", len(flow_data.get('services', [])))
                        with col4:
                            st.metric("Affected", len(flow_data.get('affected_activities', [])))
            else:
                st.info("No source activities available for network visualization.")
        
        with tab2:
            st.subheader("Affected Economic Activities")
            st.caption("Industries impacted by the portfolio's environmental footprint")
            st.pyplot(analysis['fig_affected'])
            
            if 'top_activities' in analysis['result_affected']:
                st.divider()
                st.subheader("📋 Top Affected Activities")
                top_activities = analysis['result_affected']['top_activities']
                
                activity_data = []
                for activity, total, contributions in top_activities[:10]:
                    top_contrib = max(contributions.items(), key=lambda x: x[1]) if contributions else ('N/A', 0)
                    activity_data.append({
                        'Activity': activity,
                        'Total Score': round(total, 2),
                        'Top Contributor': top_contrib[0],
                        'Contribution': round(top_contrib[1], 2)
                    })
                
                if activity_data:
                    df = pd.DataFrame(activity_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
        
        with tab3:
            st.subheader("Source ISIC Industries")
            st.caption("ISIC sectors the portfolio maps to - showing which industries CAUSE the impact")
            st.pyplot(analysis['fig_source'])
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
        Built with ENCORE data framework • Sustainable Finance Research
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
