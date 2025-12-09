"""
ENCORE Data Infrastructure
Loads and provides access to all ENCORE environmental database files
"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import textwrap
import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path

# Import preset configurations from central config file
from configuration.presets_config import (
    get_preset_names, 
    get_preset_params, 
    get_presets_as_param_dict,
    get_all_presets,
    get_default_preset_name,
    get_preset_categories,
    get_presets_by_category,
    get_category_descriptions
)


class CrossSectorImpactEngine:
    def __init__(self, data_dir="ENCORE_data/ENCORE files"):
        self.data_dir = Path(data_dir)
        self.data = {}
        self.activities = []
        
    def load_all_data(self):
        """Load all ENCORE CSV files"""
        file_mapping = {
            'ecosystem_services_def': "02. Ecosystem services definitions.csv",
            'dependency_links': "03. Dependency links.csv", 
            'pressure_definitions': "04. Pressure definitions.csv",
            'pressure_links': "05. Pressure links.csv",
            'dependency_ratings': "06. Dependency mat ratings.csv",
            'pressure_ratings': "07. Pressure mat ratings.csv",
            'ecosystem_components_def': "10. Ecosystem components definitions.csv",
            'services_components': "11. Ecosystem services and ecosystem components.csv",
            'mechanisms_def': "12. Mechanisms of change in state definitions.csv",
            'pressure_components': "13. Pressures to components.csv",
            'isic_crosswalk': "14. EXIOBASE NACE ISIC crosswalk.csv",
            'value_chain_notes': "15. Note on value chain links.csv",
            'upstream_links': "16. Upstream links.csv",
            'downstream_links': "17. Downstream links.csv",
            'explanatory_notes': "18. Explanatory notes.csv"
        }
        
        for key, filename in file_mapping.items():
            try:
                self.data[key] = pd.read_csv(self.data_dir / filename)
                print(f"✅ Loaded {key}: {self.data[key].shape}")
            except Exception as e:
                print(f"❌ Failed to load {key}: {e}")
        
        # Extract activities from main data
        if 'pressure_links' in self.data:
            sections = self.data['pressure_links']['ISIC Section'].dropna().unique()
            self.activities = [s for s in sections if len(s) > 5]
        
        return self.data
    
    def get_dataset_info(self):
        """Get summary information about all loaded datasets"""
        info = {}
        for key, df in self.data.items():
            info[key] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
            }
        return info
    
    def get_activities(self, level='Section'):
        """Get unique economic activities at specified ISIC level"""
        if 'pressure_links' not in self.data:
            return []
        
        column_map = {
            'Section': 'ISIC Section',
            'Division': 'ISIC Division', 
            'Group': 'ISIC Group',
            'Class': 'ISIC Class'
        }
        
        if level in column_map:
            return self.data['pressure_links'][column_map[level]].dropna().unique().tolist()
        return []
    
    def explore_dataset(self, dataset_name, head=5):
        """Explore a specific dataset"""
        if dataset_name not in self.data:
            print(f"Dataset '{dataset_name}' not found")
            return None
            
        df = self.data[dataset_name]
        print(f"\n📊 Dataset: {dataset_name}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst {head} rows:")
        return df.head(head)
    
    def get_activity_links(self, link_type='upstream', limit=50):
        """Get activity linkage data for visualization"""
        if link_type == 'upstream' and 'upstream_links' in self.data:
            df = self.data['upstream_links'].head(limit)
            return df[['Direct operations (ISIC Section)', 'Upstream tier 1 (ISIC Division/Group/Class)']].dropna()
        elif link_type == 'downstream' and 'downstream_links' in self.data:
            df = self.data['downstream_links'].head(limit)
            return df[['Direct operations (ISIC Section)', 'Downstream tier 1 (ISIC Division/Group/Class)']].dropna()
        return None
    
    def get_single_activity_flow(self, activity_name, min_intensity='H', 
                                   timescale='All', direct_indirect='All', spatial='All',
                                   comp_svc_rating='All', ecosystem_type='All',
                                   svc_act_intensity='M'):
        """Get comprehensive flow data for a single economic activity including all sub-activities
        
        Returns actual 1-to-many relationships between:
        - Pressures → Ecosystem Components (from ENCORE pressure_components)
        - Ecosystem Components → Ecosystem Services (from ENCORE services_components)
        - Ecosystem Services → Affected Activities (from ENCORE dependency_ratings)
        
        Filters:
        - min_intensity: Minimum intensity for Activity → Pressure connection
        - timescale: Filter for Pressure → Component (Short term, Mid term, Long term, All)
        - direct_indirect: Filter for Pressure → Component (Direct, Indirect, All)
        - spatial: Filter for Pressure → Component (Local, Regional, Global, All)
        - comp_svc_rating: Filter for Component → Service (R=Red/High, A=Amber/Medium, G=Green/Low, All)
        - ecosystem_type: Filter for Component → Service by ecosystem type
        - svc_act_intensity: Minimum dependency intensity for Service → Activity connection
        """
        flow_data = {
            'activity': activity_name,
            'pressures': [],
            'pressure_to_components': {},  # Maps each pressure to its affected components
            'components': [],
            'component_to_services': {},   # Maps each component to its ecosystem services
            'services': [],
            'service_to_activities': {},   # Maps each service to activities that depend on it
            'affected_activities': []
        }
        
        # Define intensity hierarchy (ENCORE dependency scoring: 2-6 scale)
        intensity_levels = {'VL': 2, 'L': 3, 'M': 4, 'H': 5, 'VH': 6}
        min_level = intensity_levels.get(min_intensity, 5)
        
        # Get all pressures created by this activity (across all sub-activities)
        if 'pressure_ratings' in self.data:
            activity_data = self.data['pressure_ratings'][
                self.data['pressure_ratings']['ISIC Section'].str.contains(activity_name, na=False, case=False)
            ]
            
            if not activity_data.empty:
                pressure_cols = [col for col in self.data['pressure_ratings'].columns 
                               if col not in ['ISIC Unique code', 'ISIC Section', 'ISIC Division', 
                                            'ISIC Group', 'ISIC Class', 'ISIC level used for analysis']]
                
                # Collect all significant pressures across all sub-activities
                pressure_set = set()
                for _, row in activity_data.iterrows():
                    for col in pressure_cols:
                        rating = str(row[col]).strip()
                        if rating in intensity_levels and intensity_levels[rating] >= min_level:
                            pressure_set.add(col)
                
                flow_data['pressures'] = list(pressure_set)
        
        # Build pressure-to-components mapping from the actual ENCORE data
        # Apply Timescale, Direct/Indirect, and Spatial filters here
        if 'pressure_components' in self.data and flow_data['pressures']:
            pc_df = self.data['pressure_components'].copy()
            pressure_col = 'Pressures(/Impact drivers)'
            component_col = 'Ecosystem component'
            
            # Apply Timescale filter
            if timescale != 'All':
                pc_df = pc_df[pc_df['Timescale'].str.lower().str.contains(timescale.lower(), na=False)]
            
            # Apply Direct vs Indirect filter
            if direct_indirect != 'All':
                if direct_indirect == 'Direct':
                    pc_df = pc_df[pc_df['Direct vs. indirect'].str.lower().str.contains('direct', na=False) & 
                                  ~pc_df['Direct vs. indirect'].str.lower().str.contains('indirect', na=False)]
                elif direct_indirect == 'Indirect':
                    pc_df = pc_df[pc_df['Direct vs. indirect'].str.lower().str.contains('indirect', na=False)]
                elif direct_indirect == 'Both':
                    pc_df = pc_df[pc_df['Direct vs. indirect'].str.lower().str.contains('and', na=False)]
            
            # Apply Spatial filter
            if spatial != 'All':
                pc_df = pc_df[pc_df['Spatial characteristics'].str.lower().str.contains(spatial.lower(), na=False)]
            
            all_components = set()
            for pressure in flow_data['pressures']:
                # Find matching pressures in the pressure_components dataset
                matching_rows = pc_df[pc_df[pressure_col].str.contains(pressure.split()[0], na=False, case=False, regex=False)]
                
                if matching_rows.empty:
                    # Try more flexible matching
                    pressure_keywords = pressure.lower().replace('emissions of ', '').replace('area of ', '').split()
                    for keyword in pressure_keywords:
                        if len(keyword) > 3:  # Skip short words
                            matching_rows = pc_df[pc_df[pressure_col].str.lower().str.contains(keyword, na=False, regex=False)]
                            if not matching_rows.empty:
                                break
                
                components = matching_rows[component_col].dropna().unique().tolist()
                if components:
                    flow_data['pressure_to_components'][pressure] = components
                    all_components.update(components)
            
            flow_data['components'] = list(all_components)
        
        # Build component-to-services mapping from the actual ENCORE data
        # Apply Rating and Ecosystem Type filters here
        if 'services_components' in self.data and flow_data['components']:
            sc_df = self.data['services_components'].copy()
            service_col = 'Ecosystem services'
            component_col = 'Ecosystem components'
            
            # Apply Rating filter (R=Red/High sensitivity, A=Amber/Medium, G=Green/Low)
            if comp_svc_rating != 'All':
                sc_df = sc_df[sc_df['Rating'] == comp_svc_rating]
            
            # Apply Ecosystem Type filter
            if ecosystem_type != 'All':
                sc_df = sc_df[sc_df['Ecosystem types'].str.lower().str.contains(ecosystem_type.lower(), na=False)]
            
            all_services = set()
            for component in flow_data['components']:
                # Find services linked to this component (exact match on component name)
                matching_rows = sc_df[sc_df[component_col].str.strip().str.lower() == component.strip().lower()]
                
                services = matching_rows[service_col].dropna().unique().tolist()
                if services:
                    flow_data['component_to_services'][component] = services
                    all_services.update(services)
            
            flow_data['services'] = list(all_services)
        
        # Build service-to-activities mapping from the actual ENCORE dependency_ratings data
        # Now also stores intensity weights for each edge
        if 'dependency_ratings' in self.data and flow_data['services']:
            dep_df = self.data['dependency_ratings']
            # ENCORE dependency scoring: 2-6 scale
            intensity_levels = {'VL': 2, 'L': 3, 'M': 4, 'H': 5, 'VH': 6}
            min_svc_act_level = intensity_levels.get(svc_act_intensity, 4)  # Default to M
            
            # ISIC columns to exclude when looking for service columns
            isic_cols = ['ISIC Unique code', 'ISIC Section', 'ISIC Division', 
                        'ISIC Group', 'ISIC Class', 'ISIC level used for analysis ']
            
            all_affected = set()
            # Store edge weights: {(service, activity): intensity_score}
            flow_data['service_activity_weights'] = {}
            
            for service in flow_data['services']:
                # Find the column that matches this service (may need flexible matching)
                matching_col = None
                for col in dep_df.columns:
                    if col not in isic_cols:
                        # Try exact match first
                        if col.strip().lower() == service.strip().lower():
                            matching_col = col
                            break
                        # Try partial match
                        if service.lower() in col.lower() or col.lower() in service.lower():
                            matching_col = col
                            break
                
                if matching_col:
                    # Find activities that depend on this service (filtered by dependency intensity)
                    dependent_activities = []
                    for _, row in dep_df.iterrows():
                        rating = str(row.get(matching_col, '')).strip()
                        if rating in intensity_levels and intensity_levels[rating] >= min_svc_act_level:
                            section = row.get('ISIC Section')
                            if pd.notna(section):
                                act_name = str(section).strip()
                                # Exclude the source activity itself
                                if act_name != activity_name:
                                    if act_name not in dependent_activities:
                                        dependent_activities.append(act_name)
                                    # Store the weight for this edge (use max if multiple sub-activities)
                                    edge_key = (service, act_name)
                                    current_weight = flow_data['service_activity_weights'].get(edge_key, 0)
                                    flow_data['service_activity_weights'][edge_key] = max(current_weight, intensity_levels[rating])
                    
                    if dependent_activities:
                        flow_data['service_to_activities'][service] = dependent_activities
                        all_affected.update(dependent_activities)
            
            flow_data['affected_activities'] = list(all_affected)
            
            # Calculate total weighted score for each affected activity
            flow_data['activity_scores'] = {}
            for (service, activity), weight in flow_data['service_activity_weights'].items():
                flow_data['activity_scores'][activity] = flow_data['activity_scores'].get(activity, 0) + weight
        
        return flow_data
    
    def wrap_text(self, text, width=15):
        """Wrap long text to multiple lines"""
        return '\n'.join(textwrap.wrap(text, width=width))
    
    def create_flow_visualization(self, target_activity, min_intensity='M',
                                    timescale='All', direct_indirect='All', spatial='All',
                                    comp_svc_rating='All', ecosystem_type='All',
                                    svc_act_intensity='M', highlight_rank=1):
        """Create and display the flow visualization for a given activity
        
        Shows actual 1-to-many relationships between pressures and ecosystem components.
        
        Args:
            highlight_rank: Which most-affected activity to highlight (1=most, 2=2nd, etc., 0=none)
        """
        
        # Get specific flow data for the target activity with filters
        flow_data = self.get_single_activity_flow(target_activity, min_intensity,
                                                   timescale, direct_indirect, spatial,
                                                   comp_svc_rating, ecosystem_type,
                                                   svc_act_intensity)

        if flow_data['pressures']:
            # Create directed graph
            G = nx.DiGraph()
            
            # Add the target activity
            activity_node = flow_data['activity']
            activity_wrapped = self.wrap_text(activity_node)
            G.add_node(activity_wrapped, node_type='activity', layer=0)
            
            # Add pressure nodes and connect to activity
            pressures = flow_data['pressures']
            pressure_nodes = []
            for pressure in pressures:
                pressure_wrapped = self.wrap_text(pressure, 18)
                G.add_node(pressure_wrapped, node_type='pressure', layer=1)
                G.add_edge(activity_wrapped, pressure_wrapped)
                pressure_nodes.append((pressure, pressure_wrapped))
            
            # Add component nodes (unique)
            component_nodes = {}
            for component in flow_data['components']:
                component_wrapped = self.wrap_text(component, 14)
                G.add_node(component_wrapped, node_type='component', layer=2)
                component_nodes[component] = component_wrapped
            
            # Connect pressures to their actual components (1-to-many)
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
                service_wrapped = self.wrap_text(service_short, 12)
                G.add_node(service_wrapped, node_type='service', layer=3)
                service_nodes[service] = service_wrapped
            
            # Connect components to their actual services (1-to-many)
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
                affected_wrapped = self.wrap_text(affected_activity, 25)
                G.add_node(affected_wrapped, node_type='affected_activity', layer=4)
                affected_nodes[affected_activity] = affected_wrapped
            
            # Connect services to their actual dependent activities (1-to-many)
            # Store edge weights for visualization
            service_to_activities = flow_data.get('service_to_activities', {})
            service_activity_weights = flow_data.get('service_activity_weights', {})
            edge_weights = {}  # (source_node, target_node) -> weight
            
            for service, service_wrapped in service_nodes.items():
                if service in service_to_activities:
                    for activity in service_to_activities[service]:
                        if activity in affected_nodes:
                            edge = (service_wrapped, affected_nodes[activity])
                            G.add_edge(*edge)
                            # Get the weight for this edge
                            weight = service_activity_weights.get((service, activity), 1)
                            edge_weights[edge] = weight
            
            # Create layered layout
            fig_width = max(20, len(pressures) * 2)
            fig_height = max(14, len(component_nodes) * 1.5)
            plt.figure(figsize=(fig_width, fig_height))
            
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
            
            # Define colors for each node type
            node_colors = {
                'activity': '#e74c3c',      # Red
                'pressure': '#f39c12',       # Orange
                'component': '#2ecc71',      # Green
                'service': '#3498db',        # Blue
                'affected_activity': '#e91e63'  # Pink
            }
            
            # Find the most affected activity based on WEIGHTED SCORE (not edge count)
            # Sort activities by score (descending) and find ties at the selected rank
            tied_activities = []  # List of (name, node, score, color)
            highlight_colors = ['#c0392b', '#8e44ad', '#16a085', '#d35400', '#2980b9']  # Red, Purple, Teal, Orange, Blue
            
            if activity_scores and highlight_rank > 0:
                sorted_by_score = sorted(activity_scores.items(), key=lambda x: x[1], reverse=True)
                if highlight_rank <= len(sorted_by_score):
                    # Get the score at the selected rank
                    target_score = sorted_by_score[highlight_rank - 1][1]
                    
                    # Find all activities with the same score (ties)
                    color_idx = 0
                    for act_name, score in sorted_by_score:
                        if score == target_score:
                            node = affected_nodes.get(act_name)
                            if node:
                                tied_activities.append((act_name, node, score, highlight_colors[color_idx % len(highlight_colors)]))
                                color_idx += 1
            
            # Find all edges in the paths to all tied activities
            highlight_edges = {}  # edge -> color
            highlight_nodes = {}  # node -> color
            
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
                for component, services in component_to_services.items():
                    for svc in services:
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
            
            # Draw nodes by type (non-highlighted first, then highlighted)
            for node_type, color in node_colors.items():
                nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == node_type]
                if nodes:
                    # Separate highlighted and non-highlighted nodes
                    normal_nodes = [n for n in nodes if n not in highlight_nodes]
                    highlighted = [n for n in nodes if n in highlight_nodes]
                    
                    # Different sizes for different node types
                    if node_type == 'affected_activity':
                        node_size = 3000  # Larger for longer activity names
                    elif node_type in ['pressure', 'component']:
                        node_size = 2500
                    else:
                        node_size = 2000
                    
                    # If no highlighting, draw all nodes at full visibility
                    if highlight_rank == 0:
                        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, 
                                              node_size=node_size, alpha=0.85)
                    else:
                        # Draw normal nodes with lower alpha
                        if normal_nodes:
                            nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color=color, 
                                                  node_size=node_size, alpha=0.4)
                        
                        # Draw highlighted nodes with full alpha and colored edge based on their highlight color
                        for h_node in highlighted:
                            h_color = highlight_nodes.get(h_node, '#c0392b')
                            nx.draw_networkx_nodes(G, pos, nodelist=[h_node], node_color=color, 
                                                  node_size=node_size * 1.2, alpha=0.95,
                                                  edgecolors=h_color, linewidths=3)
            
            # Draw edges - full visibility when no highlighting, faded otherwise
            # Service→Activity edges are drawn with weighted widths
            if highlight_rank == 0:
                # No highlighting - draw non-weighted edges first
                non_weighted_edges = [e for e in G.edges() if e not in edge_weights]
                if non_weighted_edges:
                    nx.draw_networkx_edges(G, pos, edgelist=non_weighted_edges, edge_color='#7f8c8d', 
                                          alpha=0.7, arrows=True, arrowsize=15, arrowstyle='->', width=1.5)
                
                # Draw weighted service→activity edges with scaled widths
                for edge, weight in edge_weights.items():
                    if edge in G.edges():
                        # Scale width: VL(2)=0.5, L(3)=1, M(4)=1.5, H(5)=2.5, VH(6)=4
                        width = 0.5 + (weight - 2) * 0.875
                        # Color intensity based on weight (ENCORE 2-6 scale)
                        colors = {2: '#95a5a6', 3: '#7f8c8d', 4: '#e91e63', 5: '#c2185b', 6: '#880e4f'}
                        edge_color = colors.get(weight, '#7f8c8d')
                        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=edge_color, 
                                              alpha=0.8, arrows=True, arrowsize=12 + weight*2, 
                                              arrowstyle='->', width=width)
            else:
                # Draw non-highlighted edges first (faded)
                non_highlight_edges = [e for e in G.edges() if e not in highlight_edges]
                if non_highlight_edges:
                    # Separate weighted and non-weighted
                    non_weighted = [e for e in non_highlight_edges if e not in edge_weights]
                    weighted = [e for e in non_highlight_edges if e in edge_weights]
                    
                    if non_weighted:
                        nx.draw_networkx_edges(G, pos, edgelist=non_weighted, edge_color='#bdc3c7', 
                                              alpha=0.3, arrows=True, arrowsize=12, arrowstyle='->', width=1)
                    
                    # Draw weighted edges with scaled widths but faded
                    for edge in weighted:
                        weight = edge_weights[edge]
                        width = 0.5 + (weight - 1) * 0.875
                        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color='#bdc3c7', 
                                              alpha=0.25, arrows=True, arrowsize=10 + weight, 
                                              arrowstyle='->', width=width)
                
                # Draw highlighted edges grouped by color
                edges_by_color = {}
                for edge, edge_color in highlight_edges.items():
                    if edge in G.edges():
                        if edge_color not in edges_by_color:
                            edges_by_color[edge_color] = []
                        edges_by_color[edge_color].append(edge)
                
                for edge_color, edge_list in edges_by_color.items():
                    # Check if these are weighted edges
                    for edge in edge_list:
                        if edge in edge_weights:
                            weight = edge_weights[edge]
                            width = 1 + (weight - 1) * 1.0  # Slightly thicker when highlighted
                            nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=edge_color, 
                                                  alpha=0.9, arrows=True, arrowsize=14 + weight*2, 
                                                  arrowstyle='->', width=width)
                        else:
                            nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=edge_color, 
                                                  alpha=0.9, arrows=True, arrowsize=18, arrowstyle='->', width=3)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', font_color='black')
            
            # Draw scores below affected activity nodes
            for act_name, act_node in affected_nodes.items():
                if act_node in pos:
                    x, y = pos[act_node]
                    score = activity_scores.get(act_name, 0)
                    plt.text(x, y - 0.25, f"{score}", ha='center', va='top', 
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
            
            # Add highlight legend entries for tied activities
            from matplotlib.lines import Line2D
            if tied_activities:
                if len(tied_activities) == 1:
                    act_name, _, score, color = tied_activities[0]
                    short_name = act_name[:25] + '...' if len(act_name) > 25 else act_name
                    rank_label = f"#{highlight_rank}" if highlight_rank > 1 else "Most"
                    legend_elements.append(Line2D([0], [0], color=color, linewidth=3, 
                                                 label=f'→ {rank_label} affected: {short_name} (score: {score})'))
                else:
                    # Multiple tied activities
                    score = tied_activities[0][2]
                    rank_label = f"#{highlight_rank}" if highlight_rank > 1 else "Most"
                    legend_elements.append(Line2D([0], [0], color='gray', linewidth=1, linestyle='--',
                                                 label=f'→ {rank_label} affected ({len(tied_activities)}-way tie, score: {score}):'))
                    for act_name, _, _, color in tied_activities:
                        short_name = act_name[:22] + '...' if len(act_name) > 22 else act_name
                        legend_elements.append(Line2D([0], [0], color=color, linewidth=3, 
                                                     label=f'    • {short_name}'))
            
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize=9)
            
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        else:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"No environmental impact data found for:\n{target_activity}\nat intensity {min_intensity}+", 
                    ha='center', va='center', fontsize=16, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            plt.title("Data Not Available", fontsize=14, fontweight='bold')
            plt.show()
    
    def compute_flow_stats(self, flow_data):
        """Compute statistics about the flow network"""
        stats = {
            'edge_counts': {},
            'most_connected': {}
        }
        
        # Count edges between layers
        # Activity → Pressures
        stats['edge_counts']['activity_to_pressures'] = len(flow_data['pressures'])
        
        # Pressures → Components
        p2c_edges = sum(len(comps) for comps in flow_data.get('pressure_to_components', {}).values())
        stats['edge_counts']['pressures_to_components'] = p2c_edges
        
        # Components → Services
        c2s_edges = sum(len(svcs) for svcs in flow_data.get('component_to_services', {}).values())
        stats['edge_counts']['components_to_services'] = c2s_edges
        
        # Services → Activities
        s2a_edges = sum(len(acts) for acts in flow_data.get('service_to_activities', {}).values())
        stats['edge_counts']['services_to_activities'] = s2a_edges
        
        # Find most connected nodes at each layer (OUTGOING edges)
        # Most connected pressure (by outgoing edges to components)
        if flow_data.get('pressure_to_components'):
            max_pressure = max(flow_data['pressure_to_components'].items(), 
                              key=lambda x: len(x[1]), default=(None, []))
            stats['most_connected']['pressure'] = (max_pressure[0], len(max_pressure[1]))
        
        # Most connected component (by outgoing edges to services)
        if flow_data.get('component_to_services'):
            max_component = max(flow_data['component_to_services'].items(),
                               key=lambda x: len(x[1]), default=(None, []))
            stats['most_connected']['component'] = (max_component[0], len(max_component[1]))
        
        # Most connected service (by outgoing edges to activities)
        if flow_data.get('service_to_activities'):
            max_service = max(flow_data['service_to_activities'].items(),
                             key=lambda x: len(x[1]), default=(None, []))
            stats['most_connected']['service'] = (max_service[0], len(max_service[1]))
        
        # Find nodes with most INCOMING edges
        stats['most_incoming'] = {}
        
        # Component with most incoming (from pressures)
        if flow_data.get('pressure_to_components'):
            component_incoming = {}
            for pressure, components in flow_data['pressure_to_components'].items():
                for comp in components:
                    component_incoming[comp] = component_incoming.get(comp, 0) + 1
            if component_incoming:
                max_comp = max(component_incoming.items(), key=lambda x: x[1])
                stats['most_incoming']['component'] = max_comp
        
        # Service with most incoming (from components)
        if flow_data.get('component_to_services'):
            service_incoming = {}
            for component, services in flow_data['component_to_services'].items():
                for svc in services:
                    service_incoming[svc] = service_incoming.get(svc, 0) + 1
            if service_incoming:
                max_svc = max(service_incoming.items(), key=lambda x: x[1])
                stats['most_incoming']['service'] = max_svc
        
        # Activity with most incoming (from services)
        if flow_data.get('service_to_activities'):
            activity_incoming = {}
            for service, activities in flow_data['service_to_activities'].items():
                for act in activities:
                    activity_incoming[act] = activity_incoming.get(act, 0) + 1
            if activity_incoming:
                max_act = max(activity_incoming.items(), key=lambda x: x[1])
                stats['most_incoming']['affected_activity'] = max_act
        
        return stats
    
    def create_interactive_widget(self):
        """Create interactive widget for environmental impact flow visualization
        
        Organized UI with filters grouped by connection level:
        - Each row has filters on left and corresponding stats on right
        """
        
        # Stats display widgets for each row
        stats_row1 = widgets.HTML(value="")
        stats_row2 = widgets.HTML(value="")
        stats_row3 = widgets.HTML(value="")
        stats_row4 = widgets.HTML(value="")
        stats_row5 = widgets.HTML(value="")  # For Services → Activities and totals
        
        def update_stats():
            """Update the statistics panels"""
            flow_data = self.get_single_activity_flow(
                activity_dropdown.value,
                intensity_dropdown.value,
                timescale_dropdown.value,
                direct_indirect_dropdown.value,
                spatial_dropdown.value,
                rating_dropdown.value,
                'All',
                svc_act_intensity_dropdown.value
            )
            stats = self.compute_flow_stats(flow_data)
            total_edges = sum(stats['edge_counts'].values())
            
            # Row 1: Activity stats + Total edges (on right)
            stats_row1.value = f"<div style='font-size: 11px; color: #666;'><b style='color: #e74c3c;'>Total: {total_edges} edges</b></div>"
            
            # Row 2: Activity → Pressures stats (edges on right)
            edge_count = stats['edge_counts']['activity_to_pressures']
            most_out = stats['most_connected'].get('pressure', (None, 0))
            parts = []
            if most_out[0]:
                short_name = most_out[0][:18] + '...' if len(most_out[0]) > 18 else most_out[0]
                parts.append(f"Top out: <b>{short_name}</b> ({most_out[1]}→)")
            parts.append(f"Edges: <b>{edge_count}</b>")
            stats_row2.value = f"<div style='font-size: 11px; color: #666;'>{' | '.join(parts)}</div>"
            
            # Row 3: Pressures → Components stats (edges on right)
            edge_count = stats['edge_counts']['pressures_to_components']
            most_out = stats['most_connected'].get('component', (None, 0))
            most_in = stats['most_incoming'].get('component', (None, 0))
            parts = []
            if most_in[0]:
                parts.append(f"Top in: <b>{most_in[0]}</b> (→{most_in[1]})")
            if most_out[0]:
                parts.append(f"Top out: <b>{most_out[0]}</b> ({most_out[1]}→)")
            parts.append(f"Edges: <b>{edge_count}</b>")
            stats_row3.value = f"<div style='font-size: 11px; color: #666;'>{' | '.join(parts)}</div>"
            
            # Row 4: Components → Services stats (edges on right)
            edge_count = stats['edge_counts']['components_to_services']
            most_out = stats['most_connected'].get('service', (None, 0))
            most_in = stats['most_incoming'].get('service', (None, 0))
            parts = []
            if most_in[0]:
                short_in = most_in[0][:15] + '...' if len(most_in[0]) > 15 else most_in[0]
                parts.append(f"Top in: <b>{short_in}</b> (→{most_in[1]})")
            if most_out[0]:
                short_out = most_out[0][:15] + '...' if len(most_out[0]) > 15 else most_out[0]
                parts.append(f"Top out: <b>{short_out}</b> ({most_out[1]}→)")
            parts.append(f"Edges: <b>{edge_count}</b>")
            stats_row4.value = f"<div style='font-size: 11px; color: #666;'>{' | '.join(parts)}</div>"
            
            # Row 5: Services → Activities stats (edges on right)
            edge_count = stats['edge_counts']['services_to_activities']
            most_in = stats['most_incoming'].get('affected_activity', (None, 0))
            parts = []
            if most_in[0]:
                short_in = most_in[0][:18] + '...' if len(most_in[0]) > 18 else most_in[0]
                parts.append(f"Top in: <b>{short_in}</b> (→{most_in[1]})")
            parts.append(f"Edges: <b>{edge_count}</b>")
            stats_row5.value = f"<div style='font-size: 11px; color: #666;'>{' | '.join(parts)}</div>"
        
        def on_widget_change(change):
            """Handle widget changes"""
            update_stats()
            with output_widget:
                clear_output(wait=True)
                self.create_flow_visualization(
                    activity_dropdown.value, 
                    intensity_dropdown.value,
                    timescale_dropdown.value,
                    direct_indirect_dropdown.value,
                    spatial_dropdown.value,
                    rating_dropdown.value,
                    'All',
                    svc_act_intensity_dropdown.value,
                    highlight_dropdown.value
                )

        # Get available activities for dropdown
        available_activities = sorted(list(set([a.strip() for a in self.activities])))

        # Find a good default activity
        default_activity = available_activities[0]
        for candidate in ['Agriculture, forestry and fishing', 'Manufacturing', 'Construction']:
            if candidate in available_activities:
                default_activity = candidate
                break

        # ========== PRESETS DEFINITION ==========
        # Load presets from central config file (presets_config.jsonc)
        presets = get_presets_as_param_dict()
        
        # Preset dropdown (will be added to Row 1)
        preset_dropdown = widgets.Dropdown(
            options=list(presets.keys()),
            value=get_default_preset_name(),
            description='Preset:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='280px')
        )
        
        def on_preset_change(change):
            """Apply preset values to all dropdowns"""
            preset_name = change['new']
            if preset_name in presets:
                values = presets[preset_name]
                # Temporarily disable observers to prevent multiple updates
                intensity_dropdown.unobserve(on_widget_change, names='value')
                timescale_dropdown.unobserve(on_widget_change, names='value')
                direct_indirect_dropdown.unobserve(on_widget_change, names='value')
                spatial_dropdown.unobserve(on_widget_change, names='value')
                rating_dropdown.unobserve(on_widget_change, names='value')
                svc_act_intensity_dropdown.unobserve(on_widget_change, names='value')
                
                # Apply preset values
                intensity_dropdown.value = values[0]
                timescale_dropdown.value = values[1]
                direct_indirect_dropdown.value = values[2]
                spatial_dropdown.value = values[3]
                rating_dropdown.value = values[4]
                svc_act_intensity_dropdown.value = values[5]
                
                # Re-enable observers
                intensity_dropdown.observe(on_widget_change, names='value')
                timescale_dropdown.observe(on_widget_change, names='value')
                direct_indirect_dropdown.observe(on_widget_change, names='value')
                spatial_dropdown.observe(on_widget_change, names='value')
                rating_dropdown.observe(on_widget_change, names='value')
                svc_act_intensity_dropdown.observe(on_widget_change, names='value')
                
                # Trigger single update
                on_widget_change(None)

        # ========== ROW 1: Activity Selection + Preset + Highlight ==========
        activity_label = widgets.HTML(value="<b style='color: #e74c3c;'>📍 Source Activity</b>")
        activity_dropdown = widgets.Dropdown(
            options=available_activities,
            value=default_activity,
            description='ISIC Section:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        # Highlight dropdown - select which rank to highlight
        highlight_dropdown = widgets.Dropdown(
            options=[('None', 0), ('1st Most Affected', 1), ('2nd Most Affected', 2), 
                     ('3rd Most Affected', 3), ('4th Most Affected', 4), ('5th Most Affected', 5)],
            value=1,
            description='🔦Highlight:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='180px')
        )
        
        row1_filters = widgets.HBox([activity_label, activity_dropdown, preset_dropdown, highlight_dropdown])
        row1 = widgets.HBox([row1_filters, stats_row1], 
                           layout=widgets.Layout(margin='3px 0px', padding='5px',
                                                border='1px solid #e74c3c', border_radius='5px',
                                                justify_content='space-between'))

        # ========== ROW 2: Activity → Pressures (Intensity) ==========
        intensity_label = widgets.HTML(value="<b style='color: #f39c12;'>🔗 Activity → Pressures</b>")
        intensity_dropdown = widgets.Dropdown(
            options=[('Very Low (VL+)', 'VL'), ('Low (L+)', 'L'), ('Medium (M+)', 'M'), 
                     ('High (H+)', 'H'), ('Very High (VH)', 'VH')],
            value='M',
            description='Min Intensity:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        row2_filters = widgets.HBox([intensity_label, intensity_dropdown])
        row2 = widgets.HBox([row2_filters, stats_row2],
                           layout=widgets.Layout(margin='3px 0px', padding='5px',
                                                border='1px solid #f39c12', border_radius='5px',
                                                justify_content='space-between'))

        # ========== ROW 3: Pressures → Components (Timescale, Direct/Indirect, Spatial) ==========
        pressure_comp_label = widgets.HTML(value="<b style='color: #2ecc71;'>🔗 Pressures → Components</b>")
        
        timescale_dropdown = widgets.Dropdown(
            options=[('All Timescales', 'All'), ('Short term', 'Short'), 
                     ('Mid term', 'Mid'), ('Long term', 'Long')],
            value='All',
            description='Timescale:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='170px')
        )
        
        direct_indirect_dropdown = widgets.Dropdown(
            options=[('All Types', 'All'), ('Direct only', 'Direct'), 
                     ('Indirect only', 'Indirect'), ('Both', 'Both')],
            value='All',
            description='Impact:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='160px')
        )
        
        spatial_dropdown = widgets.Dropdown(
            options=[('All Spatial', 'All'), ('Local', 'Local'), 
                     ('Regional', 'Regional'), ('Global', 'Global')],
            value='All',
            description='Spatial:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='150px')
        )
        
        row3_filters = widgets.HBox([pressure_comp_label, timescale_dropdown, 
                                     direct_indirect_dropdown, spatial_dropdown])
        row3 = widgets.HBox([row3_filters, stats_row3],
                           layout=widgets.Layout(margin='3px 0px', padding='5px',
                                                border='1px solid #2ecc71', border_radius='5px',
                                                justify_content='space-between'))

        # ========== ROW 4: Components → Services (Rating, Ecosystem Type) ==========
        comp_svc_label = widgets.HTML(value="<b style='color: #3498db;'>🔗 Components → Services</b>")
        
        rating_dropdown = widgets.Dropdown(
            options=[('All Ratings', 'All'), ('Red (High)', 'R'), 
                     ('Amber (Medium)', 'A'), ('Green (Low)', 'G')],
            value='All',
            description='Sensitivity:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='180px')
        )
        
        row4_filters = widgets.HBox([comp_svc_label, rating_dropdown])
        row4 = widgets.HBox([row4_filters, stats_row4],
                           layout=widgets.Layout(margin='3px 0px', padding='5px',
                                                border='1px solid #3498db', border_radius='5px',
                                                justify_content='space-between'))

        # ========== ROW 5: Services → Affected Activities (Dependency Intensity) ==========
        svc_act_label = widgets.HTML(value="<b style='color: #e91e63;'>🔗 Services → Affected Activities</b>")
        
        svc_act_intensity_dropdown = widgets.Dropdown(
            options=[('Very Low (VL+)', 'VL'), ('Low (L+)', 'L'), ('Medium (M+)', 'M'), 
                     ('High (H+)', 'H'), ('Very High (VH)', 'VH')],
            value='M',
            description='Dependency:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        row5_filters = widgets.HBox([svc_act_label, svc_act_intensity_dropdown])
        row5 = widgets.HBox([row5_filters, stats_row5],
                           layout=widgets.Layout(margin='3px 0px', padding='5px',
                                                border='1px solid #e91e63', border_radius='5px',
                                                justify_content='space-between'))

        # Create output widget for the visualization
        output_widget = widgets.Output()

        # Observe changes in all dropdowns
        activity_dropdown.observe(on_widget_change, names='value')
        intensity_dropdown.observe(on_widget_change, names='value')
        timescale_dropdown.observe(on_widget_change, names='value')
        direct_indirect_dropdown.observe(on_widget_change, names='value')
        spatial_dropdown.observe(on_widget_change, names='value')
        rating_dropdown.observe(on_widget_change, names='value')
        svc_act_intensity_dropdown.observe(on_widget_change, names='value')
        highlight_dropdown.observe(on_widget_change, names='value')
        preset_dropdown.observe(on_preset_change, names='value')

        # Stack all rows vertically
        controls = widgets.VBox([row1, row2, row3, row4, row5],
                               layout=widgets.Layout(width='100%'))
        
        display(controls)
        display(output_widget)

        # Initialize stats and visualization
        update_stats()
        with output_widget:
            self.create_flow_visualization(
                activity_dropdown.value, 
                intensity_dropdown.value,
                timescale_dropdown.value,
                direct_indirect_dropdown.value,
                spatial_dropdown.value,
                rating_dropdown.value,
                'All',
                svc_act_intensity_dropdown.value,
                highlight_dropdown.value
            )
        
        return {
            'activity': activity_dropdown,
            'intensity': intensity_dropdown,
            'timescale': timescale_dropdown,
            'direct_indirect': direct_indirect_dropdown,
            'spatial': spatial_dropdown,
            'rating': rating_dropdown,
            'svc_act_intensity': svc_act_intensity_dropdown,
            'highlight': highlight_dropdown,
            'preset': preset_dropdown,
            'output': output_widget
        }
    
    def visualize_activity_impact(self, activity_name, min_intensity='M'):
        """Simple method to create a standalone visualization for an activity"""
        print(f"Analyzing environmental impact flow for: {activity_name}")
        print(f"Minimum intensity level: {min_intensity}+")
        print()
        
        self.create_flow_visualization(activity_name, min_intensity)
        
        # Also print summary statistics
        flow_data = self.get_single_activity_flow(activity_name, min_intensity)
        print(f"\nImpact Summary:")
        print(f"  Environmental Pressures: {len(flow_data['pressures'])}")
        print(f"  Affected Ecosystem Components: {len(flow_data['components'])}")
        print(f"  Affected Ecosystem Services: {len(flow_data['services'])}")
        print(f"  Other Affected Activities: {len(flow_data['affected_activities'])}")
        
        # Show pressure-to-component mapping details
        if flow_data.get('pressure_to_components'):
            print(f"\nPressure → Component Mappings:")
            for pressure, components in flow_data['pressure_to_components'].items():
                print(f"  {pressure}:")
                for comp in components:
                    print(f"    → {comp}")
        
        # Show component-to-services mapping details
        if flow_data.get('component_to_services'):
            print(f"\nComponent → Service Mappings:")
            for component, services in flow_data['component_to_services'].items():
                print(f"  {component} ({len(services)} services):")
                for svc in services[:5]:
                    print(f"    → {svc}")
                if len(services) > 5:
                    print(f"    ... and {len(services) - 5} more")
        
        # Show service-to-activities mapping details
        if flow_data.get('service_to_activities'):
            print(f"\nService → Affected Activities Mappings:")
            for service, activities in flow_data['service_to_activities'].items():
                print(f"  {service} ({len(activities)} activities):")
                for act in activities[:4]:
                    print(f"    → {act[:40]}")
                if len(activities) > 4:
                    print(f"    ... and {len(activities) - 4} more")
        
        return flow_data
    
    def get_affected_activities_stats(self, source_activity, preset=None,
                                       intensity='M', timescale='All', 
                                       direct_indirect='All', spatial='All',
                                       rating='All', svc_act_intensity='M',
                                       return_dataframe=True):
        """
        Get statistics about affected activities for a given source activity.
        
        Args:
            source_activity: The source ISIC Section activity name
            preset: Optional preset name from presets_config.jsonc
                   If provided, overrides individual filter parameters
            intensity: Min intensity for Activity → Pressures (VL/L/M/H/VH)
            timescale: Timescale filter for Pressures → Components (All/Short/Mid/Long)
            direct_indirect: Impact type filter (All/Direct/Indirect/Both)
            spatial: Spatial scale filter (All/Local/Regional/Global)
            rating: Sensitivity rating for Components → Services (All/R/A/G)
            svc_act_intensity: Min dependency intensity for Services → Activities (VL/L/M/H/VH)
            return_dataframe: If True, returns pandas DataFrame; if False, returns dict
            
        Returns:
            DataFrame or dict with affected activities and their incoming edge counts
        """
        import pandas as pd
        
        # Load presets from config file
        presets = get_presets_as_param_dict()
        
        # Apply preset if provided
        if preset and preset in presets:
            intensity, timescale, direct_indirect, spatial, rating, svc_act_intensity = presets[preset]
        
        # Get flow data
        flow_data = self.get_single_activity_flow(
            source_activity, intensity, timescale, direct_indirect, 
            spatial, rating, 'All', svc_act_intensity
        )
        
        # Count incoming edges for each affected activity
        activity_incoming_count = {}
        service_to_activities = flow_data.get('service_to_activities', {})
        
        for service, activities in service_to_activities.items():
            for act in activities:
                activity_incoming_count[act] = activity_incoming_count.get(act, 0) + 1
        
        # Get weighted scores
        activity_scores = flow_data.get('activity_scores', {})
        
        # Sort by weighted score (descending), then by edge count
        sorted_activities = sorted(
            [(act, count, activity_scores.get(act, 0)) for act, count in activity_incoming_count.items()],
            key=lambda x: (x[2], x[1]), reverse=True
        )
        
        # Build result
        result = {
            'source_activity': source_activity,
            'filters': {
                'intensity': intensity,
                'timescale': timescale,
                'direct_indirect': direct_indirect,
                'spatial': spatial,
                'rating': rating,
                'svc_act_intensity': svc_act_intensity
            },
            'summary': {
                'total_pressures': len(flow_data['pressures']),
                'total_components': len(flow_data['components']),
                'total_services': len(flow_data['services']),
                'total_affected_activities': len(sorted_activities),
                'total_service_activity_edges': sum(activity_incoming_count.values()),
                'total_weighted_score': sum(activity_scores.values())
            },
            'affected_activities': sorted_activities
        }
        
        if return_dataframe:
            # Create DataFrame with weighted scores
            df = pd.DataFrame(sorted_activities, columns=['Activity', 'Incoming_Edges', 'Weighted_Score'])
            df['Rank'] = range(1, len(df) + 1)
            df['Edge_Percentage'] = (df['Incoming_Edges'] / df['Incoming_Edges'].sum() * 100).round(2)
            df['Score_Percentage'] = (df['Weighted_Score'] / df['Weighted_Score'].sum() * 100).round(2)
            df = df[['Rank', 'Activity', 'Incoming_Edges', 'Weighted_Score', 'Edge_Percentage', 'Score_Percentage']]
            
            # Print summary
            print(f"═══════════════════════════════════════════════════════════════")
            print(f"📊 CROSS-SECTOR IMPACT ANALYSIS")
            print(f"═══════════════════════════════════════════════════════════════")
            print(f"Source Activity: {source_activity}")
            print(f"Preset: {preset if preset else 'Custom'}")
            print(f"───────────────────────────────────────────────────────────────")
            print(f"Filters Applied:")
            print(f"  • Activity → Pressures intensity: {intensity}+")
            print(f"  • Pressures → Components timescale: {timescale}")
            print(f"  • Pressures → Components impact type: {direct_indirect}")
            print(f"  • Pressures → Components spatial: {spatial}")
            print(f"  • Components → Services sensitivity: {rating}")
            print(f"  • Services → Activities dependency: {svc_act_intensity}+")
            print(f"───────────────────────────────────────────────────────────────")
            print(f"Flow Summary:")
            print(f"  • Environmental Pressures: {result['summary']['total_pressures']}")
            print(f"  • Ecosystem Components: {result['summary']['total_components']}")
            print(f"  • Ecosystem Services: {result['summary']['total_services']}")
            print(f"  • Affected Activities: {result['summary']['total_affected_activities']}")
            print(f"  • Total Service→Activity Edges: {result['summary']['total_service_activity_edges']}")
            print(f"  • Total Weighted Score: {result['summary']['total_weighted_score']}")
            print(f"───────────────────────────────────────────────────────────────")
            print(f"Weight Scale (ENCORE): VL=2, L=3, M=4, H=5, VH=6")
            print(f"═══════════════════════════════════════════════════════════════")
            print()
            
            return df
        else:
            return result
    
    def compare_source_activities(self, activities=None, preset=None, top_n=10):
        """
        Compare multiple source activities and show which affected activities 
        are most impacted across all sources.
        
        Args:
            activities: List of source activity names. If None, uses all available activities.
            preset: Preset to use for filters (from presets_config.jsonc). If None, uses default.
            top_n: Number of top affected activities to show per source
            
        Returns:
            DataFrame with comparison results
        """
        import pandas as pd
        
        # Use default preset if none specified
        if preset is None:
            preset = get_default_preset_name()
        
        if activities is None:
            activities = self.activities
        
        all_results = []
        
        for source in activities:
            result = self.get_affected_activities_stats(
                source, preset=preset, return_dataframe=False
            )
            
            for affected, count, score in result['affected_activities']:
                all_results.append({
                    'Source_Activity': source,
                    'Affected_Activity': affected,
                    'Incoming_Edges': count,
                    'Weighted_Score': score
                })
        
        df = pd.DataFrame(all_results)
        
        if df.empty:
            print("No data found.")
            return df
        
        # Aggregate: for each affected activity, sum edges and scores from all sources
        agg_df = df.groupby('Affected_Activity').agg({
            'Incoming_Edges': 'sum',
            'Weighted_Score': 'sum',
            'Source_Activity': 'count'
        }).reset_index()
        agg_df.columns = ['Activity', 'Total_Edges', 'Total_Score', 'Affected_By_N_Sources']
        agg_df = agg_df.sort_values('Total_Score', ascending=False)
        agg_df['Rank'] = range(1, len(agg_df) + 1)
        agg_df['Score_Percentage'] = (agg_df['Total_Score'] / agg_df['Total_Score'].sum() * 100).round(2)
        agg_df = agg_df[['Rank', 'Activity', 'Total_Edges', 'Total_Score', 'Score_Percentage', 'Affected_By_N_Sources']]
        
        print(f"═══════════════════════════════════════════════════════════════")
        print(f"📊 CROSS-SECTOR IMPACT COMPARISON")
        print(f"═══════════════════════════════════════════════════════════════")
        print(f"Source Activities Analyzed: {len(activities)}")
        print(f"Preset: {preset}")
        print(f"Total Weighted Score (all sources): {agg_df['Total_Score'].sum()}")
        print(f"───────────────────────────────────────────────────────────────")
        print(f"Most Affected Activities (ranked by Total Score):")
        print(f"Weight Scale (ENCORE): VL=2, L=3, M=4, H=5, VH=6")
        print(f"═══════════════════════════════════════════════════════════════")
        print()
        
        return agg_df
    
    def plot_affected_activities_by_preset(self, activities=None, presets=None, top_n=15, figsize=(18, 12)):
        """
        Create subplots comparing affected activities rankings across different presets.
        
        Args:
            activities: List of source activities. If None, uses all available.
            presets: List of preset names to compare. If None, uses all presets from config.
            top_n: Number of top affected activities to show per subplot
            figsize: Figure size tuple (width, height)
            
        Returns:
            matplotlib figure object
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Load available presets from config
        all_presets = get_preset_names()
        
        if presets is None:
            presets = all_presets
        
        if activities is None:
            activities = self.activities
        
        # First pass: collect all data to find global max score range for consistent coloring
        preset_data = {}
        global_max_score = 0
        global_min_score = float('inf')
        
        for preset in presets:
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                df = self.compare_source_activities(activities=activities, preset=preset)
            finally:
                sys.stdout = old_stdout
            
            if not df.empty:
                top_df = df.head(top_n).copy()
                preset_data[preset] = top_df
                if len(top_df) > 0:
                    max_score = top_df['Total_Score'].max()
                    min_score = top_df['Total_Score'].min()
                    if max_score > global_max_score:
                        global_max_score = max_score
                    if min_score < global_min_score:
                        global_min_score = min_score
            else:
                preset_data[preset] = pd.DataFrame()
        
        # Calculate grid dimensions
        n_presets = len(presets)
        n_cols = min(3, n_presets)
        n_rows = (n_presets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_presets == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_presets > 1 else [axes]
        
        for idx, preset in enumerate(presets):
            ax = axes[idx]
            
            top_df = preset_data.get(preset, pd.DataFrame())
            
            if top_df.empty:
                ax.text(0.5, 0.5, f'No data for\n{preset}', ha='center', va='center', fontsize=12)
                ax.set_title(f'{preset}', fontsize=14, fontweight='bold')
                continue
            
            # Shorten activity names for display
            top_df['Short_Name'] = top_df['Activity'].apply(
                lambda x: x[:30] + '...' if len(x) > 30 else x
            )
            
            # Calculate colors based on global score range (normalized across all subplots)
            # Red (0.1) for highest scores, Green (0.9) for lowest scores
            score_range = global_max_score - global_min_score if global_max_score > global_min_score else 1
            bar_colors = []
            for score in top_df['Total_Score']:
                # Normalize: high score -> 0.1 (red), low score -> 0.9 (green)
                normalized = (score - global_min_score) / score_range
                color_val = 0.1 + (1 - normalized) * 0.8  # Inverted: high score = red
                bar_colors.append(plt.cm.RdYlGn(color_val))
            
            # Create horizontal bar chart
            y_pos = range(len(top_df))
            bars = ax.barh(y_pos, top_df['Total_Score'], color=bar_colors, edgecolor='black', linewidth=0.5)
            
            # Add value labels INSIDE bars (only if score > 0)
            for i, (bar, score) in enumerate(zip(bars, top_df['Total_Score'])):
                if score > 0:
                    # Position text inside bar, near the right edge
                    text_x = bar.get_width() * 0.95  # 95% of bar width
                    ax.text(text_x, bar.get_y() + bar.get_height()/2, 
                           f'{int(score)}', va='center', ha='right', fontsize=9, 
                           color='black', fontweight='bold')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_df['Short_Name'], fontsize=9)
            ax.invert_yaxis()  # Highest at top
            ax.set_xlabel('Total Weighted Score', fontsize=10)
            ax.set_title(f'{preset}', fontsize=12, fontweight='bold', pad=10)
            
            # Add grid
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Add total score annotation
            total = top_df['Total_Score'].sum()
            ax.annotate(f'Top {len(top_df)} Total: {int(total)}', 
                       xy=(0.98, 0.02), xycoords='axes fraction',
                       ha='right', va='bottom', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for idx in range(len(presets), len(axes)):
            axes[idx].set_visible(False)
        
        # Add main title
        fig.suptitle('Cross-Sector Impact Analysis: Most Affected Activities by Preset\n'
                    f'(Aggregated across {len(activities)} source activities, Top {top_n} shown)',
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def get_source_activity_impact(self, source_activity, presets=None):
        """
        Calculate the total impact a single source activity causes across all presets.
        
        Args:
            source_activity: The source activity to analyze
            presets: List of preset names. If None, uses all presets from config.
            
        Returns:
            Dictionary with total score per preset and overall total
        """
        # Load presets from config file
        preset_configs = get_presets_as_param_dict()
        all_presets = get_preset_names()
        
        if presets is None:
            presets = all_presets
        
        impact_data = {'Activity': source_activity, 'preset_scores': {}, 'Total_Impact': 0}
        
        for preset in presets:
            if preset not in preset_configs:
                impact_data['preset_scores'][preset] = 0
                continue
                
            # Get filter values for this preset
            intensity, timescale, direct_indirect, spatial, rating, svc_act_intensity = preset_configs[preset]
            
            # Suppress print output
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                flow_data = self.get_single_activity_flow(
                    source_activity,
                    min_intensity=intensity,
                    timescale=timescale,
                    direct_indirect=direct_indirect,
                    spatial=spatial,
                    comp_svc_rating=rating,
                    svc_act_intensity=svc_act_intensity
                )
            finally:
                sys.stdout = old_stdout
            
            if flow_data and 'activity_scores' in flow_data:
                preset_total = sum(flow_data['activity_scores'].values())
                impact_data['preset_scores'][preset] = preset_total
                impact_data['Total_Impact'] += preset_total
            else:
                impact_data['preset_scores'][preset] = 0
        
        return impact_data
    
    def compare_source_activity_impacts(self, activities=None, presets=None):
        """
        Compare the total impact caused by each source activity across all presets.
        
        Args:
            activities: List of source activities. If None, uses all available.
            presets: List of preset names. If None, uses all presets.
            
        Returns:
            DataFrame with source activities ranked by total impact caused
        """
        import pandas as pd
        
        # Load presets from config
        all_presets = get_preset_names()
        
        if presets is None:
            presets = all_presets
            
        if activities is None:
            activities = self.activities
        
        # Collect impact data for each source activity
        impact_results = []
        
        for activity in activities:
            impact_data = self.get_source_activity_impact(activity, presets)
            
            row = {'Source_Activity': activity}
            for preset in presets:
                row[preset] = impact_data['preset_scores'].get(preset, 0)
            row['Total_Impact'] = impact_data['Total_Impact']
            
            impact_results.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(impact_results)
        
        # Sort by Total_Impact descending
        df = df.sort_values('Total_Impact', ascending=False).reset_index(drop=True)
        
        # Add rank column
        df.insert(0, 'Rank', range(1, len(df) + 1))
        
        # Calculate percentage of total
        total_all = df['Total_Impact'].sum()
        if total_all > 0:
            df['Impact_Percentage'] = (df['Total_Impact'] / total_all * 100).round(2)
        else:
            df['Impact_Percentage'] = 0
        
        # Print summary
        print(f"═══════════════════════════════════════════════════════════════")
        print(f"🎯 SOURCE ACTIVITY IMPACT ANALYSIS")
        print(f"═══════════════════════════════════════════════════════════════")
        print(f"Source Activities Analyzed: {len(activities)}")
        print(f"Presets Included: {', '.join(presets)}")
        print(f"Total Impact Score (all sources, all presets): {int(total_all)}")
        print(f"───────────────────────────────────────────────────────────────")
        print(f"Most Impactful Source Activities (ranked by Total Impact caused):")
        print(f"Weight Scale (ENCORE): VL=2, L=3, M=4, H=5, VH=6")
        print(f"═══════════════════════════════════════════════════════════════")
        print()
        
        return df
    
    def plot_source_activity_impacts(self, activities=None, presets=None, top_n=15, figsize=(18, 12)):
        """
        Create subplots comparing source activity impacts across different presets.
        Same format as plot_affected_activities_by_preset.
        
        Args:
            activities: List of source activities. If None, uses all available.
            presets: List of preset names. If None, uses all presets from config.
            top_n: Number of top source activities to show per subplot
            figsize: Figure size tuple (width, height)
            
        Returns:
            matplotlib figure object and DataFrame
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Load presets from config
        all_presets = get_preset_names()
        
        if presets is None:
            presets = all_presets
        
        if activities is None:
            activities = self.activities
        
        # First pass: collect all data to find global max score range for consistent coloring
        all_preset_data = {}
        global_max_score = 0
        global_min_score = float('inf')
        
        for preset in presets:
            # Get impact data for this preset
            impact_results = []
            
            for activity in activities:
                # Suppress print output
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                
                try:
                    impact_data = self.get_source_activity_impact(activity, presets=[preset])
                finally:
                    sys.stdout = old_stdout
                
                preset_score = impact_data['preset_scores'].get(preset, 0)
                impact_results.append({'Source_Activity': activity, 'Impact_Score': preset_score})
            
            # Create DataFrame and sort
            df = pd.DataFrame(impact_results)
            df = df.sort_values('Impact_Score', ascending=False).reset_index(drop=True)
            df.insert(0, 'Rank', range(1, len(df) + 1))
            
            top_df = df.head(top_n).copy()
            all_preset_data[preset] = top_df
            
            if len(top_df) > 0 and top_df['Impact_Score'].sum() > 0:
                max_score = top_df['Impact_Score'].max()
                min_score = top_df['Impact_Score'].min()
                if max_score > global_max_score:
                    global_max_score = max_score
                if min_score < global_min_score:
                    global_min_score = min_score
        
        # Calculate grid dimensions
        n_presets = len(presets)
        n_cols = min(3, n_presets)
        n_rows = (n_presets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_presets == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_presets > 1 else [axes]
        
        for idx, preset in enumerate(presets):
            ax = axes[idx]
            
            top_df = all_preset_data.get(preset, pd.DataFrame())
            
            if top_df.empty or top_df['Impact_Score'].sum() == 0:
                ax.text(0.5, 0.5, f'No data for\n{preset}', ha='center', va='center', fontsize=12)
                ax.set_title(f'{preset}', fontsize=14, fontweight='bold')
                continue
            
            # Shorten activity names for display
            top_df['Short_Name'] = top_df['Source_Activity'].apply(
                lambda x: x[:30] + '...' if len(x) > 30 else x
            )
            
            # Calculate colors based on global score range (normalized across all subplots)
            # Red (0.1) for highest scores, Green (0.9) for lowest scores
            score_range = global_max_score - global_min_score if global_max_score > global_min_score else 1
            bar_colors = []
            for score in top_df['Impact_Score']:
                # Normalize: high score -> 0.1 (red), low score -> 0.9 (green)
                normalized = (score - global_min_score) / score_range
                color_val = 0.1 + (1 - normalized) * 0.8  # Inverted: high score = red
                bar_colors.append(plt.cm.RdYlGn(color_val))
            
            # Create horizontal bar chart
            y_pos = range(len(top_df))
            bars = ax.barh(y_pos, top_df['Impact_Score'], color=bar_colors, edgecolor='black', linewidth=0.5)
            
            # Add value labels INSIDE bars (only if score > 0)
            for i, (bar, score) in enumerate(zip(bars, top_df['Impact_Score'])):
                if score > 0:
                    # Position text inside bar, near the right edge
                    text_x = bar.get_width() * 0.95  # 95% of bar width
                    ax.text(text_x, bar.get_y() + bar.get_height()/2, 
                           f'{int(score)}', va='center', ha='right', fontsize=9,
                           color='black', fontweight='bold')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_df['Short_Name'], fontsize=9)
            ax.invert_yaxis()  # Highest at top
            ax.set_xlabel('Impact Score', fontsize=10)
            ax.set_title(f'{preset}', fontsize=12, fontweight='bold', pad=10)
            
            # Add grid
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Add total score annotation
            total = top_df['Impact_Score'].sum()
            ax.annotate(f'Top {len(top_df)} Total: {int(total)}', 
                       xy=(0.98, 0.02), xycoords='axes fraction',
                       ha='right', va='bottom', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for idx in range(len(presets), len(axes)):
            axes[idx].set_visible(False)
        
        # Add main title
        fig.suptitle('Source Activity Impact Analysis: Most Impactful Activities by Preset\n'
                    f'(Which activities CAUSE the most environmental impact, Top {top_n} shown)',
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.show()
        
        # Also get the combined comparison data
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            combined_df = self.compare_source_activity_impacts(activities=activities, presets=presets)
        finally:
            sys.stdout = old_stdout
        
        # Print summary
        print(f"\n{'═'*65}")
        print(f"SOURCE ACTIVITY IMPACT SUMMARY")
        print(f"{'═'*65}")
        print(f"Total Impact (all sources, all presets): {int(combined_df['Total_Impact'].sum())}")
        print(f"Top 3 Most Impactful Source Activities:")
        for _, row in combined_df.head(3).iterrows():
            print(f"  {row['Rank']}. {row['Source_Activity'][:40]}: {int(row['Total_Impact'])} ({row['Impact_Percentage']:.1f}%)")
        print(f"{'═'*65}")
        
        return fig, combined_df
    
    def plot_cumulative_affected_activities(self, activities=None, presets=None, top_n=15, figsize=(14, 10)):
        """
        Create a stacked horizontal bar graph showing contributions from each preset per affected activity.
        
        Args:
            activities: List of source activities. If None, uses all available.
            presets: List of preset names. If None, uses all presets from config.
            top_n: Number of top affected activities to show
            figsize: Figure size tuple (width, height)
            
        Returns:
            matplotlib figure object and DataFrame
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Load presets from config
        all_presets = get_preset_names()
        
        if presets is None:
            presets = all_presets
        
        if activities is None:
            activities = self.activities
        
        # Collect cumulative data for each affected activity across all presets
        cumulative_data = {}
        
        for preset in presets:
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                df = self.compare_source_activities(activities=activities, preset=preset)
            finally:
                sys.stdout = old_stdout
            
            if not df.empty:
                for _, row in df.iterrows():
                    activity = row['Activity']
                    score = row['Total_Score']
                    if activity not in cumulative_data:
                        cumulative_data[activity] = {'total': 0, 'by_preset': {}}
                    cumulative_data[activity]['total'] += score
                    cumulative_data[activity]['by_preset'][preset] = score
        
        # Convert to DataFrame
        result_data = []
        for activity, data in cumulative_data.items():
            row = {'Activity': activity, 'Cumulative_Score': data['total']}
            for preset in presets:
                row[preset] = data['by_preset'].get(preset, 0)
            result_data.append(row)
        
        result_df = pd.DataFrame(result_data)
        result_df = result_df.sort_values('Cumulative_Score', ascending=False).reset_index(drop=True)
        result_df.insert(0, 'Rank', range(1, len(result_df) + 1))
        
        # Get top N
        top_df = result_df.head(top_n).copy()
        
        # Shorten names
        top_df['Short_Name'] = top_df['Activity'].apply(
            lambda x: x[:35] + '...' if len(x) > 35 else x
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Preset colors for stacked bars
        preset_colors = plt.cm.Set2(np.linspace(0, 1, len(presets)))
        
        # Create stacked horizontal bar chart
        y_pos = range(len(top_df))
        bottom = np.zeros(len(top_df))
        
        for i, preset in enumerate(presets):
            values = top_df[preset].values
            ax.barh(y_pos, values, left=bottom, color=preset_colors[i], 
                   label=preset, edgecolor='white', linewidth=0.5)
            bottom += values
        
        # Add total value labels at end of bars (only if > 0)
        for i, (y, total) in enumerate(zip(y_pos, top_df['Cumulative_Score'])):
            if total > 0:
                ax.text(total + 20, y, f'{int(total)}', va='center', ha='left', 
                       fontsize=10, color='black', fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_df['Short_Name'], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Cumulative Score (Stacked by Preset)', fontsize=11)
        ax.set_title(f'Most Affected Activities - Cumulative Scores Across All Presets\n'
                    f'(Top {top_n} shown, aggregated from {len(activities)} source activities)',
                    fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=9)
        
        # Add total annotation
        total = top_df['Cumulative_Score'].sum()
        ax.annotate(f'Top {len(top_df)} Total: {int(total)}', 
                   xy=(0.98, 0.02), xycoords='axes fraction',
                   ha='right', va='bottom', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig, result_df
    
    def plot_cumulative_source_impacts(self, activities=None, presets=None, top_n=15, figsize=(14, 10)):
        """
        Create a stacked horizontal bar graph showing contributions from each preset per source/impacting activity.
        
        Args:
            activities: List of source activities. If None, uses all available.
            presets: List of preset names. If None, uses all presets from config.
            top_n: Number of top source activities to show
            figsize: Figure size tuple (width, height)
            
        Returns:
            matplotlib figure object and DataFrame
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Load presets from config
        all_presets = get_preset_names()
        
        if presets is None:
            presets = all_presets
        
        # Get the combined comparison data (already has per-preset columns)
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            result_df = self.compare_source_activity_impacts(activities=activities, presets=presets)
        finally:
            sys.stdout = old_stdout
        
        if result_df.empty:
            print("No data available for source activity impact analysis")
            return None, result_df
        
        # Get top N
        top_df = result_df.head(top_n).copy()
        
        # Shorten names
        top_df['Short_Name'] = top_df['Source_Activity'].apply(
            lambda x: x[:35] + '...' if len(x) > 35 else x
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Preset colors for stacked bars
        preset_colors = plt.cm.Set2(np.linspace(0, 1, len(presets)))
        
        # Create stacked horizontal bar chart
        y_pos = range(len(top_df))
        bottom = np.zeros(len(top_df))
        
        for i, preset in enumerate(presets):
            if preset in top_df.columns:
                values = top_df[preset].values
                ax.barh(y_pos, values, left=bottom, color=preset_colors[i], 
                       label=preset, edgecolor='white', linewidth=0.5)
                bottom += values
        
        # Add total value labels at end of bars (only if > 0)
        for i, (y, total) in enumerate(zip(y_pos, top_df['Total_Impact'])):
            if total > 0:
                ax.text(total + 20, y, f'{int(total)}', va='center', ha='left', 
                       fontsize=10, color='black', fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_df['Short_Name'], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Cumulative Impact Score (Stacked by Preset)', fontsize=11)
        ax.set_title(f'Most Impactful Source Activities - Cumulative Scores Across All Presets\n'
                    f'(Top {top_n} shown, which activities CAUSE the most environmental impact)',
                    fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=9)
        
        # Add total annotation
        total = top_df['Total_Impact'].sum()
        ax.annotate(f'Top {len(top_df)} Total: {int(total)}', 
                   xy=(0.98, 0.02), xycoords='axes fraction',
                   ha='right', va='bottom', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig, result_df
    
    def _export_graph_data_to_json(self, category, graph_type, all_data, presets):
        """
        Export graph data to a JSON file in the graph_data folder.
        
        Args:
            category: The category name (e.g., 'Intensity', 'Timescale')
            graph_type: Type of graph ('affected_activities' or 'source_impacts')
            all_data: Dictionary of DataFrames per preset
            presets: List of preset names in order
        """
        import json
        import os
        from datetime import datetime
        from configuration.presets_config import get_all_presets
        
        presets_data = get_all_presets()
        
        # Build export data structure
        export_data = {
            'metadata': {
                'category': category,
                'graph_type': graph_type,
                'generated_at': datetime.now().isoformat(),
                'num_presets': len(presets)
            },
            'subplots': []
        }
        
        for preset in presets:
            preset_info = presets_data.get(preset, {})
            params = preset_info.get('params', {})
            
            df = all_data.get(preset)
            
            # Build scores list
            scores = []
            if df is not None and not df.empty:
                # Handle both affected_activities and source_impacts column names
                if 'Activity' in df.columns:
                    activity_col = 'Activity'
                    score_col = 'Total_Score'
                else:
                    activity_col = 'Source_Activity'
                    score_col = 'Impact_Score'
                
                for _, row in df.iterrows():
                    scores.append({
                        'activity': row[activity_col],
                        'score': int(row[score_col]) if row[score_col] > 0 else 0
                    })
            
            subplot_data = {
                'preset_name': preset,
                'parameters': params,
                'description': preset_info.get('description', ''),
                'scores': scores,
                'total_score': sum(s['score'] for s in scores)
            }
            export_data['subplots'].append(subplot_data)
        
        # Create filename
        filename = f"{category.lower().replace(' ', '_')}_{graph_type}.json"
        filepath = os.path.join(os.path.dirname(__file__), 'graph_data', filename)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Graph data exported to: graph_data/{filename}")
    
    def _draw_parameter_toggles(self, ax, preset, y_start=-0.12, show_labels=True, category=None):
        """
        Draw 6 parameter toggle scales below a subplot.
        
        Args:
            ax: The matplotlib axes to draw on
            preset: The preset name to get parameters for
            y_start: Starting y position in axes fraction (negative = below plot)
            show_labels: Whether to show parameter labels on the left (for all subplots now)
            category: The category name to determine which param to highlight for baseline presets
        """
        from configuration.presets_config import get_all_presets
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
        import matplotlib.colors as mcolors
        
        presets_data = get_all_presets()
        if preset not in presets_data:
            return
        
        params = presets_data[preset]['params']
        
        # Baseline parameters for comparison (to highlight changed params)
        baseline_params = {
            'intensity': 'M',
            'timescale': 'All',
            'direct_indirect': 'All',
            'spatial': 'All',
            'comp_svc_rating': 'All',
            'svc_act_intensity': 'M'
        }
        
        # Map category to the parameter it varies
        category_to_param = {
            'Intensity': 'intensity',
            'Timescale': 'timescale',
            'Impact Type': 'direct_indirect',
            'Spatial': 'spatial',
            'Sensitivity': 'comp_svc_rating',
            'Dependency': 'svc_act_intensity'
        }
        
        # Check if this preset IS the baseline (all params match baseline)
        is_baseline_preset = all(
            params.get(k) == v for k, v in baseline_params.items()
        )
        
        # Get the parameter that this category focuses on
        category_param = category_to_param.get(category) if category else None
        
        # Parameter definitions with self-explanatory names
        # Format: (param_key, display_name, options_list)
        # Options ordered from least strict (left) to most strict (right)
        param_defs = [
            ('intensity', 'Pressure Intensity', ['VL', 'L', 'M', 'H', 'VH']),
            ('timescale', 'Timescale', ['Long', 'Mid', 'Short']),
            ('direct_indirect', 'Impact Type', ['Indirect', 'Direct']),
            ('spatial', 'Spatial Scale', ['Global', 'Regional', 'Local']),
            ('comp_svc_rating', 'Sensitivity', ['G', 'A', 'R']),
            ('svc_act_intensity', 'Dependency', ['VL', 'L', 'M', 'H', 'VH']),
        ]
        
        # Parameters that are cumulative (fill all to the right of selected)
        cumulative_params = {'intensity', 'svc_act_intensity'}
        
        # Drawing parameters
        line_spacing = 0.12  # Much more vertical spacing between toggles
        circle_radius = 0.008
        frame_padding = 0.03  # Padding for the frame around each toggle
        
        # Green to Yellow to Red colormap for labels
        def get_spectrum_color(index, total):
            """Get color from green (left) to red (right) spectrum"""
            if total <= 1:
                return '#f39c12'  # Yellow for single option
            ratio = index / (total - 1)
            # Green -> Yellow -> Red
            if ratio <= 0.5:
                # Green to Yellow
                r = int(255 * (ratio * 2))
                g = 180
                b = 50
            else:
                # Yellow to Red
                r = 220
                g = int(180 * (1 - (ratio - 0.5) * 2))
                b = 50
            return f'#{r:02x}{g:02x}{b:02x}'
        
        for i, (param_key, display_name, options) in enumerate(param_defs):
            y_pos = y_start - (i * line_spacing)
            current_value = params.get(param_key, '')
            
            # Check if this parameter differs from baseline
            is_changed = params.get(param_key) != baseline_params.get(param_key)
            
            # Highlight if:
            # - This param is changed from baseline, OR
            # - This is a baseline preset AND this param is the one the category focuses on
            should_highlight = is_changed or (is_baseline_preset and param_key == category_param)
            
            # Check if "All" or "Both" is selected - fill all sockets
            fill_all = current_value in ('All', 'Both')
            
            # Find index of selected value
            selected_idx = options.index(current_value) if current_value in options else -1
            
            # Check if this is a cumulative parameter
            is_cumulative = param_key in cumulative_params
            
            # Calculate x positions for options - wider spread
            n_options = len(options)
            x_start = 0.02
            x_end = 0.98
            x_positions = [x_start + (x_end - x_start) * j / (n_options - 1) for j in range(n_options)]
            
            # Draw frame (rounded rectangle) around the toggle
            # Use blue frame if this parameter is changed from baseline OR if this is the baseline preset
            frame_height = 0.085  # Taller frame to fit labels inside
            frame_padding_h = 0.05  # Horizontal padding for wider box
            frame = FancyBboxPatch(
                (x_start - frame_padding_h, y_pos - frame_height/2),
                (x_end - x_start) + frame_padding_h * 2,
                frame_height,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                transform=ax.transAxes,
                facecolor='#e8f4fc' if should_highlight else 'white',  # Light blue bg if highlighted
                edgecolor='#3498db' if should_highlight else '#aaaaaa',  # Blue frame if highlighted
                linewidth=2.5 if should_highlight else 1.5,
                clip_on=False,
                zorder=0
            )
            ax.add_patch(frame)
            
            # Draw parameter name on the left (now for all subplots)
            # Make label blue if highlighted
            ax.text(x_start - frame_padding_h - 0.04, y_pos, display_name, 
                   transform=ax.transAxes,
                   fontsize=7, ha='right', va='center', 
                   color='#2980b9' if should_highlight else '#222222', 
                   fontweight='bold')
            
            # Draw the horizontal line
            ax.plot([x_start, x_end], [y_pos, y_pos], transform=ax.transAxes,
                   color='#cccccc', linewidth=2, clip_on=False, zorder=1)
            
            # Draw circles for each option
            for j, (x, option) in enumerate(zip(x_positions, options)):
                # Determine if this socket should be filled
                if fill_all:
                    is_filled = True
                elif is_cumulative and selected_idx >= 0:
                    is_filled = (j >= selected_idx)
                else:
                    is_filled = (option == current_value)
                
                # Get spectrum color for this option (green to red)
                spectrum_color = get_spectrum_color(j, n_options)
                
                # Use Ellipse with equal width/height for perfect circles
                # Filled = black, Empty = hollow grey
                circle = mpatches.Ellipse((x, y_pos), width=circle_radius*2, height=circle_radius*2,
                                         transform=ax.transAxes,
                                         facecolor='black' if is_filled else 'none',
                                         edgecolor='black' if is_filled else '#999999',
                                         linewidth=2 if is_filled else 1.5, 
                                         clip_on=False, zorder=2)
                ax.add_patch(circle)
                
                # Draw option label below with spectrum color
                ax.text(x, y_pos - 0.025, option, transform=ax.transAxes,
                       fontsize=6, ha='center', va='top', 
                       color=spectrum_color,
                       fontweight='bold' if is_filled else 'normal')

    def plot_affected_by_category(self, category, activities=None, top_n=15, figsize=None):
        """
        Plot most affected activities for a single category, with subplots for each preset.
        
        Args:
            category: Category name - one of 'Scope', 'Risk', 'Timeline', 'Spatial'
            activities: List of source activities. If None, uses all available.
            top_n: Number of top affected activities to show per preset
            figsize: Figure size tuple. If None, auto-calculated based on number of presets.
            
        Returns:
            tuple: (matplotlib figure, dict of DataFrames per preset)
            
        Example:
            fig, data = engine.plot_affected_by_category('Scope', top_n=15)
            fig, data = engine.plot_affected_by_category('Risk', top_n=10)
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Validate category
        valid_categories = get_preset_categories()
        if category not in valid_categories:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {valid_categories}")
        
        if activities is None:
            activities = self.activities
        
        # Get presets for this category
        presets_by_cat = get_presets_by_category()
        category_descriptions = get_category_descriptions()
        
        presets = presets_by_cat.get(category, [])
        n_presets = len(presets)
        
        if n_presets == 0:
            raise ValueError(f"No presets found for category '{category}'")
        
        # Category colors
        category_colors = {
            'Scope': '#3498db',      # Blue
            'Risk': '#e74c3c',       # Red
            'Timeline': '#2ecc71',   # Green
            'Spatial': '#9b59b6'     # Purple
        }
        cat_color = category_colors.get(category, '#7f8c8d')
        cat_desc = category_descriptions.get(category, '')
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (6 * n_presets, 8)
        
        # Create figure with subplots (1 row, n_presets columns)
        fig, axes = plt.subplots(1, n_presets, figsize=figsize)
        
        # Ensure axes is iterable
        if n_presets == 1:
            axes = [axes]
        
        # First pass: collect all data to find global max score for consistent coloring
        all_data = {}
        global_max_score = 0
        global_min_score = float('inf')
        
        for preset in presets:
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                df = self.compare_source_activities(activities=activities, preset=preset)
            finally:
                sys.stdout = old_stdout
            
            if not df.empty:
                top_df = df.head(top_n).copy()
                all_data[preset] = top_df
                if len(top_df) > 0 and top_df['Total_Score'].sum() > 0:
                    max_score = top_df['Total_Score'].max()
                    min_score = top_df[top_df['Total_Score'] > 0]['Total_Score'].min() if (top_df['Total_Score'] > 0).any() else 0
                    if max_score > global_max_score:
                        global_max_score = max_score
                    if min_score < global_min_score and min_score > 0:
                        global_min_score = min_score
        
        if global_min_score == float('inf'):
            global_min_score = 0
        
        # Plot each preset
        for col_idx, preset in enumerate(presets):
            ax = axes[col_idx]
            top_df = all_data.get(preset, pd.DataFrame())
            
            if top_df.empty or top_df['Total_Score'].sum() == 0:
                ax.text(0.5, 0.5, f'No data for\n{preset}', 
                       ha='center', va='center', fontsize=11,
                       transform=ax.transAxes)
                ax.set_title(f'{preset}', fontsize=12, fontweight='bold', color='black')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Shorten activity names
            top_df['Short_Name'] = top_df['Activity'].apply(
                lambda x: x[:35] + '...' if len(x) > 38 else x
            )
            
            # Calculate colors based on global score range
            score_range = global_max_score - global_min_score if global_max_score > global_min_score else 1
            bar_colors = []
            for score in top_df['Total_Score']:
                if score == 0:
                    bar_colors.append('#ecf0f1')  # Light gray for zero
                else:
                    normalized = (score - global_min_score) / score_range
                    color_val = 0.1 + (1 - normalized) * 0.8
                    bar_colors.append(plt.cm.RdYlGn(color_val))
            
            # Create horizontal bar chart
            y_pos = range(len(top_df))
            bars = ax.barh(y_pos, top_df['Total_Score'], color=bar_colors, 
                          edgecolor='black', linewidth=0.5)
            
            # Add value labels inside bars (only if score > 0)
            for i, (bar, score) in enumerate(zip(bars, top_df['Total_Score'])):
                if score > 0:
                    text_x = bar.get_width() * 0.95
                    ax.text(text_x, bar.get_y() + bar.get_height()/2, 
                           f'{int(score)}', va='center', ha='right', fontsize=9, 
                           color='black', fontweight='bold')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_df['Short_Name'], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Impact Score', fontsize=10)
            
            # Simple title without parameters (toggles shown below)
            ax.set_title(f'{preset}', fontsize=12, fontweight='bold', color='black')
            
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Add total annotation
            total = top_df['Total_Score'].sum()
            if total > 0:
                ax.annotate(f'Total: {int(total)}', 
                           xy=(0.98, 0.02), xycoords='axes fraction',
                           ha='right', va='bottom', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Draw parameter toggle scales below the subplot
            # Only show labels on the first (leftmost) subplot
            self._draw_parameter_toggles(ax, preset, y_start=-0.18, show_labels=(col_idx == 0), category=category)
        
        # Main title with category info (black font)
        fig.suptitle(f'{category.upper()}: Most Affected Activities\n{cat_desc}',
                    fontsize=14, fontweight='bold', color='black')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.38, left=0.12)  # More room for toggle scales and labels
        plt.show()
        
        # Export graph data to JSON
        self._export_graph_data_to_json(
            category=category,
            graph_type='affected_activities',
            all_data=all_data,
            presets=presets
        )
        
        return fig, all_data
    
    def plot_source_impacts_by_category(self, category, activities=None, top_n=15, figsize=None):
        """
        Plot source activity impacts for a single category, with subplots for each preset.
        Shows which activities CAUSE the most environmental impact.
        
        Args:
            category: Category name - one of 'Scope', 'Risk', 'Timeline', 'Spatial'
            activities: List of source activities. If None, uses all available.
            top_n: Number of top source activities to show per preset
            figsize: Figure size tuple. If None, auto-calculated based on number of presets.
            
        Returns:
            tuple: (matplotlib figure, dict of DataFrames per preset)
            
        Example:
            fig, data = engine.plot_source_impacts_by_category('Scope', top_n=15)
            fig, data = engine.plot_source_impacts_by_category('Timeline', top_n=10)
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Validate category
        valid_categories = get_preset_categories()
        if category not in valid_categories:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {valid_categories}")
        
        if activities is None:
            activities = self.activities
        
        # Get presets for this category
        presets_by_cat = get_presets_by_category()
        category_descriptions = get_category_descriptions()
        
        presets = presets_by_cat.get(category, [])
        n_presets = len(presets)
        
        if n_presets == 0:
            raise ValueError(f"No presets found for category '{category}'")
        
        # Category colors
        category_colors = {
            'Scope': '#3498db',
            'Risk': '#e74c3c',
            'Timeline': '#2ecc71',
            'Spatial': '#9b59b6'
        }
        cat_color = category_colors.get(category, '#7f8c8d')
        cat_desc = category_descriptions.get(category, '')
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (6 * n_presets, 8)
        
        # Create figure with subplots (1 row, n_presets columns)
        fig, axes = plt.subplots(1, n_presets, figsize=figsize)
        
        # Ensure axes is iterable
        if n_presets == 1:
            axes = [axes]
        
        # First pass: collect all data
        all_data = {}
        global_max_score = 0
        global_min_score = float('inf')
        
        for preset in presets:
            # Get impact data for this preset
            impact_results = []
            
            for activity in activities:
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                
                try:
                    impact_data = self.get_source_activity_impact(activity, presets=[preset])
                finally:
                    sys.stdout = old_stdout
                
                preset_score = impact_data['preset_scores'].get(preset, 0)
                impact_results.append({'Source_Activity': activity, 'Impact_Score': preset_score})
            
            df = pd.DataFrame(impact_results)
            df = df.sort_values('Impact_Score', ascending=False).reset_index(drop=True)
            
            top_df = df.head(top_n).copy()
            all_data[preset] = top_df
            
            if len(top_df) > 0 and top_df['Impact_Score'].sum() > 0:
                max_score = top_df['Impact_Score'].max()
                min_score = top_df[top_df['Impact_Score'] > 0]['Impact_Score'].min() if (top_df['Impact_Score'] > 0).any() else 0
                if max_score > global_max_score:
                    global_max_score = max_score
                if min_score < global_min_score and min_score > 0:
                    global_min_score = min_score
        
        if global_min_score == float('inf'):
            global_min_score = 0
        
        # Plot each preset
        for col_idx, preset in enumerate(presets):
            ax = axes[col_idx]
            top_df = all_data.get(preset, pd.DataFrame())
            
            if top_df.empty or top_df['Impact_Score'].sum() == 0:
                ax.text(0.5, 0.5, f'No data for\n{preset}', 
                       ha='center', va='center', fontsize=11,
                       transform=ax.transAxes)
                ax.set_title(f'{preset}', fontsize=12, fontweight='bold', color='black')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Shorten activity names
            top_df['Short_Name'] = top_df['Source_Activity'].apply(
                lambda x: x[:35] + '...' if len(x) > 38 else x
            )
            
            # Calculate colors
            score_range = global_max_score - global_min_score if global_max_score > global_min_score else 1
            bar_colors = []
            for score in top_df['Impact_Score']:
                if score == 0:
                    bar_colors.append('#ecf0f1')
                else:
                    normalized = (score - global_min_score) / score_range
                    color_val = 0.1 + (1 - normalized) * 0.8
                    bar_colors.append(plt.cm.RdYlGn(color_val))
            
            # Create horizontal bar chart
            y_pos = range(len(top_df))
            bars = ax.barh(y_pos, top_df['Impact_Score'], color=bar_colors, 
                          edgecolor='black', linewidth=0.5)
            
            # Add value labels (only if > 0)
            for i, (bar, score) in enumerate(zip(bars, top_df['Impact_Score'])):
                if score > 0:
                    text_x = bar.get_width() * 0.95
                    ax.text(text_x, bar.get_y() + bar.get_height()/2, 
                           f'{int(score)}', va='center', ha='right', fontsize=9,
                           color='black', fontweight='bold')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_df['Short_Name'], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Impact Score', fontsize=10)
            
            # Simple title without parameters (toggles shown below)
            ax.set_title(f'{preset}', fontsize=12, fontweight='bold', color='black')
            
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Add total annotation
            total = top_df['Impact_Score'].sum()
            if total > 0:
                ax.annotate(f'Total: {int(total)}', 
                           xy=(0.98, 0.02), xycoords='axes fraction',
                           ha='right', va='bottom', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Draw parameter toggle scales below the subplot
            # Only show labels on the first (leftmost) subplot
            self._draw_parameter_toggles(ax, preset, y_start=-0.18, show_labels=(col_idx == 0), category=category)
        
        # Main title with category info (black font)
        fig.suptitle(f'{category.upper()}: Source Activity Impacts\n{cat_desc}',
                    fontsize=14, fontweight='bold', color='black')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.38, left=0.12)  # More room for toggle scales and labels
        plt.show()
        
        # Export graph data to JSON
        self._export_graph_data_to_json(
            category=category,
            graph_type='source_impacts',
            all_data=all_data,
            presets=presets
        )
        
        return fig, all_data
    
    # ========================================================================
    # PORTFOLIO ANALYSIS METHODS
    # ========================================================================
    
    def load_portfolio(self, csv_path):
        """
        Load a portfolio from a CSV file.
        
        Args:
            csv_path: Path to the portfolio CSV file
            
        Returns:
            DataFrame with portfolio holdings
        """
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        # Filter to only equity holdings (skip cash)
        if 'Asset Class' in df.columns:
            df = df[df['Asset Class'] == 'Equity'].copy()
        
        # Clean up Weight column
        if 'Weight (%)' in df.columns:
            df['Weight'] = df['Weight (%)']
        elif 'Weight' not in df.columns:
            # Calculate weight from Market Value if not present
            if 'Market Value' in df.columns:
                df['Weight'] = df['Market Value'] / df['Market Value'].sum() * 100
        
        return df
    
    def get_sector_isic_mapping(self):
        """
        Return mapping from portfolio sector names to ISIC Section activities.
        Based on standard GICS sector classification to ISIC mapping.
        
        Returns:
            Dict mapping sector names to lists of ISIC Section activities
        """
        # Map common portfolio sectors to ISIC Sections
        # These are the ISIC Section names as they appear in the ENCORE data
        sector_mapping = {
            # Energy sector maps to Mining (oil/gas extraction) and Utilities
            'Energy': [
                'Mining and quarrying',
            ],
            # Industrials covers manufacturing and construction
            'Industrials': [
                'Manufacturing',
                'Construction',
            ],
            # Utilities sector
            'Utilities': [
                'Electricity, gas, steam and air conditioning supply',
                'Water supply; sewerage, waste management and remediation activities',
            ],
            # Consumer Staples - Food & Beverage
            'Consumer Staples': [
                'Agriculture, forestry and fishing',
                'Manufacturing',  # Food manufacturing
            ],
            # Materials
            'Materials': [
                'Mining and quarrying',
                'Manufacturing',  # Basic materials processing
            ],
            # Financials
            'Financials': [
                'Financial and insurance activities',
            ],
            # Health Care
            'Health Care': [
                'Human health and social work activities',
                'Manufacturing',  # Pharma manufacturing
            ],
            # Information Technology
            'Information Technology': [
                'Information and communication',
                'Professional, scientific and technical activities',
            ],
            # Consumer Discretionary
            'Consumer Discretionary': [
                'Wholesale and retail trade; repair of motor vehicles and motorcycles',
                'Manufacturing',
            ],
            # Communication Services
            'Communication Services': [
                'Information and communication',
            ],
            # Real Estate
            'Real Estate': [
                'Real estate activities',
                'Construction',
            ],
        }
        
        return sector_mapping
    
    def get_isic_styles(self):
        """
        Return consistent styling for ISIC economic activities.
        Includes colors and logo file paths for each sector.
        """
        assets_dir = Path(__file__).parent / 'assets' / 'ISIC_logos'
        
        return {
            'Mining and quarrying': {
                'color': '#2C3E50',  # Dark blue-grey
                'logo': assets_dir / 'Mining.png',
                'short': 'Mining'
            },
            'Manufacturing': {
                'color': '#E74C3C',  # Red
                'logo': assets_dir / 'Factory.png',
                'short': 'Manufacturing'
            },
            'Construction': {
                'color': '#F39C12',  # Orange
                'logo': assets_dir / 'Construction.png',
                'short': 'Construction'
            },
            'Agriculture, forestry and fishing': {
                'color': '#27AE60',  # Green
                'logo': assets_dir / 'Agriculture.png',
                'short': 'Agriculture'
            },
            'Electricity, gas, steam and air conditioning supply': {
                'color': '#F1C40F',  # Yellow
                'logo': assets_dir / 'Electricity.png',
                'short': 'Electricity'
            },
            'Water supply; sewerage, waste management and remediation activities': {
                'color': '#3498DB',  # Blue
                'logo': assets_dir / 'Water Supply.png',
                'short': 'Water/Waste'
            },
            'Transportation and storage': {
                'color': '#9B59B6',  # Purple
                'logo': assets_dir / 'Transportation.png',
                'short': 'Transport'
            },
            'Accommodation and food service activities': {
                'color': '#E67E22',  # Dark orange
                'logo': assets_dir / 'Hotel.png',
                'short': 'Hospitality'
            },
            'Information and communication': {
                'color': '#1ABC9C',  # Teal
                'logo': assets_dir / 'Information.png',
                'short': 'ICT'
            },
            'Financial and insurance activities': {
                'color': '#34495E',  # Dark grey
                'logo': assets_dir / 'Finance.png',
                'short': 'Finance'
            },
            'Real estate activities': {
                'color': '#95A5A6',  # Grey
                'logo': assets_dir / 'Real Estate.png',
                'short': 'Real Estate'
            },
            'Professional, scientific and technical activities': {
                'color': '#8E44AD',  # Dark purple
                'logo': assets_dir / 'Laboratory.png',
                'short': 'Professional'
            },
            'Administrative and support service activities': {
                'color': '#16A085',  # Dark teal
                'logo': assets_dir / 'Retail.png',  # Reuse retail for admin
                'short': 'Admin'
            },
            'Human health and social work activities': {
                'color': '#C0392B',  # Dark red
                'logo': assets_dir / 'Healthcare.png',
                'short': 'Health'
            },
            'Education': {
                'color': '#2980B9',  # Medium blue
                'logo': assets_dir / 'Education.png',
                'short': 'Education'
            },
            'Arts, entertainment and recreation': {
                'color': '#D35400',  # Burnt orange
                'logo': assets_dir / 'Arts.png',
                'short': 'Arts'
            },
            'Wholesale and retail trade; repair of motor vehicles and motorcycles': {
                'color': '#7F8C8D',  # Grey
                'logo': assets_dir / 'Retail.png',
                'short': 'Retail'
            },
            'Public administration and defence; compulsory social security': {
                'color': '#2C3E50',  # Navy
                'logo': assets_dir / 'Finance.png',  # Reuse finance for public admin
                'short': 'Public Admin'
            },
            'Other service activities': {
                'color': '#BDC3C7',  # Light grey
                'logo': None,
                'short': 'Other Services'
            },
            # Fallback for unknown
            'Other ISIC': {
                'color': '#95A5A6',
                'logo': None,
                'short': 'Other'
            },
            'Other Holdings': {
                'color': '#BDC3C7',
                'logo': None,
                'short': 'Other'
            }
        }
    
    def get_holding_styles(self, portfolio_type='oil_gas'):
        """
        Return styling for known major holdings.
        Uses company brand-inspired colors and logo file paths.
        
        Args:
            portfolio_type: 'oil_gas' or 'food_beverage' to load appropriate logos
        """
        if portfolio_type == 'food_beverage':
            assets_dir = Path(__file__).parent / 'assets' / 'holdings_logos_Food_Beverage'
            return {
                'NESTLE SA': {'color': '#7B6855', 'logo': assets_dir / 'Nestlé.svg.png', 'short': 'Nestlé'},
                'ANHEUSER-BUSCH INBEV SA': {'color': '#C8102E', 'logo': assets_dir / 'AB-InBev-logo.png', 'short': 'AB InBev'},
                'DANONE SA': {'color': '#0072CE', 'logo': assets_dir / 'Danone-Logo.png', 'short': 'Danone'},
                'DIAGEO PLC': {'color': '#004B87', 'logo': assets_dir / 'diageo-logo-png_seeklogo-40863.png', 'short': 'Diageo'},
                'HEINEKEN NV': {'color': '#00843D', 'logo': assets_dir / 'heineken-14-logo-png-transparent.png', 'short': 'Heineken'},
                'PERNOD RICARD SA': {'color': '#1E1E1E', 'logo': assets_dir / '1280px-Pernod_Ricard_logo_2019.svg.png', 'short': 'Pernod'},
                'KERRY GROUP PLC': {'color': '#00A651', 'logo': assets_dir / '2560px-Kerry_Group_logo_2020.svg.png', 'short': 'Kerry'},
                'DSM FIRMENICH AG': {'color': '#00A3E0', 'logo': assets_dir / 'DSM-Firmenich_Logo_2023.svg', 'short': 'DSM'},
                'CARLSBERG AS CL B': {'color': '#005F3B', 'logo': assets_dir / 'Carlsberg-logo.jpg', 'short': 'Carlsberg'},
                'COCA COLA HBC AG': {'color': '#F40009', 'logo': assets_dir / '2560px-Coca-Cola_logo.svg.png', 'short': 'Coca-Cola'},
                'CHOCOLADEFABRIKEN LINDT': {'color': '#8B4513', 'logo': assets_dir / 'logo_L&S.PNG', 'short': 'Lindt'},
                # Default palette for unknown holdings - more distinct colors
                '_palette': [
                    '#E63946', '#1D3557', '#2A9D8F', '#E9C46A', '#7209B7',
                    '#00B4D8', '#FB8500', '#606C38', '#D62828', '#023E8A',
                    '#6A0572', '#F77F00', '#118AB2', '#8338EC', '#06D6A0'
                ]
            }
        else:  # oil_gas default
            assets_dir = Path(__file__).parent / 'assets' / 'holdings_logos_Oil_Gas'
            return {
                'SHELL PLC': {'color': '#FBCE07', 'logo': assets_dir / 'Shell.png', 'short': 'Shell'},
                'TOTALENERGIES': {'color': '#E2001A', 'logo': assets_dir / 'TotalEnergies_logo.svg.png', 'short': 'Total'},
                'BP PLC': {'color': '#009E49', 'logo': assets_dir / 'BP_Helios_logo.svg.png', 'short': 'BP'},
                'ENI SPA': {'color': '#FDB813', 'logo': assets_dir / 'Logo_ENI.svg.png', 'short': 'Eni'},
                'EQUINOR': {'color': '#E31836', 'logo': assets_dir / 'Equinor.svg.png', 'short': 'Equinor'},
                'REPSOL SA': {'color': '#FF6B00', 'logo': assets_dir / 'Repsol_logo.svg.png', 'short': 'Repsol'},
                'ORLEN SA': {'color': '#E30613', 'logo': assets_dir / 'Orlen_wordmark_logo.svg.png', 'short': 'Orlen'},
                'OMV AG': {'color': '#003A70', 'logo': None, 'short': 'OMV'},
                'GALP ENERGIA SGPS SA CLASS B': {'color': '#FF6600', 'logo': None, 'short': 'Galp'},
                'NESTE': {'color': '#00A651', 'logo': None, 'short': 'Neste'},
                # Industrials
                'SIEMENS ENERGY AG': {'color': '#009999', 'logo': assets_dir / 'Siemens_Energy_logo.svg.png', 'short': 'Siemens E'},
                'VESTAS WIND SYSTEMS': {'color': '#0F3B6B', 'logo': assets_dir / 'Vestas_Logo.svg.png', 'short': 'Vestas'},
                'NORDEX': {'color': '#003366', 'logo': None, 'short': 'Nordex'},
                # Default palette for unknown holdings - more distinct colors
                '_palette': [
                    '#E63946', '#1D3557', '#2A9D8F', '#E9C46A', '#7209B7',
                    '#00B4D8', '#FB8500', '#606C38', '#D62828', '#023E8A',
                    '#6A0572', '#F77F00', '#118AB2', '#8338EC', '#06D6A0'
                ]
            }
    def _get_item_style(self, item_name, style_type='isic', portfolio_type='oil_gas'):
        """
        Get style (color, logo) for an item. Falls back to defaults if not found.
        
        Args:
            item_name: Name of the ISIC activity or holding
            style_type: 'isic' or 'holding'
            portfolio_type: 'oil_gas' or 'food_beverage' (only for holdings)
        """
        if style_type == 'isic':
            styles = self.get_isic_styles()
        else:
            styles = self.get_holding_styles(portfolio_type)
        
        if item_name in styles:
            return styles[item_name]
        
        # For holdings, try partial match
        if style_type == 'holding':
            for key in styles:
                if key != '_palette' and key.lower() in item_name.lower():
                    return styles[key]
        
        # Return default
        return {'color': '#95A5A6', 'logo': None, 'short': item_name[:15]}

    def get_portfolio_impact_contributions(self, portfolio_df, preset=None,
                                            intensity='M', timescale='All',
                                            direct_indirect='All', spatial='All',
                                            rating='All', svc_act_intensity='M'):
        """
        Calculate impact scores for each portfolio holding based on its sector mapping.
        
        Args:
            portfolio_df: DataFrame with portfolio holdings (must have 'Name', 'Sector', 'Weight' columns)
            preset: Optional preset name from presets_config.jsonc
            intensity: Min intensity for Activity → Pressures
            timescale: Timescale filter
            direct_indirect: Impact type filter
            spatial: Spatial scale filter
            rating: Sensitivity rating filter
            svc_act_intensity: Min dependency intensity for Services → Activities
            
        Returns:
            Dictionary with:
            - 'affected_activities': {activity_name: {holding_name: weighted_score}}
            - 'holdings': List of holdings with their scores
            - 'summary': Summary statistics
        """
        import pandas as pd
        
        # Load presets from config file
        presets = get_presets_as_param_dict()
        
        # Apply preset if provided
        if preset and preset in presets:
            intensity, timescale, direct_indirect, spatial, rating, svc_act_intensity = presets[preset]
        
        sector_mapping = self.get_sector_isic_mapping()
        
        result = {
            'affected_activities': {},  # {affected_activity: {holding_name: contribution_score}}
            'source_activities': {},    # {source_isic: {holding_name: (weight, raw_score)}}
            'holdings': [],
            'summary': {
                'total_holdings': len(portfolio_df),
                'total_weight': 0,
                'total_impact_score': 0
            }
        }
        
        for _, holding in portfolio_df.iterrows():
            holding_name = holding.get('Name', 'Unknown')
            sector = holding.get('Sector', '')
            weight = float(holding.get('Weight', 0))
            
            # Check if ISIC_Section is directly provided (from mapped portfolios)
            isic_section = holding.get('ISIC_Section', '')
            if isic_section and pd.notna(isic_section) and isic_section.strip():
                # Use ISIC_Section directly
                isic_activities = [isic_section.strip()]
            elif sector in sector_mapping:
                # Fall back to sector mapping
                isic_activities = sector_mapping[sector]
            else:
                continue
            
            result['summary']['total_weight'] += weight
            
            holding_data = {
                'name': holding_name,
                'sector': isic_section if isic_section else sector,
                'weight': weight,
                'isic_activities': isic_activities,
                'impact_score': 0
            }
            
            # For each ISIC activity this holding maps to
            for isic_activity in isic_activities:
                # Suppress print output
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                
                try:
                    flow_data = self.get_single_activity_flow(
                        isic_activity,
                        min_intensity=intensity,
                        timescale=timescale,
                        direct_indirect=direct_indirect,
                        spatial=spatial,
                        comp_svc_rating=rating,
                        svc_act_intensity=svc_act_intensity
                    )
                finally:
                    sys.stdout = old_stdout
                
                if flow_data and 'activity_scores' in flow_data:
                    activity_scores = flow_data['activity_scores']
                    
                    # Weight split across mapped ISIC activities
                    weight_share = weight / len(isic_activities)
                    
                    # Track source activity contribution
                    if isic_activity not in result['source_activities']:
                        result['source_activities'][isic_activity] = {}
                    raw_score = sum(activity_scores.values())
                    result['source_activities'][isic_activity][holding_name] = (weight_share, raw_score)
                    
                    # For each affected activity, add this holding's contribution
                    for affected_act, score in activity_scores.items():
                        weighted_contribution = score * (weight_share / 100)  # Weight is percentage
                        
                        if affected_act not in result['affected_activities']:
                            result['affected_activities'][affected_act] = {}
                        
                        if holding_name not in result['affected_activities'][affected_act]:
                            result['affected_activities'][affected_act][holding_name] = 0
                        
                        result['affected_activities'][affected_act][holding_name] += weighted_contribution
                        holding_data['impact_score'] += weighted_contribution
            
            result['holdings'].append(holding_data)
            result['summary']['total_impact_score'] += holding_data['impact_score']
        
        return result
    
    def plot_portfolio_impact_stacked(self, csv_path, preset=None, top_n=15, figsize=(16, 12),
                                       intensity='M', timescale='All',
                                       direct_indirect='All', spatial='All',
                                       rating='All', svc_act_intensity='M',
                                       min_contribution_pct=2.0,
                                       group_by='holding',
                                       portfolio_type='oil_gas'):
        """
        Create a stacked horizontal bar chart showing portfolio impact on affected activities.
        Each bar shows which portfolio holdings contribute to that activity's impact.
        
        Args:
            csv_path: Path to portfolio CSV file
            preset: Optional preset name from presets_config.jsonc
            top_n: Number of top affected activities to show
            figsize: Figure size tuple
            intensity: Min intensity for Activity → Pressures
            timescale: Timescale filter
            direct_indirect: Impact type filter
            spatial: Spatial scale filter
            rating: Sensitivity rating filter
            svc_act_intensity: Min dependency intensity
            min_contribution_pct: Minimum contribution % to show separately (others grouped as "Other")
            group_by: 'holding' to stack by portfolio holding, 'isic' to stack by ISIC source activity
            portfolio_type: 'oil_gas' or 'food_beverage' for loading correct holding logos
            
        Returns:
            tuple: (matplotlib figure, result data dictionary)
        """
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import pandas as pd
        import numpy as np
        
        # Set up clean font
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        plt.rcParams['font.size'] = 10
        
        # Load style dictionaries
        isic_styles = self.get_isic_styles()
        holding_styles = self.get_holding_styles(portfolio_type)
        
        # Helper function to load and resize logo
        def load_logo(logo_path, zoom=0.03):
            if logo_path and Path(logo_path).exists():
                try:
                    img = mpimg.imread(str(logo_path))
                    return OffsetImage(img, zoom=zoom)
                except Exception as e:
                    return None
            return None
        
        # Load portfolio
        portfolio_df = self.load_portfolio(csv_path)
        
        # Get portfolio name from CSV filename
        portfolio_name = Path(csv_path).stem.replace('_', ' ').replace('-', ' ')
        
        # Get impact contributions
        result = self.get_portfolio_impact_contributions(
            portfolio_df, preset=preset,
            intensity=intensity, timescale=timescale,
            direct_indirect=direct_indirect, spatial=spatial,
            rating=rating, svc_act_intensity=svc_act_intensity
        )
        
        if not result['affected_activities']:
            print("No impact data found for this portfolio")
            return None, result
        
        # Calculate total score per affected activity and sort
        activity_totals = []
        for activity, contributions in result['affected_activities'].items():
            total = sum(contributions.values())
            activity_totals.append((activity, total, contributions))
        
        activity_totals.sort(key=lambda x: x[1], reverse=True)
        top_activities = activity_totals[:top_n]
        
        # Get the maximum total for fixed offset calculation
        max_total = max(t[1] for t in top_activities) if top_activities else 1
        label_offset = max_total * 0.01  # Smaller offset - closer to bar end
        
        # Create figure with clean background
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('#FAFAFA')
        
        # Prepare data for stacking
        activities = [a[0] for a in top_activities]
        y_pos = list(range(len(top_activities)))
        
        # Prepare y-axis labels (short names only, logos added separately)
        y_labels = []
        y_logos = []  # Store logo paths for later
        for activity in activities:
            style = isic_styles.get(activity, {'logo': None, 'short': activity[:35]})
            short_name = style.get('short', activity[:35])
            if len(activity) > 41:
                short_name = activity[:38] + '...'
            y_labels.append(short_name)
            y_logos.append(style.get('logo'))
        
        if group_by == 'isic':
            # Group by ISIC source activity instead of holding
            isic_contributions = self._build_isic_contributions(result, top_activities)
            
            # Get all unique ISIC sources
            all_isic = set()
            for _, _, contribs in isic_contributions:
                all_isic.update(contribs.keys())
            
            # Sort by total contribution
            isic_totals = {}
            for isic in all_isic:
                total = sum(c.get(isic, 0) for _, _, c in isic_contributions)
                isic_totals[isic] = total
            
            sorted_isic = sorted(isic_totals.items(), key=lambda x: x[1], reverse=True)
            
            # Group small contributors
            total_all = sum(i[1] for i in sorted_isic)
            major_items = []
            for isic, total in sorted_isic:
                pct = (total / total_all * 100) if total_all > 0 else 0
                if pct >= min_contribution_pct:
                    major_items.append(isic)
            
            if len(major_items) < len(sorted_isic):
                major_items.append('Other ISIC')
            
            # Use ISIC color scheme
            item_colors = {}
            item_labels = {}
            item_logos = {}
            for item in major_items:
                style = isic_styles.get(item, {'color': '#95A5A6', 'logo': None, 'short': item[:15]})
                item_colors[item] = style['color']
                item_labels[item] = style.get('short', item[:15])
                item_logos[item] = style.get('logo')
            
            # Build stacked bars
            bottom = np.zeros(len(top_activities))
            bar_segments = []
            bar_height = 0.8  # Default matplotlib bar height
            
            for item in major_items:
                values = []
                for activity, total, _ in isic_contributions:
                    contribs = isic_contributions[[i for i, (a, _, _) in enumerate(isic_contributions) if a == activity][0]][2]
                    if item == 'Other ISIC':
                        other_sum = sum(v for k, v in contribs.items() if k not in major_items[:-1])
                        values.append(other_sum)
                    else:
                        values.append(contribs.get(item, 0))
                
                values = np.array(values)
                bars = ax.barh(y_pos, values, left=bottom, color=item_colors[item],
                       label=item_labels[item], edgecolor='white', linewidth=1.5,
                       alpha=0.9, height=bar_height)
                bar_segments.append((item, values, bottom.copy(), bars))
                bottom += values
            
            legend_title = 'ISIC Source Activity'
            
        else:
            # Group by holding
            all_holdings = set()
            for _, _, contributions in top_activities:
                all_holdings.update(contributions.keys())
            
            holding_totals = {}
            for holding in all_holdings:
                total = 0
                for _, _, contributions in top_activities:
                    total += contributions.get(holding, 0)
                holding_totals[holding] = total
            
            sorted_holdings = sorted(holding_totals.items(), key=lambda x: x[1], reverse=True)
            
            # Group small contributors
            total_all = sum(h[1] for h in sorted_holdings)
            major_items = []
            other_total = 0
            
            for holding, total in sorted_holdings:
                pct = (total / total_all * 100) if total_all > 0 else 0
                if pct >= min_contribution_pct:
                    major_items.append(holding)
                else:
                    other_total += total
            
            if other_total > 0:
                major_items.append('Other Holdings')
            
            # Use holding color scheme with fallback palette
            item_colors = {}
            item_labels = {}
            item_logos = {}
            palette = holding_styles.get('_palette', ['#3498DB', '#E74C3C', '#2ECC71'])
            palette_idx = 0
            
            for item in major_items:
                if item in holding_styles:
                    style = holding_styles[item]
                    item_colors[item] = style['color']
                    item_labels[item] = style['short']
                    item_logos[item] = style.get('logo')
                elif item == 'Other Holdings':
                    item_colors[item] = '#BDC3C7'
                    item_labels[item] = 'Other'
                    item_logos[item] = None
                else:
                    # Try partial match
                    matched = False
                    for key, style in holding_styles.items():
                        if key != '_palette' and key.upper() in item.upper():
                            item_colors[item] = style['color']
                            item_labels[item] = style['short']
                            item_logos[item] = style.get('logo')
                            matched = True
                            break
                    if not matched:
                        item_colors[item] = palette[palette_idx % len(palette)]
                        short_name = item[:12] + '...' if len(item) > 15 else item
                        item_labels[item] = short_name
                        item_logos[item] = None
                        palette_idx += 1
            
            # Build stacked bars
            bottom = np.zeros(len(top_activities))
            bar_segments = []
            bar_height = 0.8  # Default matplotlib bar height
            
            for item in major_items:
                values = []
                for activity, total, contributions in top_activities:
                    if item == 'Other Holdings':
                        other_sum = sum(v for k, v in contributions.items() if k not in major_items[:-1])
                        values.append(other_sum)
                    else:
                        values.append(contributions.get(item, 0))
                
                values = np.array(values)
                bars = ax.barh(y_pos, values, left=bottom, color=item_colors[item],
                       label=item_labels[item], edgecolor='white', linewidth=1.5,
                       alpha=0.9, height=bar_height)
                bar_segments.append((item, values, bottom.copy(), bars))
                bottom += values
            
            legend_title = 'Holdings'
        
        # Add logos inside bars - scale to fit both height and width
        target_logo_height = 25  # Target height in display pixels
        min_bar_width_ratio = 0.03  # Minimum bar width as ratio of total to show logo
        
        # Get figure DPI and size to calculate pixels per data unit
        fig_width_inches = fig.get_figwidth()
        fig_dpi = fig.get_dpi()
        fig_width_pixels = fig_width_inches * fig_dpi
        
        # Approximate plot area (accounting for margins ~25% left, ~28% right for legend)
        plot_width_pixels = fig_width_pixels * 0.47  # Usable plot area
        pixels_per_data_unit = plot_width_pixels / max_total if max_total > 0 else 1
        
        # Single pass: check if logo fits, make bar hollow, and add logo
        from matplotlib.patches import FancyBboxPatch
        for item, values, lefts, bars in bar_segments:
            logo_path = item_logos.get(item)
            if not logo_path or not Path(logo_path).exists():
                continue
                
            try:
                img = mpimg.imread(str(logo_path))
                img_height, img_width = img.shape[0], img.shape[1]
                
                for i, (val, left, bar) in enumerate(zip(values, lefts, bars)):
                    # Check if bar is wide enough (at least 3% of max total)
                    if val < max_total * min_bar_width_ratio:
                        continue
                    
                    # Calculate bar width in pixels
                    bar_width_pixels = val * pixels_per_data_unit
                    max_logo_width_pixels = bar_width_pixels * 0.7  # Logo can use up to 70% of bar width
                    
                    # Calculate zoom based on height
                    zoom_by_height = target_logo_height / img_height
                    logo_width_at_height_zoom = img_width * zoom_by_height
                    
                    # If logo would be too wide, scale by width instead
                    if logo_width_at_height_zoom > max_logo_width_pixels:
                        zoom = max_logo_width_pixels / img_width
                    else:
                        zoom = zoom_by_height
                    
                    # Skip if logo would be too small to see
                    if zoom * img_height < 8 or zoom * img_width < 8:
                        continue
                    
                    # Logo fits! Draw inner white rectangle to create hollow effect
                    # Keep original bar, add white fill inside with colored inner border
                    bar_color = item_colors[item]
                    # Use fixed border width for consistent frame appearance (3x thinner)
                    border_width = max_total * 0.003  # Fixed fraction of max total
                    border_height = 0.06  # Border height in data units
                    
                    # Draw white inner rectangle (slightly smaller than bar)
                    inner_rect = plt.Rectangle(
                        (left + border_width, i - bar_height/2 + border_height),
                        max(val - 2*border_width, 0),  # Ensure non-negative width
                        bar_height - 2*border_height,
                        facecolor='white',
                        edgecolor='none',
                        zorder=bar.get_zorder() + 1
                    )
                    ax.add_patch(inner_rect)
                    
                    imagebox = OffsetImage(img, zoom=zoom)
                    imagebox.image.axes = ax
                    # Center logo in bar segment
                    center_x = left + val / 2
                    ab = AnnotationBbox(imagebox, (center_x, i),
                                       frameon=False, box_alignment=(0.5, 0.5),
                                       xycoords=('data', 'data'),
                                       zorder=bar.get_zorder() + 2)
                    ax.add_artist(ab)
            except Exception:
                pass
        
        # Add total value labels at end of bars
        for i, (activity, total, _) in enumerate(top_activities):
            if total > 0:
                ax.text(total + label_offset, i, f'{total:.1f}', va='center', ha='left',
                       fontsize=10, color='#2C3E50', fontweight='bold')
        
        # Formatting with clean style
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=10)
        ax.tick_params(axis='y', pad=35)  # Add padding for ISIC logos between labels and axis
        ax.invert_yaxis()
        
        # Add ISIC logos as part of ylabel area (between labels and axis)
        isic_logo_size = 24  # Fixed pixel size for ISIC logos
        from matplotlib.transforms import blended_transform_factory
        # Create transform: x in axes fraction, y in data coordinates
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for i, logo_path in enumerate(y_logos):
            if logo_path and Path(logo_path).exists():
                try:
                    img = mpimg.imread(str(logo_path))
                    img_width = img.shape[1]
                    uniform_zoom = isic_logo_size / img_width
                    imagebox = OffsetImage(img, zoom=uniform_zoom)
                    imagebox.image.axes = ax
                    # Position logo between ylabel and y-axis
                    ab = AnnotationBbox(imagebox, (-0.005, i),
                                       frameon=False, box_alignment=(1, 0.5),
                                       xycoords=trans)
                    ax.add_artist(ab)
                except Exception:
                    pass
        
        ax.set_xlabel('Weighted Impact Score (by Portfolio Weight)', fontsize=11, fontweight='bold', color='#2C3E50')
        
        group_label = 'ISIC Source Activity' if group_by == 'isic' else 'Holding'
        ax.set_title(f'Portfolio Impact Analysis: {portfolio_name}\n'
                    f'Affected Activities with Contributions by {group_label}',
                    fontsize=14, fontweight='bold', color='#2C3E50', pad=20)
        
        # Clean grid with automatic tick spacing
        ax.grid(axis='x', alpha=0.3, linestyle='-', color='#BDC3C7')
        ax.set_axisbelow(True)
        
        # Remove spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
        
        # Set x-axis limits - ISIC logos are now in ylabel area, so start at 0
        ax.set_xlim(left=0, right=max_total * 1.15)
        
        # Create custom legend with logos - position in bottom right
        from matplotlib.patches import Patch
        legend_handles = []
        for item in major_items:
            patch = Patch(facecolor=item_colors[item], edgecolor='white', 
                         label=item_labels[item], alpha=0.9)
            legend_handles.append(patch)
        
        legend = ax.legend(handles=legend_handles, loc='lower right', 
                          fontsize=20, title=legend_title, title_fontsize=24,
                          frameon=True, fancybox=True, shadow=False,
                          edgecolor='#BDC3C7', facecolor='white', ncol=2)
        
        # Add summary annotation - position in bottom center
        summary = result['summary']
        summary_text = (f"Holdings: {len(portfolio_df)} | "
                       f"Weight: {summary['total_weight']:.1f}% | "
                       f"Total Impact: {summary['total_impact_score']:.1f}")
        ax.annotate(summary_text, xy=(0.5, 0.02), xycoords='axes fraction',
                   ha='center', va='bottom', fontsize=11, color='#2C3E50',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECF0F1', 
                            edgecolor='#BDC3C7', alpha=0.95))
        
        plt.tight_layout()
        plt.show()
        
        return fig, result
    
    def _build_isic_contributions(self, result, top_activities):
        """
        Build ISIC-based contributions for affected activities.
        Maps holdings back to their ISIC source activities.
        
        Returns list of (affected_activity, total, {isic_source: contribution})
        """
        sector_mapping = self.get_sector_isic_mapping()
        
        # Build holding -> ISIC mapping from the holdings data
        holding_to_isic = {}
        for holding_data in result['holdings']:
            holding_to_isic[holding_data['name']] = holding_data.get('isic_activities', [])
        
        isic_contributions = []
        for activity, total, holding_contribs in top_activities:
            isic_contribs = {}
            for holding, contrib in holding_contribs.items():
                isic_list = holding_to_isic.get(holding, [])
                if isic_list:
                    # Split contribution equally among ISIC activities
                    per_isic = contrib / len(isic_list)
                    for isic in isic_list:
                        isic_contribs[isic] = isic_contribs.get(isic, 0) + per_isic
            isic_contributions.append((activity, total, isic_contribs))
        
        return isic_contributions
    
    def plot_portfolio_source_impact_stacked(self, csv_path, preset=None, top_n=10, figsize=(16, 10),
                                              intensity='M', timescale='All',
                                              direct_indirect='All', spatial='All',
                                              rating='All', svc_act_intensity='M',
                                              min_contribution_pct=2.0,
                                              portfolio_type='oil_gas'):
        """
        Create a stacked horizontal bar chart showing source activities (ISIC Sections)
        with contributions from portfolio holdings that map to each source activity.
        
        Args:
            csv_path: Path to portfolio CSV file
            preset: Optional preset name from presets_config.jsonc
            top_n: Number of top source activities to show
            figsize: Figure size tuple
            intensity: Min intensity for Activity → Pressures
            timescale: Timescale filter
            direct_indirect: Impact type filter
            spatial: Spatial scale filter
            rating: Sensitivity rating filter
            svc_act_intensity: Min dependency intensity
            min_contribution_pct: Minimum contribution % to show separately
            portfolio_type: 'oil_gas' or 'food_beverage' for loading correct holding logos
            
        Returns:
            tuple: (matplotlib figure, result data dictionary)
        """
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from matplotlib.patches import Patch
        import pandas as pd
        import numpy as np
        
        # Set up clean font
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        plt.rcParams['font.size'] = 10
        
        # Load style dictionaries
        isic_styles = self.get_isic_styles()
        holding_styles = self.get_holding_styles(portfolio_type)
        
        # Load portfolio
        portfolio_df = self.load_portfolio(csv_path)
        
        # Get portfolio name from CSV filename
        portfolio_name = Path(csv_path).stem.replace('_', ' ').replace('-', ' ')
        
        # Get impact contributions
        result = self.get_portfolio_impact_contributions(
            portfolio_df, preset=preset,
            intensity=intensity, timescale=timescale,
            direct_indirect=direct_indirect, spatial=spatial,
            rating=rating, svc_act_intensity=svc_act_intensity
        )
        
        source_activities = result.get('source_activities', {})
        
        if not source_activities:
            print("No source activity data found for this portfolio")
            return None, result
        
        # Calculate total score per source activity
        activity_data = []
        for source_act, holdings_data in source_activities.items():
            # holdings_data is {holding_name: (weight_share, raw_score)}
            total_weighted_score = sum(w * s for w, s in holdings_data.values()) / 100
            holding_contributions = {h: (w * s / 100) for h, (w, s) in holdings_data.items()}
            activity_data.append((source_act, total_weighted_score, holding_contributions))
        
        activity_data.sort(key=lambda x: x[1], reverse=True)
        top_activities = activity_data[:top_n]
        
        # Get the maximum total for fixed offset calculation
        max_total = max(t[1] for t in top_activities) if top_activities else 1
        label_offset = max_total * 0.01  # Smaller offset - closer to bar end
        
        # Get unique holdings
        all_holdings = set()
        for _, _, contributions in top_activities:
            all_holdings.update(contributions.keys())
        
        # Sort holdings by total contribution
        holding_totals = {}
        for holding in all_holdings:
            total = sum(c.get(holding, 0) for _, _, c in top_activities)
            holding_totals[holding] = total
        
        sorted_holdings = sorted(holding_totals.items(), key=lambda x: x[1], reverse=True)
        
        # Group small contributors
        total_all = sum(h[1] for h in sorted_holdings)
        major_holdings = []
        
        for holding, total in sorted_holdings:
            pct = (total / total_all * 100) if total_all > 0 else 0
            if pct >= min_contribution_pct:
                major_holdings.append(holding)
        
        if len(major_holdings) < len(sorted_holdings):
            major_holdings.append('Other Holdings')
        
        # Create figure with clean background
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('#FAFAFA')
        
        # Prepare data - use ISIC styles for y-axis labels
        activities = [a[0] for a in top_activities]
        y_labels = []
        y_logos = []
        for activity in activities:
            style = isic_styles.get(activity, {'logo': None, 'short': activity[:35]})
            short_name = style.get('short', activity[:35])
            if len(activity) > 41:
                short_name = activity[:38] + '...'
            y_labels.append(short_name)
            y_logos.append(style.get('logo'))
        
        # Use holding color scheme with fallback palette
        item_colors = {}
        item_labels = {}
        item_logos = {}
        palette = holding_styles.get('_palette', ['#3498DB', '#E74C3C', '#2ECC71'])
        palette_idx = 0
        
        for item in major_holdings:
            if item in holding_styles:
                style = holding_styles[item]
                item_colors[item] = style['color']
                item_labels[item] = style['short']
                item_logos[item] = style.get('logo')
            elif item == 'Other Holdings':
                item_colors[item] = '#BDC3C7'
                item_labels[item] = 'Other'
                item_logos[item] = None
            else:
                # Try partial match
                matched = False
                for key, style in holding_styles.items():
                    if key != '_palette' and key.upper() in item.upper():
                        item_colors[item] = style['color']
                        item_labels[item] = style['short']
                        item_logos[item] = style.get('logo')
                        matched = True
                        break
                if not matched:
                    item_colors[item] = palette[palette_idx % len(palette)]
                    short_name = item[:12] + '...' if len(item) > 15 else item
                    item_labels[item] = short_name
                    item_logos[item] = None
                    palette_idx += 1
        
        # Build stacked bars
        y_pos = list(range(len(top_activities)))
        bottom = np.zeros(len(top_activities))
        bar_segments = []  # Store for percentage labels
        bar_height = 0.8  # Default matplotlib bar height
        
        for holding in major_holdings:
            values = []
            for activity, total, contributions in top_activities:
                if holding == 'Other Holdings':
                    other_sum = sum(v for k, v in contributions.items() if k not in major_holdings[:-1])
                    values.append(other_sum)
                else:
                    values.append(contributions.get(holding, 0))
            
            values = np.array(values)
            bars = ax.barh(y_pos, values, left=bottom, color=item_colors[holding],
                   label=item_labels[holding], edgecolor='white', linewidth=1.5,
                   alpha=0.9, height=bar_height)
            bar_segments.append((holding, values, bottom.copy(), bars))
            bottom += values
        
        # Add logos inside bars - scale to fit both height and width
        target_logo_height = 25  # Target height in display pixels
        min_bar_width_ratio = 0.03  # Minimum bar width as ratio of total to show logo
        
        # Get figure DPI and size to calculate pixels per data unit
        fig_width_inches = fig.get_figwidth()
        fig_dpi = fig.get_dpi()
        fig_width_pixels = fig_width_inches * fig_dpi
        
        # Approximate plot area (accounting for margins ~25% left, ~28% right for legend)
        plot_width_pixels = fig_width_pixels * 0.47  # Usable plot area
        pixels_per_data_unit = plot_width_pixels / max_total if max_total > 0 else 1
        
        # Single pass: check if logo fits, make bar hollow, and add logo
        from matplotlib.patches import FancyBboxPatch
        for holding, values, lefts, bars in bar_segments:
            logo_path = item_logos.get(holding)
            if not logo_path or not Path(logo_path).exists():
                continue
                
            try:
                img = mpimg.imread(str(logo_path))
                img_height, img_width = img.shape[0], img.shape[1]
                
                for i, (val, left, bar) in enumerate(zip(values, lefts, bars)):
                    # Check if bar is wide enough (at least 3% of max total)
                    if val < max_total * min_bar_width_ratio:
                        continue
                    
                    # Calculate bar width in pixels
                    bar_width_pixels = val * pixels_per_data_unit
                    max_logo_width_pixels = bar_width_pixels * 0.7  # Logo can use up to 70% of bar width
                    
                    # Calculate zoom based on height
                    zoom_by_height = target_logo_height / img_height
                    logo_width_at_height_zoom = img_width * zoom_by_height
                    
                    # If logo would be too wide, scale by width instead
                    if logo_width_at_height_zoom > max_logo_width_pixels:
                        zoom = max_logo_width_pixels / img_width
                    else:
                        zoom = zoom_by_height
                    
                    # Skip if logo would be too small to see
                    if zoom * img_height < 8 or zoom * img_width < 8:
                        continue
                    
                    # Logo fits! Draw inner white rectangle to create hollow effect
                    bar_color = item_colors[holding]
                    # Use fixed border width for consistent frame appearance (3x thinner)
                    border_width = max_total * 0.003  # Fixed fraction of max total
                    border_height = 0.02  # Border height in data units
                    
                    # Draw white inner rectangle (slightly smaller than bar)
                    inner_rect = plt.Rectangle(
                        (left + border_width, i - bar_height/2 + border_height),
                        max(val - 2*border_width, 0),  # Ensure non-negative width
                        bar_height - 2*border_height,
                        facecolor='white',
                        edgecolor='none',
                        zorder=bar.get_zorder() + 1
                    )
                    ax.add_patch(inner_rect)
                    
                    imagebox = OffsetImage(img, zoom=zoom)
                    imagebox.image.axes = ax
                    # Center logo in bar segment
                    center_x = left + val / 2
                    ab = AnnotationBbox(imagebox, (center_x, i),
                                       frameon=False, box_alignment=(0.5, 0.5),
                                       xycoords=('data', 'data'),
                                       zorder=bar.get_zorder() + 2)
                    ax.add_artist(ab)
            except Exception:
                pass
        
        # Add total labels (fixed offset)
        for i, (activity, total, _) in enumerate(top_activities):
            if total > 0:
                ax.text(total + label_offset, i, f'{total:.1f}', va='center', ha='left',
                       fontsize=10, color='#2C3E50', fontweight='bold')
        
        # Formatting with clean style
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=10)
        ax.tick_params(axis='y', pad=35)  # Add padding for ISIC logos between labels and axis
        ax.invert_yaxis()
        
        # Add ISIC logos as part of ylabel area (between labels and axis)
        isic_logo_size = 24  # Fixed pixel size for ISIC logos
        from matplotlib.transforms import blended_transform_factory
        # Create transform: x in axes fraction, y in data coordinates
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for i, logo_path in enumerate(y_logos):
            if logo_path and Path(logo_path).exists():
                try:
                    img = mpimg.imread(str(logo_path))
                    img_width = img.shape[1]
                    uniform_zoom = isic_logo_size / img_width
                    imagebox = OffsetImage(img, zoom=uniform_zoom)
                    imagebox.image.axes = ax
                    # Position logo between ylabel and y-axis
                    ab = AnnotationBbox(imagebox, (-0.005, i),
                                       frameon=False, box_alignment=(1, 0.5),
                                       xycoords=trans)
                    ax.add_artist(ab)
                except Exception:
                    pass
        
        ax.set_xlabel('Weighted Impact Score (by Portfolio Weight)', fontsize=11, fontweight='bold', color='#2C3E50')
        ax.set_title(f'Portfolio Source Impact Analysis: {portfolio_name}\n'
                    f'ISIC Sections with Contributions by Holding',
                    fontsize=14, fontweight='bold', color='#2C3E50', pad=20)
        
        # Clean grid with automatic tick spacing
        ax.grid(axis='x', alpha=0.3, linestyle='-', color='#BDC3C7')
        ax.set_axisbelow(True)
        
        # Remove spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
        
        # Set x-axis limits - ISIC logos are now in ylabel area, so start at 0
        ax.set_xlim(left=0, right=max_total * 1.15)
        
        # Create custom legend with color patches - position in bottom right
        legend_handles = []
        for item in major_holdings:
            patch = Patch(facecolor=item_colors[item], edgecolor='white', 
                         label=item_labels[item], alpha=0.9)
            legend_handles.append(patch)
        
        legend = ax.legend(handles=legend_handles, loc='lower right', 
                          fontsize=14, title='Holdings', title_fontsize=15,
                          frameon=True, fancybox=True, shadow=False,
                          edgecolor='#BDC3C7', facecolor='white', ncol=2)
        
        # Summary - position in bottom center
        summary = result['summary']
        summary_text = (f"Holdings: {len(portfolio_df)} | "
                       f"Weight: {summary['total_weight']:.1f}% | "
                       f"Total Impact: {summary['total_impact_score']:.1f}")
        ax.annotate(summary_text, xy=(0.5, 0.02), xycoords='axes fraction',
                   ha='center', va='bottom', fontsize=11, color='#2C3E50',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECF0F1', 
                            edgecolor='#BDC3C7', alpha=0.95))
        
        plt.tight_layout()
        plt.show()
        
        return fig, result