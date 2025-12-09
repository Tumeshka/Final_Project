"""
Portfolio Environmental Impact Network Engine

This module contains the core functionality for creating and visualizing
environmental impact networks from investment portfolio data using ENCORE data.
Includes dashboard, analysis, and export functionality.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class PortfolioImpactEngine:
    """
    Engine for creating and analyzing portfolio environmental impact networks.
    """
    
    def __init__(self, data_directory="ENCORE_data"):
        """
        Initialize the engine with ENCORE data.
        
        Args:
            data_directory (str): Path to ENCORE data directory
        """
        self.data_dir = Path(data_directory)
        self.all_data = {}
        self.portfolio_networks = {}
        
        # Load data on initialization
        self._load_encore_data()
    
    def _load_encore_data(self):
        """Load all ENCORE data files."""
        encore_files_dir = self.data_dir / 'ENCORE files'
        crosswalk_dir = self.data_dir / 'Crosswalk tables'
        
        # Load ENCORE files
        encore_files = {
            'ecosystem_services_def': pd.read_csv(encore_files_dir / '02. Ecosystem services definitions.csv'),
            'dependency_links': pd.read_csv(encore_files_dir / '03. Dependency links.csv'),
            'pressure_definitions': pd.read_csv(encore_files_dir / '04. Pressure definitions.csv'),
            'pressure_links': pd.read_csv(encore_files_dir / '05. Pressure links.csv'),
            'dependency_mat_ratings': pd.read_csv(encore_files_dir / '06. Dependency mat ratings.csv'),
            'pressure_mat_ratings': pd.read_csv(encore_files_dir / '07. Pressure mat ratings.csv'),
            'ecosystem_components_def': pd.read_csv(encore_files_dir / '10. Ecosystem components definitions.csv'),
            'ecosystem_services_components': pd.read_csv(encore_files_dir / '11. Ecosystem services and ecosystem components.csv'),
            'mechanisms_change_def': pd.read_csv(encore_files_dir / '12. Mechanisms of change in state definitions.csv'),
            'pressures_to_components': pd.read_csv(encore_files_dir / '13. Pressures to components.csv'),
            'exiobase_nace_isic': pd.read_csv(encore_files_dir / '14. EXIOBASE NACE ISIC crosswalk.csv'),
            'value_chain_notes': pd.read_csv(encore_files_dir / '15. Note on value chain links.csv'),
            'upstream_links': pd.read_csv(encore_files_dir / '16. Upstream links.csv'),
            'downstream_links': pd.read_csv(encore_files_dir / '17. Downstream links.csv'),
            'explanatory_notes': pd.read_csv(encore_files_dir / '18. Explanatory notes.csv')
        }
        
        # Load crosswalk files
        crosswalk_files = {
            'exiobase_nace_isic_crosswalk': pd.read_csv(crosswalk_dir / 'EXIOBASE NACE ISIC crosswalk.csv'),
            'impact_drivers_pressures': pd.read_csv(crosswalk_dir / 'List of impact drivers and list of pressures.csv'),
            'gics_encore_isic': pd.read_excel(crosswalk_dir / 'GICS - ENCORE production processes - ISIC .xlsx')
        }
        
        # Combine all data
        self.all_data = {**encore_files, **crosswalk_files}
    
    def create_comprehensive_impact_network(self, portfolio_theme="Energy", max_activities=8):
        """
        Create a comprehensive circular impact network modeling the complete ENCORE system:
        Economic Activities → Pressures → Mechanisms → Natural Capital → Ecosystem Services → Economic Activities
        Plus: Supply Chain Links (Upstream/Downstream) → Economic Activities & Pressures
        
        Args:
            portfolio_theme (str): Theme for the portfolio
            max_activities (int): Maximum number of activities to include
            
        Returns:
            networkx.DiGraph: The comprehensive circular impact network
        """
        G = nx.DiGraph()
        
        # Get all ENCORE data
        pressure_links = self.all_data['pressure_links']
        dependency_links = self.all_data['dependency_links']
        pressures_to_components = self.all_data['pressures_to_components']
        ecosystem_services_components = self.all_data['ecosystem_services_components']
        upstream_links = self.all_data['upstream_links']
        downstream_links = self.all_data['downstream_links']
        
        # Define thematic portfolios with broader keyword matching
        themes = {
            "Energy": ["electric", "power", "petroleum", "coal", "gas", "energy", "solar", "wind", "hydro", "nuclear", "oil", "fuel"],
            "Agriculture": ["growing", "crop", "livestock", "farming", "agricultural", "cultivation", "animal", "plant", "food", "cattle", "poultry"],
            "Technology": ["software", "computer", "data", "information", "tech", "digital", "electronic", "communication", "programming"],
            "Manufacturing": ["manufacturing", "production", "industrial", "machinery", "fabrication", "assembly", "processing"],
            "Mining": ["mining", "extraction", "quarrying", "ore", "mineral", "excavation"],
            "Financial": ["banking", "finance", "insurance", "credit", "investment", "financial"],
            "Healthcare": ["health", "medical", "pharmaceutical", "hospital", "care", "medicine"],
            "Transport": ["transport", "logistics", "shipping", "aviation", "railway", "automotive"],
            "Construction": ["construction", "building", "infrastructure", "real estate", "development"],
            "Retail": ["retail", "wholesale", "trade", "commercial", "sales", "distribution"]
        }
        
        # Add portfolio node
        portfolio_node = f"{portfolio_theme} Portfolio"
        G.add_node(portfolio_node, node_type="portfolio", color="#FFD700", size=5000, layer=0)
        
        # Find activities matching the theme
        theme_keywords = themes.get(portfolio_theme, ["general"])
        matching_activities = []
        
        for _, row in pressure_links.iterrows():
            activity_name = str(row['ISIC Class']) if pd.notna(row['ISIC Class']) else ""
            division_name = str(row['ISIC Division']) if pd.notna(row['ISIC Division']) else ""
            
            # Check both class and division for matches
            combined_text = f"{activity_name} {division_name}".lower()
            if any(keyword in combined_text for keyword in theme_keywords):
                matching_activities.append((activity_name, row))
        
        # Remove duplicates and limit
        unique_activities = []
        seen_activities = set()
        for activity_name, row in matching_activities:
            if activity_name not in seen_activities and pd.notna(activity_name) and activity_name.strip():
                unique_activities.append((activity_name, row))
                seen_activities.add(activity_name)
                if len(unique_activities) >= max_activities:
                    break
        
        # LAYER 1: Economic Activities
        activity_nodes = []
        for activity_name, activity_row in unique_activities:
            # Clean activity name
            clean_name = activity_name[:40] + "..." if len(activity_name) > 40 else activity_name
            activity_node = f"Activity: {clean_name}"
            activity_nodes.append((activity_node, activity_name))
            
            # Add activity node
            G.add_node(activity_node, node_type="activity", color="#00FF7F", size=3000, 
                      full_name=activity_name, layer=1)
            G.add_edge(portfolio_node, activity_node, relationship="investment")
            
            # LAYER 2: Environmental Pressures
            pressure_categories = [
                'Emissions of GHG', 'Area of land use', 'Volume of water use', 
                'Emissions of toxic soil and water pollutants', 
                'Emissions of non-GHG air pollutants',
                'Generation and release of solid waste',
                'Other biotic resource extraction (e.g. fish, timber)',
                'Area of Freshwater use'
            ]
            
            for pressure_cat in pressure_categories:
                if (pressure_cat in activity_row and 
                    pd.notna(activity_row[pressure_cat]) and 
                    activity_row[pressure_cat] not in ['N/A', 'ND']):
                    
                    pressure_node = f"Pressure: {pressure_cat}"
                    
                    # Add pressure node if not exists
                    if pressure_node not in G.nodes():
                        G.add_node(pressure_node, node_type="pressure", color="#FF6B35", 
                                 size=2200, layer=2)
                    
                    # Add edge with materiality
                    materiality = activity_row[pressure_cat]
                    edge_colors = {
                        'VH': '#FF0000', 'H': '#FF4500', 'M': '#FFD700', 
                        'L': '#87CEEB', 'VL': '#D3D3D3'
                    }
                    edge_color = edge_colors.get(materiality, '#808080')
                    edge_width = {'VH': 5, 'H': 4, 'M': 3, 'L': 2, 'VL': 1}.get(materiality, 2)
                    
                    G.add_edge(activity_node, pressure_node, 
                             materiality=materiality, color=edge_color, width=edge_width,
                             relationship="generates")
        
        # LAYER 3: Mechanisms of Change in State
        pressure_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'pressure']
        mechanisms_added = set()
        
        for _, mech_row in pressures_to_components.iterrows():
            pressure_name = str(mech_row.get('Pressures(/Impact drivers)', '')).strip()
            mechanism_name = str(mech_row.get('Mechanism causing state change', '')).strip()
            
            if pd.notna(mechanism_name) and mechanism_name and mechanism_name != 'nan':
                pressure_node = f"Pressure: {pressure_name}"
                mech_node = f"Mechanism: {mechanism_name}"
                
                # Check if this pressure exists in our network
                if pressure_node in G.nodes() and mech_node not in mechanisms_added:
                    G.add_node(mech_node, node_type="mechanism", color="#DC143C", 
                             size=1800, layer=3)
                    G.add_edge(pressure_node, mech_node, relationship="causes")
                    mechanisms_added.add(mech_node)
        
        # LAYER 4: Natural Capital / Ecosystem Components
        mechanism_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'mechanism']
        components_added = set()
        
        for _, comp_row in pressures_to_components.iterrows():
            mechanism_name = str(comp_row.get('Mechanism causing state change', '')).strip()
            component_name = str(comp_row.get('Ecosystem component', '')).strip()
            
            if pd.notna(component_name) and component_name and component_name != 'nan':
                mech_node = f"Mechanism: {mechanism_name}"
                comp_node = f"Natural Capital: {component_name}"
                
                # Check if this mechanism exists in our network
                if mech_node in G.nodes() and comp_node not in components_added:
                    G.add_node(comp_node, node_type="natural_capital", color="#228B22", 
                             size=1600, layer=4)
                    G.add_edge(mech_node, comp_node, relationship="affects")
                    components_added.add(comp_node)
        
        # LAYER 5: Ecosystem Services
        # Connect natural capital to ecosystem services
        if 'ecosystem_services_components' in self.all_data:
            eco_services = self.all_data['ecosystem_services_components']
            services_added = set()
            
            # Get a sample of key ecosystem services
            key_services = [
                "Water supply", "Global climate regulation", "Local (micro and meso) climate regulation",
                "Pollination", "Air Filtration", "Water purification services", 
                "Biomass provisioning", "Genetic material"
            ]
            
            for service in key_services:
                service_node = f"Ecosystem Service: {service}"
                if service_node not in services_added:
                    G.add_node(service_node, node_type="ecosystem_service", color="#9370DB", 
                             size=1400, layer=5)
                    services_added.add(service_node)
                    
                    # Connect to relevant natural capital components
                    component_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'natural_capital']
                    for comp_node in component_nodes[:2]:  # Connect to first 2 components
                        G.add_edge(comp_node, service_node, relationship="provides")
        
        # LAYER 6: Circular Connection - Ecosystem Services back to Economic Activities
        # Connect ecosystem services back to economic activities (dependency)
        service_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'ecosystem_service']
        
        # Find dependencies for our activities
        for activity_node, activity_full_name in activity_nodes[:2]:  # Connect first 2 activities
            # Look for dependencies in dependency_links
            activity_deps = dependency_links[dependency_links['ISIC Class'] == activity_full_name]
            
            if not activity_deps.empty:
                activity_row = activity_deps.iloc[0]
                
                # Check for key ecosystem service dependencies
                key_deps = ['Water supply ', 'Global climate regulation', 'Pollination', 'Biomass provisioning']
                
                for dep in key_deps:
                    if dep in activity_row and pd.notna(activity_row[dep]) and activity_row[dep] not in ['N/A', 'ND']:
                        service_node = f"Ecosystem Service: {dep.strip()}"
                        if service_node in G.nodes():
                            # Create circular dependency
                            G.add_edge(service_node, activity_node, 
                                     materiality=activity_row[dep], 
                                     relationship="supports",
                                     color="#9370DB", width=2)
        
        # CORRECTED SUPPLY CHAIN CONNECTIONS
        # Ecosystem Services → Supply Chain Links → Economic Activities & Pressures
        
        # Add upstream supply chain connections driven by ecosystem services
        upstream_nodes_created = set()
        for service_node in service_nodes[:3]:  # Use first 3 ecosystem services
            for activity_node, activity_full_name in activity_nodes:
                upstream_data = upstream_links[upstream_links['Direct operations (ISIC Group/Class)'] == activity_full_name]
                
                for _, up_row in upstream_data.head(1).iterrows():  # Limit to 1 upstream per service
                    upstream_activity = up_row.get('Upstream tier 1 (ISIC Division/Group/Class)', '')
                    if pd.notna(upstream_activity) and upstream_activity != activity_full_name:
                        upstream_node = f"Upstream: {str(upstream_activity)[:30]}..."
                        
                        if upstream_node not in upstream_nodes_created:
                            G.add_node(upstream_node, node_type="upstream_supply", color="#FFA500", 
                                     size=1400, layer=5.5)  # Between services and downstream
                            upstream_nodes_created.add(upstream_node)
                            
                            # Ecosystem Service → Upstream Supply Chain
                            G.add_edge(service_node, upstream_node, relationship="enables_supply_chain")
                        
                        # Upstream Supply Chain → Economic Activity
                        G.add_edge(upstream_node, activity_node, relationship="supplies")
                        
                        # Upstream Supply Chain → Environmental Pressures (they create pressures too!)
                        relevant_pressures = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'pressure']
                        if relevant_pressures:
                            # Connect to a relevant pressure (e.g., first one)
                            pressure_node = relevant_pressures[0]
                            G.add_edge(upstream_node, pressure_node, 
                                     relationship="supply_chain_pressure",
                                     color="#FF8C00", width=1.5)
        
        # Add downstream supply chain connections driven by ecosystem services
        downstream_nodes_created = set()
        for service_node in service_nodes[:3]:  # Use first 3 ecosystem services
            for activity_node, activity_full_name in activity_nodes:
                downstream_data = downstream_links[downstream_links['Direct operations (ISIC Group/Class)'] == activity_full_name]
                
                for _, down_row in downstream_data.head(1).iterrows():  # Limit to 1 downstream per service
                    downstream_activity = down_row.get('Downstream tier 1 (ISIC Division/Group/Class)', '')
                    if pd.notna(downstream_activity) and downstream_activity not in ['Final consumption category']:
                        downstream_node = f"Downstream: {str(downstream_activity)[:30]}..."
                        
                        if downstream_node not in downstream_nodes_created:
                            G.add_node(downstream_node, node_type="downstream_supply", color="#FF69B4", 
                                     size=1400, layer=5.5)  # Between services and final
                            downstream_nodes_created.add(downstream_node)
                            
                            # Ecosystem Service → Downstream Supply Chain  
                            G.add_edge(service_node, downstream_node, relationship="enables_supply_chain")
                        
                        # Economic Activity → Downstream Supply Chain
                        G.add_edge(activity_node, downstream_node, relationship="supplies_to")
                        
                        # Downstream Supply Chain → Environmental Pressures (they create pressures too!)
                        relevant_pressures = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'pressure']
                        if relevant_pressures and len(relevant_pressures) > 1:
                            # Connect to a different pressure than upstream
                            pressure_node = relevant_pressures[1] if len(relevant_pressures) > 1 else relevant_pressures[0]
                            G.add_edge(downstream_node, pressure_node, 
                                     relationship="supply_chain_pressure",
                                     color="#FF8C00", width=1.5)
        
        return G
        """
        Create a thematic portfolio network with environmental impact pathway.
        
        Args:
            portfolio_theme (str): Theme for the portfolio ('Energy', 'Agriculture', 'Technology', etc.)
            max_activities (int): Maximum number of activities to include
            
        Returns:
            networkx.DiGraph: The portfolio impact network
        """
        G = nx.DiGraph()
        
        # Get pressure links data
        pressure_links = self.all_data['pressure_links']
        
        # Define thematic portfolios with broader keyword matching
        themes = {
            "Energy": ["electric", "power", "petroleum", "coal", "gas", "energy", "solar", "wind", "hydro", "nuclear", "oil", "fuel"],
            "Agriculture": ["growing", "crop", "livestock", "farming", "agricultural", "cultivation", "animal", "plant", "food", "cattle", "poultry"],
            "Technology": ["software", "computer", "data", "information", "tech", "digital", "electronic", "communication", "programming"],
            "Manufacturing": ["manufacturing", "production", "industrial", "machinery", "fabrication", "assembly", "processing"],
            "Mining": ["mining", "extraction", "quarrying", "ore", "mineral", "excavation"],
            "Financial": ["banking", "finance", "insurance", "credit", "investment", "financial"],
            "Healthcare": ["health", "medical", "pharmaceutical", "hospital", "care", "medicine"],
            "Transport": ["transport", "logistics", "shipping", "aviation", "railway", "automotive"],
            "Construction": ["construction", "building", "infrastructure", "real estate", "development"],
            "Retail": ["retail", "wholesale", "trade", "commercial", "sales", "distribution"]
        }
        
        # Add portfolio node
        portfolio_node = f"{portfolio_theme} Portfolio"
        G.add_node(portfolio_node, node_type="portfolio", color="#FFD700", size=4000)
        
        # Find activities matching the theme
        theme_keywords = themes.get(portfolio_theme, ["general"])
        matching_activities = []
        
        for _, row in pressure_links.iterrows():
            activity_name = str(row['ISIC Class']) if pd.notna(row['ISIC Class']) else ""
            division_name = str(row['ISIC Division']) if pd.notna(row['ISIC Division']) else ""
            
            # Check both class and division for matches
            combined_text = f"{activity_name} {division_name}".lower()
            if any(keyword in combined_text for keyword in theme_keywords):
                matching_activities.append((activity_name, row))
        
        # Remove duplicates and limit
        unique_activities = []
        seen_activities = set()
        for activity_name, row in matching_activities:
            if activity_name not in seen_activities and pd.notna(activity_name) and activity_name.strip():
                unique_activities.append((activity_name, row))
                seen_activities.add(activity_name)
                if len(unique_activities) >= max_activities:
                    break
        
        # Add activities and their environmental impacts
        for activity_name, activity_row in unique_activities:
            # Clean activity name
            clean_name = activity_name[:45] + "..." if len(activity_name) > 45 else activity_name
            activity_node = f"Activity: {clean_name}"
            
            # Add activity node
            G.add_node(activity_node, node_type="activity", color="#00FF7F", size=2500, full_name=activity_name)
            G.add_edge(portfolio_node, activity_node)
            
            # Add pressures for this activity
            pressure_categories = [
                'Emissions of GHG', 'Area of land use', 'Volume of water use', 
                'Emissions of toxic soil and water pollutants', 
                'Emissions of non-GHG air pollutants',
                'Generation and release of solid waste',
                'Other biotic resource extraction (e.g. fish, timber)'
            ]
            
            for pressure_cat in pressure_categories:
                if (pressure_cat in activity_row and 
                    pd.notna(activity_row[pressure_cat]) and 
                    activity_row[pressure_cat] not in ['N/A', 'ND']):
                    
                    pressure_node = f"Pressure: {pressure_cat}"
                    
                    # Add pressure node if not exists
                    if pressure_node not in G.nodes():
                        G.add_node(pressure_node, node_type="pressure", color="#FF6B35", size=1800)
                    
                    # Add edge with materiality
                    materiality = activity_row[pressure_cat]
                    edge_colors = {
                        'VH': '#FF0000', 'H': '#FF4500', 'M': '#FFD700', 
                        'L': '#87CEEB', 'VL': '#D3D3D3'
                    }
                    edge_color = edge_colors.get(materiality, '#808080')
                    edge_width = {'VH': 5, 'H': 4, 'M': 3, 'L': 2, 'VL': 1}.get(materiality, 2)
                    
                    G.add_edge(activity_node, pressure_node, 
                             materiality=materiality, color=edge_color, width=edge_width)
        
        # Add mechanisms of change
        mechanisms = {
            "Climate change": ["ghg", "emission"],
            "Water degradation": ["water"],
            "Habitat destruction": ["land", "biotic"],
            "Pollution effects": ["toxic", "pollutant", "solid waste"]
        }
        
        pressure_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'pressure']
        
        for mechanism_name, keywords in mechanisms.items():
            mech_node = f"Mechanism: {mechanism_name}"
            G.add_node(mech_node, node_type="mechanism", color="#DC143C", size=1500)
            
            # Connect relevant pressures to this mechanism
            for pressure_node in pressure_nodes:
                if any(keyword in pressure_node.lower() for keyword in keywords):
                    G.add_edge(pressure_node, mech_node, impact_type="environmental_change")
        
        # Add ecosystem components
        components = ["Atmosphere", "Water systems", "Terrestrial ecosystems", "Biodiversity"]
        
        mechanism_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'mechanism']
        
        for component in components:
            comp_node = f"Component: {component}"
            G.add_node(comp_node, node_type="component", color="#228B22", size=1200)
            
            # Connect mechanisms to relevant components
            for mech_node in mechanism_nodes:
                if "Climate" in mech_node and component == "Atmosphere":
                    G.add_edge(mech_node, comp_node)
                elif "Water" in mech_node and component == "Water systems":
                    G.add_edge(mech_node, comp_node)
                elif "Habitat" in mech_node and component == "Terrestrial ecosystems":
                    G.add_edge(mech_node, comp_node)
                elif "Pollution" in mech_node and component == "Biodiversity":
                    G.add_edge(mech_node, comp_node)
        
        return G
    
    def visualize_portfolio_network(self, network, portfolio_name, figsize=(16, 12), 
                                  dark_theme=True, save_path=None):
        """
        Visualize the portfolio network with professional styling.
        
        Args:
            network (networkx.DiGraph): The network to visualize
            portfolio_name (str): Name of the portfolio for the title
            figsize (tuple): Figure size
            dark_theme (bool): Use dark theme if True
            save_path (str): Path to save the figure (optional)
            
        Returns:
            tuple: (figure, axis) objects
        """
        if dark_theme:
            plt.style.use('dark_background')
            bg_color = 'black'
            text_color = 'white'
        else:
            plt.style.use('default')
            bg_color = 'white'
            text_color = 'black'
        
        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        # Create clustered layout by node type
        pos = self._create_clustered_layout(network)
        
        # Get node attributes
        node_colors = []
        node_sizes = []
        node_labels = {}
        
        for node in network.nodes():
            node_data = network.nodes[node]
            node_colors.append(node_data.get('color', '#CCCCCC'))
            node_sizes.append(node_data.get('size', 800))
            
            # Create clean labels with prefix removal
            if node_data.get('node_type') == 'portfolio':
                node_labels[node] = node
            elif node_data.get('node_type') == 'activity':
                label = node.replace('Activity: ', '')
                node_labels[node] = label[:30] + '...' if len(label) > 30 else label
            elif node_data.get('node_type') == 'pressure':
                label = node.replace('Pressure: ', '')
                node_labels[node] = label[:25] + '...' if len(label) > 25 else label
            elif node_data.get('node_type') == 'mechanism':
                node_labels[node] = node.replace('Mechanism: ', '')
            elif node_data.get('node_type') == 'component' or node_data.get('node_type') == 'natural_capital':
                # Clean Natural Capital labels
                label = node.replace('Natural Capital: ', '').replace('Component: ', '')
                node_labels[node] = label[:20] + '...' if len(label) > 20 else label
            elif node_data.get('node_type') == 'ecosystem_service':
                # Clean Ecosystem Service labels
                label = node.replace('Ecosystem Service: ', '')
                node_labels[node] = label[:20] + '...' if len(label) > 20 else label
            elif node_data.get('node_type') in ['upstream_supply', 'downstream_supply']:
                # Clean Supply Chain labels
                label = node.replace('Upstream: ', '').replace('Downstream: ', '')
                node_labels[node] = label[:25] + '...' if len(label) > 25 else label
            else:
                # Generic cleaning for any other node types
                label = str(node)
                for prefix in ['Natural Capital: ', 'Ecosystem Service: ', 'Upstream: ', 'Downstream: ']:
                    label = label.replace(prefix, '')
                node_labels[node] = label[:20] + '...' if len(label) > 20 else label
        
        # Draw edges with different styles based on materiality
        for edge in network.edges():
            edge_data = network.edges[edge]
            edge_color = edge_data.get('color', '#666666')
            edge_width = edge_data.get('width', 1.5)
            materiality = edge_data.get('materiality', '')
            
            # Add alpha for better visibility
            alpha = 0.8 if materiality in ['VH', 'H'] else 0.6
            
            nx.draw_networkx_edges(network, pos, edgelist=[edge], 
                                  edge_color=edge_color, width=edge_width, 
                                  alpha=alpha, arrows=True, arrowsize=20,
                                  arrowstyle='->', connectionstyle='arc3,rad=0.1')
        
        # Draw nodes with better styling
        nx.draw_networkx_nodes(network, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9, 
                              edgecolors=text_color, linewidths=2)
        
        # Add labels with better visibility
        nx.draw_networkx_labels(network, pos, node_labels, font_size=10, 
                               font_color=text_color, font_weight='bold')
        
        # Title and styling
        ax.set_title(f"{portfolio_name}\nEnvironmental Impact Pathway", 
                    fontsize=20, fontweight='bold', color=text_color, pad=30)
        
        # Create legend with theme-appropriate styling  
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700', 
                       markersize=15, label='Portfolio', markeredgecolor=text_color),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00FF7F', 
                       markersize=12, label='Economic Activities', markeredgecolor=text_color),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B35', 
                       markersize=10, label='Environmental Pressures', markeredgecolor=text_color),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#DC143C', 
                       markersize=8, label='Change Mechanisms', markeredgecolor=text_color),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#228B22', 
                       markersize=6, label='Ecosystem Components', markeredgecolor=text_color),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9370DB', 
                       markersize=8, label='Ecosystem Services', markeredgecolor=text_color),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF8C00', 
                       markersize=6, label='Supply Chains (Services-Driven)', markeredgecolor=text_color),
            plt.Line2D([0], [0], color='#FF0000', linewidth=4, label='Very High Impact'),
            plt.Line2D([0], [0], color='#FF4500', linewidth=3, label='High Impact'),
            plt.Line2D([0], [0], color='#FFD700', linewidth=2, label='Medium Impact'),
            plt.Line2D([0], [0], color='#00FF7F', linewidth=3, linestyle='--', 
                       label='Circular Feedback (Services→Activities)'),
            plt.Line2D([0], [0], color='#9370DB', linewidth=2, linestyle=':', 
                       label='Services Enable Supply Chains'),
            plt.Line2D([0], [0], color='#FF8C00', linewidth=2, linestyle='-.', 
                       label='Supply Chains Create Pressures'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                 facecolor=bg_color, edgecolor=text_color, labelcolor=text_color,
                 fontsize=11)
        
        # Remove axes
        ax.set_axis_off()
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, facecolor=bg_color, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def _create_clustered_layout(self, network):
        """
        Create a clustered layout that groups nodes by their type.
        
        Args:
            network (networkx.DiGraph): The network to layout
            
        Returns:
            dict: Position dictionary for nodes
        """
        import numpy as np
        
        # Group nodes by type
        node_groups = {
            'portfolio': [],
            'activity': [],
            'pressure': [],
            'mechanism': [],
            'natural_capital': [],
            'ecosystem_service': [],
            'upstream_supply': [],
            'downstream_supply': []
        }
        
        for node in network.nodes():
            node_type = network.nodes[node].get('node_type', 'other')
            if node_type in node_groups:
                node_groups[node_type].append(node)
            else:
                # Handle alternative type names
                if 'activity' in node_type.lower() or 'Activity' in str(node):
                    node_groups['activity'].append(node)
                elif 'pressure' in node_type.lower() or 'Pressure' in str(node):
                    node_groups['pressure'].append(node)
                elif 'mechanism' in node_type.lower() or 'Mechanism' in str(node):
                    node_groups['mechanism'].append(node)
                elif 'ecosystem' in node_type.lower() or 'Ecosystem' in str(node):
                    node_groups['ecosystem_service'].append(node)
                else:
                    node_groups['activity'].append(node)  # default fallback
        
        pos = {}
        
        # Define cluster centers - Portfolio on far left, flowing right
        cluster_centers = {
            'portfolio': (-6, 0),         # Far left - starting point
            'activity': (-3, 3),          # Top left
            'pressure': (0, 3),           # Top center  
            'mechanism': (3, 2),          # Top right
            'natural_capital': (4, -1),   # Right
            'ecosystem_service': (2, -3), # Bottom right
            'upstream_supply': (-5, -2),  # Bottom left
            'downstream_supply': (-1, -3) # Bottom center
        }
        
        # Position nodes within their clusters with adaptive spacing
        for node_type, nodes in node_groups.items():
            if not nodes:
                continue
                
            center_x, center_y = cluster_centers.get(node_type, (0, 0))
            
            if len(nodes) == 1:
                pos[nodes[0]] = (center_x, center_y)
            else:
                # Use grid layout for clusters with many nodes, circle for few nodes
                if len(nodes) <= 6:
                    # Small clusters: circular arrangement
                    for i, node in enumerate(nodes):
                        angle = 2 * np.pi * i / len(nodes)
                        radius = 0.8 + 0.1 * len(nodes)  # Adaptive radius
                        x = center_x + radius * np.cos(angle)
                        y = center_y + radius * np.sin(angle)
                        pos[node] = (x, y)
                else:
                    # Large clusters: grid arrangement  
                    cols = int(np.ceil(np.sqrt(len(nodes))))
                    rows = int(np.ceil(len(nodes) / cols))
                    for i, node in enumerate(nodes):
                        row = i // cols
                        col = i % cols
                        x = center_x + (col - cols/2) * 0.5
                        y = center_y + (row - rows/2) * 0.5  
                        pos[node] = (x, y)
        
        return pos
    
    def create_portfolio_networks(self, themes=None):
        """
        Create networks for multiple portfolio themes.
        
        Args:
            themes (list): List of theme names. If None, uses default themes.
            
        Returns:
            dict: Dictionary of theme -> network mappings
        """
        if themes is None:
            themes = ["Energy", "Agriculture", "Technology", "Manufacturing", "Financial"]
        
        networks = {}
        
        for theme in themes:
            try:
                network = self.create_thematic_portfolio_network(theme)
                networks[theme] = network
                
                # Get network statistics
                node_types = {}
                for node, data in network.nodes(data=True):
                    node_type = data.get('node_type', 'unknown')
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                
                print(f"✅ {theme} Portfolio: {len(network.nodes())} nodes, {len(network.edges())} edges")
                if len(network.nodes()) > 5:
                    print(f"   Structure: {node_types}")
                else:
                    print(f"   ⚠️  Limited data available for {theme} theme")
                
            except Exception as e:
                print(f"❌ Error creating {theme} portfolio: {str(e)}")
                continue
        
        self.portfolio_networks = networks
        return networks
    
    def get_impact_pathway(self, network, portfolio_theme):
        """
        Extract and format the impact pathway from portfolio to ecosystem components.
        
        Args:
            network (networkx.DiGraph): The portfolio network
            portfolio_theme (str): Theme name
            
        Returns:
            dict: Structured impact pathway data
        """
        pathway = {
            'theme': portfolio_theme,
            'activities': [],
            'pressures': [],
            'mechanisms': [],
            'components': []
        }
        
        # Find the portfolio node
        portfolio_node = f"{portfolio_theme} Portfolio"
        
        if portfolio_node not in network.nodes():
            return pathway
        
        # Trace the pathway
        for activity in network.successors(portfolio_node):
            activity_data = {
                'name': activity.replace('Activity: ', ''),
                'pressures': []
            }
            
            for pressure in network.successors(activity):
                pressure_data = {
                    'name': pressure.replace('Pressure: ', ''),
                    'materiality': network.edges[activity, pressure].get('materiality', 'Unknown'),
                    'mechanisms': []
                }
                
                for mechanism in network.successors(pressure):
                    mechanism_data = {
                        'name': mechanism.replace('Mechanism: ', ''),
                        'components': []
                    }
                    
                    for component in network.successors(mechanism):
                        component_data = {
                            'name': component.replace('Component: ', '')
                        }
                        mechanism_data['components'].append(component_data)
                    
                    pressure_data['mechanisms'].append(mechanism_data)
                
                activity_data['pressures'].append(pressure_data)
            
            pathway['activities'].append(activity_data)
        
        # Also collect unique pressures, mechanisms, components
        pathway['pressures'] = list(set([n.replace('Pressure: ', '') for n in network.nodes() 
                                       if network.nodes[n].get('node_type') == 'pressure']))
        pathway['mechanisms'] = list(set([n.replace('Mechanism: ', '') for n in network.nodes() 
                                        if network.nodes[n].get('node_type') == 'mechanism']))
        pathway['components'] = list(set([n.replace('Component: ', '') for n in network.nodes() 
                                        if network.nodes[n].get('node_type') == 'component']))
        
        return pathway
    
    def get_available_themes(self):
        """Get list of available portfolio themes."""
        return ["Energy", "Agriculture", "Technology", "Manufacturing", "Mining", 
                "Financial", "Healthcare", "Transport", "Construction", "Retail"]
    
    def reset_style(self):
        """Reset matplotlib style to default."""
        plt.style.use('default')


class PortfolioDashboard:
    """Interactive dashboard for portfolio environmental impact analysis"""
    
    def __init__(self, engine):
        self.engine = engine
        self.current_network = None
        self.current_theme = "Energy"
        self.portfolio_collection = {}
        
        # Dashboard state
        self.visualization_options = {
            'theme': 'dark',
            'layout': 'spring',
            'show_legend': True,
            'node_size_factor': 1.0,
            'edge_width_factor': 1.0
        }
    
    def initialize_portfolios(self, themes=None):
        """Initialize multiple portfolio themes"""
        if themes is None:
            themes = ["Energy", "Agriculture", "Technology", "Financial", "Manufacturing"]
        
        print("🔨 Creating demonstration portfolios...")
        self.portfolio_collection = {}
        
        for theme in themes:
            print(f"   Creating {theme} Portfolio...")
            network = self.engine.create_comprehensive_impact_network(theme, max_activities=10)
            
            if network and len(network.nodes()) > 5:
                self.portfolio_collection[theme] = network
                print(f"   ✅ {theme} portfolio ready!")
            else:
                print(f"   ⚠️  Limited data for {theme} theme")
        
        print(f"📊 Created {len(self.portfolio_collection)} portfolios")
        return self.portfolio_collection
    
    def create_portfolio_network(self, theme="Energy", max_activities=12):
        """Create and store a portfolio network for the given theme"""
        print(f"🔨 Creating {theme} portfolio network...")
        
        try:
            network = self.engine.create_comprehensive_impact_network(theme, max_activities)
            self.current_network = network
            self.current_theme = theme
            
            # Store in collection
            self.portfolio_collection[theme] = network
            
            # Get network statistics
            node_types = {}
            for node, data in network.nodes(data=True):
                node_type = data.get('node_type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            print(f"✅ {theme} network created successfully!")
            print(f"   📊 {len(network.nodes())} nodes, {len(network.edges())} edges")
            print(f"   🔗 Structure: {node_types}")
            
            return network
            
        except Exception as e:
            print(f"❌ Error creating {theme} network: {e}")
            return None
    
    def visualize_portfolio(self, theme, dark_theme=True, figsize=(16, 12)):
        """Visualize a specific portfolio theme"""
        if theme not in self.portfolio_collection:
            print(f"⚠️  {theme} portfolio not available. Creating now...")
            self.create_portfolio_network(theme)
        
        if theme not in self.portfolio_collection:
            print(f"❌ Could not create {theme} portfolio")
            return None
        
        print(f"🎨 Visualizing {theme} portfolio...")
        
        try:
            network = self.portfolio_collection[theme]
            self.current_network = network
            self.current_theme = theme
            
            fig, ax = self.engine.visualize_portfolio_network(
                network, 
                theme,
                figsize=figsize,
                dark_theme=dark_theme
            )
            
            plt.show()
            return fig, ax
            
        except Exception as e:
            print(f"❌ Error visualizing {theme} network: {e}")
            return None
    
    def analyze_impact_pathway(self, theme=None):
        """Analyze and display the impact pathway for a portfolio"""
        if theme is None:
            theme = self.current_theme
        
        if theme not in self.portfolio_collection:
            print(f"⚠️  {theme} portfolio not available")
            return None
        
        print(f"🔍 Analyzing {theme} impact pathway...")
        
        try:
            network = self.portfolio_collection[theme]
            pathway = self.engine.get_impact_pathway(network, theme)
            
            print(f"\n📈 IMPACT PATHWAY ANALYSIS")
            print("=" * 50)
            print(f"🎯 Theme: {pathway['theme']}")
            print(f"🏭 Economic Activities: {len(pathway['activities'])}")
            print(f"⚠️  Environmental Pressures: {len(pathway['pressures'])}")
            print(f"🔄 Change Mechanisms: {len(pathway['mechanisms'])}")
            print(f"🌿 Ecosystem Components: {len(pathway['components'])}")
            
            # Show detailed pathway for first activity
            if pathway['activities']:
                print(f"\n💡 Sample Impact Chain:")
                activity = pathway['activities'][0]
                print(f"   📍 Activity: {activity['name']}")
                
                if activity['pressures']:
                    pressure = activity['pressures'][0]
                    print(f"   ⚠️  → Pressure: {pressure['name']} (Impact: {pressure['materiality']})")
                    
                    if pressure['mechanisms']:
                        mechanism = pressure['mechanisms'][0]
                        print(f"   🔄 → Mechanism: {mechanism['name']}")
                        
                        if mechanism['components']:
                            component = mechanism['components'][0]
                            print(f"   🌿 → Component: {component['name']}")
            
            return pathway
            
        except Exception as e:
            print(f"❌ Error analyzing pathway: {e}")
            return None
    
    def showcase_portfolios(self, themes=None):
        """Create a showcase of multiple portfolio visualizations"""
        if themes is None:
            # Select top portfolios with good data
            themes = []
            for theme, network in self.portfolio_collection.items():
                if len(network.nodes()) >= 15:  # Only well-populated networks
                    themes.append(theme)
                if len(themes) >= 3:
                    break
        
        print(f"🎨 PORTFOLIO VISUALIZATION SHOWCASE")
        print("=" * 50)
        print(f"📋 Showcasing: {themes}")
        
        # Create individual visualizations
        for theme in themes:
            print(f"\n🎯 {theme.upper()} PORTFOLIO")
            print("-" * 30)
            
            # Visualize with dark theme
            self.visualize_portfolio(theme, dark_theme=True, figsize=(18, 12))
            
            # Add portfolio analysis
            self.analyze_impact_pathway(theme)
            
            print(f"✅ {theme} visualization complete!\n")
        
        print("🏆 Portfolio Showcase Complete!")
    
    def create_comprehensive_portfolio(self, theme="Energy", max_activities=4):
        """Create comprehensive circular portfolio network"""
        print(f"🔨 Creating comprehensive {theme} portfolio network...")
        
        try:
            network = self.engine.create_comprehensive_impact_network(theme, max_activities)
            self.current_network = network
            self.current_theme = theme
            
            # Store in collection
            self.portfolio_collection[f"{theme}_Comprehensive"] = network
            
            # Get network statistics
            node_types = {}
            layer_counts = {}
            for node, data in network.nodes(data=True):
                node_type = data.get('node_type', 'unknown')
                layer = data.get('layer', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            
            print(f"✅ Comprehensive {theme} network created successfully!")
            print(f"   📊 {len(network.nodes())} nodes, {len(network.edges())} edges")
            print(f"   🔗 Node types: {node_types}")
            print(f"   🎚️  Layers: {sorted(layer_counts.items())}")
            
            # Detect circular dependencies
            circular_edges = []
            for edge in network.edges(data=True):
                relationship = edge[2].get('relationship', '')
                if relationship == 'supports':
                    circular_edges.append(edge)
            
            if circular_edges:
                print(f"   🔄 Circular dependencies detected: {len(circular_edges)} feedback loops")
            
            return network
            
        except Exception as e:
            print(f"❌ Error creating comprehensive {theme} network: {e}")
            return None
    
    def visualize_comprehensive_portfolio(self, theme, figsize=(20, 14)):
        """Visualize comprehensive circular portfolio network with improved layout"""
        comprehensive_key = f"{theme}_Comprehensive"
        
        if comprehensive_key not in self.portfolio_collection:
            print(f"⚠️  Comprehensive {theme} portfolio not available. Creating now...")
            self.create_comprehensive_portfolio(theme)
        
        if comprehensive_key not in self.portfolio_collection:
            print(f"❌ Could not create comprehensive {theme} portfolio")
            return None
        
        print(f"🎨 Visualizing comprehensive {theme} portfolio...")
        
        try:
            network = self.portfolio_collection[comprehensive_key]
            self.current_network = network
            self.current_theme = f"{theme}_Comprehensive"
            
            # Use portfolio network visualization
            fig, ax = self.engine.visualize_portfolio_network(
                network, 
                f"Comprehensive {theme} Portfolio",
                figsize=figsize,
                dark_theme=True
            )
            
            plt.show()
            return fig, ax
            
        except Exception as e:
            print(f"❌ Error visualizing comprehensive {theme} network: {e}")
            return None
    
    def get_available_portfolios(self):
        """Get list of available portfolio themes"""
        return list(self.portfolio_collection.keys())


class PortfolioAnalyzer:
    """Advanced analysis tools for portfolio environmental impacts"""
    
    def __init__(self, portfolio_collection):
        self.portfolios = portfolio_collection
    
    def calculate_impact_scores(self, network):
        """Calculate numerical impact scores for a portfolio"""
        if network is None:
            return {}
        
        scores = {
            'total_nodes': len(network.nodes()),
            'total_edges': len(network.edges()),
            'high_impact_edges': 0,
            'materiality_score': 0,
            'risk_level': 'Low'
        }
        
        # Count high impact edges and calculate materiality
        materiality_weights = {'VH': 5, 'H': 4, 'M': 3, 'L': 2, 'VL': 1}
        total_materiality = 0
        
        for edge in network.edges(data=True):
            materiality = edge[2].get('materiality', 'L')
            weight = materiality_weights.get(materiality, 2)
            total_materiality += weight
            
            if materiality in ['VH', 'H']:
                scores['high_impact_edges'] += 1
        
        scores['materiality_score'] = total_materiality / max(len(network.edges()), 1)
        
        # Determine risk level
        if scores['materiality_score'] >= 4:
            scores['risk_level'] = 'High'
        elif scores['materiality_score'] >= 3:
            scores['risk_level'] = 'Medium'
        else:
            scores['risk_level'] = 'Low'
        
        return scores
    
    def compare_portfolios(self):
        """Compare environmental impact across all portfolios"""
        print("📊 PORTFOLIO ENVIRONMENTAL IMPACT COMPARISON")
        print("=" * 60)
        
        comparison_data = []
        
        for theme, network in self.portfolios.items():
            scores = self.calculate_impact_scores(network)
            comparison_data.append({
                'Portfolio': theme,
                'Nodes': scores['total_nodes'],
                'Edges': scores['total_edges'],
                'High Impact': scores['high_impact_edges'],
                'Materiality Score': round(scores['materiality_score'], 2),
                'Risk Level': scores['risk_level']
            })
        
        # Create comparison dataframe
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Materiality Score', ascending=False)
        
        print(df.to_string(index=False))
        
        # Identify highest risk portfolio
        if len(df) > 0:
            highest_risk = df.iloc[0]['Portfolio']
            print(f"\n⚠️  HIGHEST RISK PORTFOLIO: {highest_risk}")
            print(f"   💡 Consider ESG-focused alternatives or impact mitigation strategies")
        
        return df
    
    def generate_risk_report(self, theme):
        """Generate detailed risk report for a specific portfolio"""
        if theme not in self.portfolios:
            print(f"❌ Portfolio '{theme}' not found")
            return
        
        network = self.portfolios[theme]
        scores = self.calculate_impact_scores(network)
        
        print(f"📋 ENVIRONMENTAL RISK REPORT: {theme.upper()} PORTFOLIO")
        print("=" * 60)
        print(f"🎯 Portfolio Theme: {theme}")
        print(f"📊 Network Size: {scores['total_nodes']} nodes, {scores['total_edges']} connections")
        print(f"⚠️  High Impact Connections: {scores['high_impact_edges']}")
        print(f"📈 Materiality Score: {scores['materiality_score']:.2f}/5.0")
        print(f"🚨 Risk Level: {scores['risk_level']}")
        
        # Risk recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if scores['risk_level'] == 'High':
            print("   🔴 HIGH RISK - Consider portfolio rebalancing")
            print("   📉 Reduce exposure to high-impact activities")
            print("   🌱 Increase allocation to sustainable alternatives")
        elif scores['risk_level'] == 'Medium':
            print("   🟡 MEDIUM RISK - Monitor environmental developments")
            print("   📊 Regular impact assessment recommended")
            print("   🔍 Consider ESG screening for new investments")
        else:
            print("   🟢 LOW RISK - Maintain current allocation")
            print("   📈 Opportunity for sustainable growth")
            print("   🌟 Consider expanding similar low-impact investments")
        
        return scores


class PortfolioExporter:
    """Handle exporting visualizations and analysis results"""
    
    def __init__(self, dashboard, analyzer):
        self.dashboard = dashboard
        self.analyzer = analyzer
        self.export_dir = Path("portfolio_exports")
        self.export_dir.mkdir(exist_ok=True)
    
    def export_visualization(self, theme, file_format='png', dpi=300):
        """Export portfolio visualization to file"""
        if theme not in self.dashboard.portfolio_collection:
            print(f"❌ Portfolio '{theme}' not available")
            return None
        
        print(f"💾 Exporting {theme} visualization...")
        
        # Create visualization
        network = self.dashboard.portfolio_collection[theme]
        fig, ax = self.dashboard.engine.visualize_portfolio_network(
            network,
            theme,
            figsize=(20, 16),
            dark_theme=True
        )
        
        # Save file
        filename = f"{theme.lower()}_portfolio_network.{file_format}"
        filepath = self.export_dir / filename
        
        plt.savefig(filepath, format=file_format, dpi=dpi, 
                   facecolor='black', bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {filepath}")
        return filepath
    
    def export_analysis_report(self, theme):
        """Export detailed analysis report to text file"""
        if theme not in self.dashboard.portfolio_collection:
            print(f"❌ Portfolio '{theme}' not available")
            return None
        
        print(f"📄 Exporting {theme} analysis report...")
        
        # Generate analysis data
        network = self.dashboard.portfolio_collection[theme]
        scores = self.analyzer.calculate_impact_scores(network)
        pathway = self.dashboard.engine.get_impact_pathway(network, theme)
        
        # Create report content
        report_lines = [
            f"PORTFOLIO ENVIRONMENTAL IMPACT ANALYSIS REPORT",
            f"=" * 50,
            f"",
            f"Portfolio Theme: {theme}",
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"NETWORK STRUCTURE",
            f"-" * 20,
            f"Total Nodes: {scores['total_nodes']}",
            f"Total Edges: {scores['total_edges']}",
            f"High Impact Connections: {scores['high_impact_edges']}",
            f"",
            f"ENVIRONMENTAL IMPACT ASSESSMENT",
            f"-" * 35,
            f"Materiality Score: {scores['materiality_score']:.2f}/5.0",
            f"Risk Level: {scores['risk_level']}",
            f"",
            f"IMPACT PATHWAY BREAKDOWN",
            f"-" * 25,
            f"Economic Activities: {len(pathway['activities'])}",
            f"Environmental Pressures: {len(pathway['pressures'])}",
            f"Change Mechanisms: {len(pathway['mechanisms'])}",
            f"Ecosystem Components: {len(pathway['components'])}",
            f"",
            f"DETAILED PRESSURES",
            f"-" * 18,
        ]
        
        for pressure in pathway['pressures']:
            report_lines.append(f"• {pressure}")
        
        report_lines.extend([
            f"",
            f"MECHANISMS OF CHANGE",
            f"-" * 20,
        ])
        
        for mechanism in pathway['mechanisms']:
            report_lines.append(f"• {mechanism}")
        
        report_lines.extend([
            f"",
            f"AFFECTED ECOSYSTEM COMPONENTS",
            f"-" * 30,
        ])
        
        for component in pathway['components']:
            report_lines.append(f"• {component}")
        
        # Risk recommendations
        report_lines.extend([
            f"",
            f"RISK ASSESSMENT & RECOMMENDATIONS",
            f"-" * 35,
        ])
        
        if scores['risk_level'] == 'High':
            report_lines.extend([
                f"⚠️  HIGH RISK PORTFOLIO",
                f"• Consider immediate portfolio rebalancing",
                f"• Reduce exposure to high-impact economic activities",
                f"• Increase allocation to sustainable alternatives",
                f"• Implement ESG screening for new investments",
            ])
        elif scores['risk_level'] == 'Medium':
            report_lines.extend([
                f"⚠️  MEDIUM RISK PORTFOLIO", 
                f"• Monitor environmental developments closely",
                f"• Conduct quarterly impact assessments",
                f"• Consider gradual shift to lower-impact alternatives",
                f"• Implement sustainability criteria for new investments",
            ])
        else:
            report_lines.extend([
                f"✅ LOW RISK PORTFOLIO",
                f"• Maintain current sustainable allocation",
                f"• Opportunity for growth in similar low-impact investments",
                f"• Consider leadership role in sustainable finance",
                f"• Share best practices with industry peers",
            ])
        
        # Save report
        filename = f"{theme.lower()}_analysis_report.txt"
        filepath = self.export_dir / filename
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Saved: {filepath}")
        return filepath
    
    def export_all_portfolios(self):
        """Export visualizations and reports for all portfolios"""
        print("📦 EXPORTING ALL PORTFOLIOS")
        print("=" * 40)
        
        exported_files = []
        
        for theme in self.dashboard.portfolio_collection.keys():
            print(f"\n📋 Exporting {theme}...")
            
            # Export visualization
            viz_file = self.export_visualization(theme)
            if viz_file:
                exported_files.append(viz_file)
            
            # Export analysis report
            report_file = self.export_analysis_report(theme)
            if report_file:
                exported_files.append(report_file)
        
        print(f"\n✅ Export Complete!")
        print(f"📁 Files saved to: {self.export_dir}")
        print(f"📊 Total files: {len(exported_files)}")
        
        return exported_files


def create_full_system(data_directory="ENCORE_data"):
    """
    Create complete portfolio analysis system with all components.
    
    Returns:
        tuple: (engine, dashboard, analyzer, exporter) - fully configured system
    """
    print("🚀 Initializing Complete Portfolio Analysis System...")
    print("=" * 60)
    
    # Initialize engine
    print("📊 Loading ENCORE environmental database...")
    engine = PortfolioImpactEngine(data_directory)
    print(f"✅ Engine initialized with {len(engine.all_data)} datasets")
    
    # Initialize dashboard
    print("🎛️  Setting up dashboard...")
    dashboard = PortfolioDashboard(engine)
    portfolio_collection = dashboard.initialize_portfolios()
    print(f"✅ Dashboard ready with {len(portfolio_collection)} portfolios")
    
    # Initialize analyzer
    print("🔬 Configuring impact analyzer...")
    analyzer = PortfolioAnalyzer(portfolio_collection)
    print("✅ Analyzer ready for risk assessment")
    
    # Initialize exporter
    print("💾 Setting up export system...")
    exporter = PortfolioExporter(dashboard, analyzer)
    print(f"✅ Exporter ready (output: {exporter.export_dir})")
    
    print("\n🎉 COMPLETE SYSTEM READY!")
    print("=" * 60)
    print("📋 Available Commands:")
    print("   dashboard.visualize_portfolio('Energy')")
    print("   dashboard.analyze_impact_pathway('Technology')")
    print("   dashboard.showcase_portfolios()")
    print("   dashboard.create_comprehensive_portfolio('Agriculture')  # NEW!")
    print("   dashboard.visualize_comprehensive_portfolio('Energy')    # NEW!")
    print("   analyzer.compare_portfolios()")
    print("   analyzer.generate_risk_report('Financial')")
    print("   exporter.export_all_portfolios()")
    
    print("\n🌍 NEW: Comprehensive Circular Networks Available!")
    print("   These show the complete ENCORE circular system:")
    print("   Economic Activities → Pressures → Mechanisms → Natural Capital")
    print("   → Ecosystem Services → Economic Activities (circular feedback)")
    print("   Plus: Supply chain links (upstream/downstream)")
    
    return engine, dashboard, analyzer, exporter