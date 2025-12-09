from flask import Flask, render_template, jsonify, send_from_directory, request
import os
import sys
import json
import math

# Add the parent directory to the path to import portfolio_impact_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from portfolio_impact_engine import create_full_system
    ENGINE_AVAILABLE = True
    # Initialize the portfolio system
    engine, dashboard, analyzer, exporter = create_full_system()
    print("✅ Portfolio engine initialized successfully!")
except ImportError as e:
    ENGINE_AVAILABLE = False
    print(f"⚠️  Portfolio engine not available: {e}")
    print("🔄 Running in demo mode with mock data")

app = Flask(__name__, static_folder='.')

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('.', filename)

@app.route('/api/portfolio/<portfolio_type>')
def get_portfolio_data(portfolio_type):
    """Get portfolio network data"""
    try:
        max_activities = int(request.args.get('max_activities', 12))
        node_multiplier = float(request.args.get('node_multiplier', 1.3))
        
        if ENGINE_AVAILABLE:
            # Use real portfolio engine
            if portfolio_type in dashboard.portfolio_collection:
                network = dashboard.portfolio_collection[portfolio_type]
            else:
                network = engine.create_comprehensive_impact_network(
                    portfolio_type, 
                    max_activities=max_activities
                )
                dashboard.portfolio_collection[portfolio_type] = network
            
            # Get clustered layout positions
            pos = engine._create_clustered_layout(network)
            
            # Convert NetworkX graph to JSON
            nodes = []
            edges = []
            
            for node in network.nodes():
                node_data = network.nodes[node]
                node_type = node_data.get('node_type', 'other')
                
                # Clean label
                label = str(node)
                for prefix in ['Activity: ', 'Pressure: ', 'Mechanism: ', 'Natural Capital: ', 
                              'Ecosystem Service: ', 'Upstream: ', 'Downstream: ', 'Component: ']:
                    label = label.replace(prefix, '')
                label = label[:25] + '...' if len(label) > 25 else label
                
                x, y = pos[node]
                nodes.append({
                    'id': str(node),
                    'label': label,
                    'type': node_type,
                    'x': x,
                    'y': y
                })
            
            for edge in network.edges():
                edges.append({
                    'source': str(edge[0]),
                    'target': str(edge[1])
                })
            
            return jsonify({
                'success': True,
                'nodes': nodes,
                'edges': edges,
                'portfolio_type': portfolio_type
            })
        
        else:
            # Return mock data
            return jsonify({
                'success': True,
                'nodes': generate_mock_nodes(portfolio_type, max_activities),
                'edges': generate_mock_edges(portfolio_type),
                'portfolio_type': portfolio_type,
                'demo_mode': True
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def generate_mock_nodes(portfolio_type, max_activities):
    """Generate mock nodes for demo mode"""
    mock_data = {
        'Energy': {
            'activities': ['Oil Extraction', 'Natural Gas', 'Solar Power', 'Wind Energy', 'Nuclear Power', 'Coal Mining'],
            'pressures': ['GHG Emissions', 'Water Pollution', 'Air Pollution', 'Land Use Change'],
            'mechanisms': ['Climate Change', 'Ocean Acidification', 'Habitat Destruction'],
            'natural_capital': ['Forests', 'Freshwater', 'Marine Systems', 'Atmosphere'],
            'ecosystem_services': ['Carbon Storage', 'Water Regulation', 'Climate Regulation']
        }
    }
    
    data = mock_data.get(portfolio_type, mock_data['Energy'])
    nodes = []
    
    # Portfolio center
    nodes.append({
        'id': f'{portfolio_type} Portfolio',
        'label': portfolio_type,
        'type': 'portfolio',
        'x': 0,
        'y': 0
    })
    
    # Activities in circle
    activities = data['activities'][:max_activities]
    for i, activity in enumerate(activities):
        angle = (i / len(activities)) * 2 * 3.14159
        nodes.append({
            'id': activity,
            'label': activity,
            'type': 'activity',
            'x': 2 * math.cos(angle),
            'y': 2 * math.sin(angle)
        })
    
    # Add other node types...
    return nodes

def generate_mock_edges(portfolio_type):
    """Generate mock edges for demo mode"""
    return [
        {'source': f'{portfolio_type} Portfolio', 'target': 'Oil Extraction'},
        {'source': f'{portfolio_type} Portfolio', 'target': 'Solar Power'},
        {'source': 'Oil Extraction', 'target': 'GHG Emissions'},
        {'source': 'Solar Power', 'target': 'Land Use Change'},
    ]

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'engine_available': ENGINE_AVAILABLE,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🚀 Starting Portfolio Impact Analyzer Web Server...")
    print("📍 Server will be available at: http://localhost:5000")
    print("🔗 Open your browser and navigate to the URL above")
    print("⚡ Press Ctrl+C to stop the server")
    
    # Run the Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)