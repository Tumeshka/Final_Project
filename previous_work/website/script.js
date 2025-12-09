// Global variables and configuration
let currentPortfolioData = null;
let plotInstance = null;

// Portfolio color scheme
const NODE_COLORS = {
    'portfolio': '#FFD700',
    'activity': '#00FF7F',
    'pressure': '#FF6B35',
    'mechanism': '#DC143C',
    'natural_capital': '#228B22',
    'ecosystem_service': '#9370DB',
    'upstream_supply': '#FF8C00',
    'downstream_supply': '#FF8C00'
};

// DOM Elements
const portfolioSelect = document.getElementById('portfolioSelect');
const showLabelsToggle = document.getElementById('showLabels');
const generateBtn = document.getElementById('generateBtn');
const exportBtn = document.getElementById('exportBtn');
const fullscreenBtn = document.getElementById('fullscreenBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const plotlyChart = document.getElementById('plotlyChart');

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    generateVisualization(); // Generate initial visualization
});

function initializeEventListeners() {
    // Label toggle
    showLabelsToggle.addEventListener('change', function() {
        if (currentPortfolioData) updateVisualization();
    });

    // Action buttons
    generateBtn.addEventListener('click', generateVisualization);
    portfolioSelect.addEventListener('change', generateVisualization);
    
    exportBtn.addEventListener('click', exportVisualization);
    fullscreenBtn.addEventListener('click', toggleFullscreen);
}

// Enhanced mock data generator
function generateMockPortfolioData(portfolioType) {
    const portfolioData = {
        'Energy': {
            activities: ['Oil Extraction', 'Natural Gas Production', 'Solar Photovoltaic', 'Wind Energy', 'Nuclear Power', 'Coal Mining', 'Hydroelectric', 'Geothermal', 'Biomass Energy', 'Energy Storage', 'Grid Infrastructure', 'Petroleum Refining'],
            pressures: ['GHG Emissions', 'Water Pollution', 'Air Pollution', 'Land Use Change', 'Noise Pollution', 'Thermal Pollution', 'Chemical Pollution', 'Habitat Fragmentation'],
            mechanisms: ['Climate Change', 'Ocean Acidification', 'Habitat Destruction', 'Species Loss', 'Soil Degradation', 'Water Stress'],
            natural_capital: ['Forests', 'Freshwater Systems', 'Marine Systems', 'Atmosphere', 'Soil Resources', 'Mineral Deposits'],
            ecosystem_services: ['Carbon Storage', 'Water Regulation', 'Climate Regulation', 'Biodiversity Conservation', 'Air Purification']
        },
        'Agriculture': {
            activities: ['Crop Production', 'Livestock Farming', 'Forestry', 'Aquaculture', 'Food Processing', 'Fertilizer Application', 'Pest Management', 'Irrigation', 'Land Preparation', 'Harvesting'],
            pressures: ['Pesticide Use', 'Water Consumption', 'Soil Erosion', 'Deforestation', 'Eutrophication', 'Salinization', 'Compaction', 'Chemical Runoff'],
            mechanisms: ['Habitat Loss', 'Pollution Impact', 'Overexploitation', 'Species Decline', 'Ecosystem Disruption'],
            natural_capital: ['Soil Systems', 'Pollinators', 'Freshwater', 'Forests', 'Grasslands', 'Wetlands'],
            ecosystem_services: ['Pollination', 'Soil Formation', 'Water Purification', 'Food Production', 'Carbon Sequestration']
        },
        'Technology': {
            activities: ['Electronics Manufacturing', 'Software Development', 'Data Centers', 'Telecommunications', 'Research & Development', 'E-waste Processing', 'Cloud Computing', 'AI Systems'],
            pressures: ['Energy Consumption', 'Material Extraction', 'Electronic Waste', 'Water Usage', 'Chemical Pollution', 'Electromagnetic Radiation'],
            mechanisms: ['Resource Depletion', 'Pollution Generation', 'Energy Demand', 'Waste Accumulation'],
            natural_capital: ['Rare Earth Elements', 'Water Resources', 'Energy Sources', 'Atmosphere', 'Land Resources'],
            ecosystem_services: ['Material Provision', 'Energy Provision', 'Waste Processing', 'Water Regulation']
        },
        'Financial': {
            activities: ['Commercial Banking', 'Insurance Services', 'Investment Management', 'Asset Management', 'Securities Trading', 'Fintech Operations', 'Risk Assessment'],
            pressures: ['Indirect GHG Emissions', 'Paper Consumption', 'Energy Use', 'Business Travel', 'Digital Infrastructure'],
            mechanisms: ['Financing Environmental Impact', 'Investment Risk Exposure', 'Carbon Portfolio Risk'],
            natural_capital: ['Forest Resources', 'Energy Resources', 'Digital Infrastructure', 'Human Capital'],
            ecosystem_services: ['Resource Provision', 'Risk Mitigation', 'Information Services', 'Economic Stability']
        },
        'Manufacturing': {
            activities: ['Automotive Production', 'Chemical Manufacturing', 'Steel Production', 'Textile Manufacturing', 'Plastics Production', 'Electronics Assembly', 'Food Processing', 'Pharmaceutical'],
            pressures: ['Industrial Emissions', 'Water Pollution', 'Waste Generation', 'Energy Consumption', 'Raw Material Use', 'Chemical Discharge'],
            mechanisms: ['Environmental Pollution', 'Resource Depletion', 'Habitat Destruction', 'Climate Impact', 'Ecosystem Disruption'],
            natural_capital: ['Raw Materials', 'Water Systems', 'Energy Sources', 'Atmosphere', 'Land Resources', 'Mineral Deposits'],
            ecosystem_services: ['Material Provision', 'Waste Processing', 'Climate Regulation', 'Water Purification']
        }
    };

    const data = portfolioData[portfolioType] || portfolioData['Energy'];
    
    return {
        activities: data.activities.slice(0, 8), // Standard 8 activities
        pressures: data.pressures.slice(0, 6),   // Standard 6 pressures  
        mechanisms: data.mechanisms.slice(0, 4), // Standard 4 mechanisms
        natural_capital: data.natural_capital.slice(0, 5), // Standard 5 natural capital
        ecosystem_services: data.ecosystem_services.slice(0, 4) // Standard 4 ecosystem services
    };
}

// Enhanced network data creation with CIRCULAR FLOW clustering like Python engine
function createNetworkData(portfolioType) {
    const data = generateMockPortfolioData(portfolioType);
    const nodes = [];
    const edges = [];
    const spacing = 1.0; // Fixed spacing for consistency
    
    // Define cluster centers - CIRCULAR FLOW: Portfolio → Activities → Pressures → Mechanisms → Natural Capital → Ecosystem Services → (circular back)
    const clusterCenters = {
        'portfolio': [-6 * spacing, 0],           // Far left - starting point
        'activity': [-3 * spacing, 3 * spacing], // Top left
        'pressure': [0, 3 * spacing],             // Top center  
        'mechanism': [3 * spacing, 2 * spacing], // Top right
        'natural_capital': [4 * spacing, -1 * spacing],    // Right
        'ecosystem_service': [2 * spacing, -3 * spacing],  // Bottom right
        'upstream_supply': [-5 * spacing, -2 * spacing],   // Bottom left
        'downstream_supply': [-1 * spacing, -3 * spacing]  // Bottom center
    };

    // Function to position nodes within cluster using Python algorithm
    function positionNodesInCluster(nodeArray, nodeType) {
        const [centerX, centerY] = clusterCenters[nodeType] || [0, 0];
        const positions = [];
        
        if (nodeArray.length === 1) {
            positions.push([centerX, centerY]);
        } else if (nodeArray.length <= 6) {
            // Small clusters: circular arrangement (Python algorithm)
            nodeArray.forEach((node, i) => {
                const angle = (2 * Math.PI * i) / nodeArray.length;
                const radius = (0.8 + 0.1 * nodeArray.length) * spacing;
                const x = centerX + radius * Math.cos(angle);
                const y = centerY + radius * Math.sin(angle);
                positions.push([x, y]);
            });
        } else {
            // Large clusters: grid arrangement (Python algorithm)
            const cols = Math.ceil(Math.sqrt(nodeArray.length));
            const rows = Math.ceil(nodeArray.length / cols);
            nodeArray.forEach((node, i) => {
                const row = Math.floor(i / cols);
                const col = i % cols;
                const x = centerX + (col - cols/2) * 0.5 * spacing;
                const y = centerY + (row - rows/2) * 0.5 * spacing;
                positions.push([x, y]);
            });
        }
        
        return positions;
    }

    // Add portfolio center node
    const portfolioPos = positionNodesInCluster([`${portfolioType} Portfolio`], 'portfolio');
    nodes.push({
        id: `${portfolioType} Portfolio`,
        label: portfolioType,
        type: 'portfolio',
        x: portfolioPos[0][0],
        y: portfolioPos[0][1]
    });

    // Add activities in circular arrangement around their cluster center
    const activityPositions = positionNodesInCluster(data.activities, 'activity');
    data.activities.forEach((activity, i) => {
        nodes.push({
            id: activity,
            label: activity,
            type: 'activity',
            x: activityPositions[i][0],
            y: activityPositions[i][1]
        });
        edges.push({
            source: `${portfolioType} Portfolio`,
            target: activity
        });
    });

    // Add pressures in their cluster
    const pressurePositions = positionNodesInCluster(data.pressures, 'pressure');
    data.pressures.forEach((pressure, i) => {
        nodes.push({
            id: pressure,
            label: pressure,
            type: 'pressure',
            x: pressurePositions[i][0],
            y: pressurePositions[i][1]
        });
        
        // Connect to activities (circular flow: Activities → Pressures)
        const relatedActivity = data.activities[i % data.activities.length];
        edges.push({
            source: relatedActivity,
            target: pressure
        });
    });

    // Add mechanisms in their cluster  
    const mechanismPositions = positionNodesInCluster(data.mechanisms, 'mechanism');
    data.mechanisms.forEach((mechanism, i) => {
        nodes.push({
            id: mechanism,
            label: mechanism,
            type: 'mechanism',
            x: mechanismPositions[i][0],
            y: mechanismPositions[i][1]
        });
        
        // Connect to pressures (circular flow: Pressures → Mechanisms)
        const relatedPressure = data.pressures[i % data.pressures.length];
        edges.push({
            source: relatedPressure,
            target: mechanism
        });
    });

    // Add natural capital in their cluster
    const naturalCapitalPositions = positionNodesInCluster(data.natural_capital, 'natural_capital');
    data.natural_capital.forEach((capital, i) => {
        nodes.push({
            id: capital,
            label: capital,
            type: 'natural_capital',
            x: naturalCapitalPositions[i][0],
            y: naturalCapitalPositions[i][1]
        });
        
        // Connect to mechanisms (circular flow: Mechanisms → Natural Capital)
        const relatedMechanism = data.mechanisms[i % data.mechanisms.length];
        edges.push({
            source: relatedMechanism,
            target: capital
        });
    });

    // Add ecosystem services in their cluster
    const ecoServicePositions = positionNodesInCluster(data.ecosystem_services, 'ecosystem_service');
    data.ecosystem_services.forEach((service, i) => {
        nodes.push({
            id: service,
            label: service,
            type: 'ecosystem_service',
            x: ecoServicePositions[i][0],
            y: ecoServicePositions[i][1]
        });
        
        // Connect to natural capital (circular flow: Natural Capital → Ecosystem Services)
        const relatedCapital = data.natural_capital[i % data.natural_capital.length];
        edges.push({
            source: relatedCapital,
            target: service
        });
        
        // CIRCULAR FEEDBACK: Connect back to activities (Ecosystem Services → Activities)
        const relatedActivity = data.activities[i % data.activities.length];
        edges.push({
            source: service,
            target: relatedActivity
        });
    });

    return { nodes, edges };
}

// Enhanced Plotly visualization
function createPlotlyVisualization(networkData, settings) {
    const { nodes, edges } = networkData;
    const { showLabels } = settings;
    
    // Edge traces
    const edgeX = [];
    const edgeY = [];
    
    edges.forEach(edge => {
        const sourceNode = nodes.find(n => n.id === edge.source);
        const targetNode = nodes.find(n => n.id === edge.target);
        
        if (sourceNode && targetNode) {
            edgeX.push(sourceNode.x, targetNode.x, null);
            edgeY.push(sourceNode.y, targetNode.y, null);
        }
    });

    const edgeTrace = {
        x: edgeX,
        y: edgeY,
        mode: 'lines',
        line: {
            width: 2,
            color: 'rgba(156, 163, 175, 0.4)'
        },
        hoverinfo: 'none',
        showlegend: false
    };

    // Node traces by type
    const nodeTraces = [];
    const nodeTypes = [...new Set(nodes.map(n => n.type))];
    
    nodeTypes.forEach(type => {
        const nodesOfType = nodes.filter(n => n.type === type);
        
        const nodeTrace = {
            x: nodesOfType.map(n => n.x),
            y: nodesOfType.map(n => n.y),
            mode: showLabels ? 'markers+text' : 'markers',
            marker: {
                size: nodesOfType.map(n => {
                    const baseSize = n.type === 'portfolio' ? 35 : 22;
                    return baseSize;
                }),
                color: NODE_COLORS[type],
                line: {
                    width: 2,
                    color: 'rgba(255, 255, 255, 0.3)'
                },
                opacity: 0.9
            },
            text: showLabels ? nodesOfType.map(n => {
                const maxLength = 15;
                return n.label.length > maxLength ? n.label.substring(0, maxLength) + '...' : n.label;
            }) : [],
            textposition: 'middle center',
            textfont: {
                size: 10,
                color: '#f8fafc',
                family: 'Inter, sans-serif'
            },
            name: type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
            hovertemplate: '<b>%{text}</b><br>Type: ' + type.replace('_', ' ') + '<extra></extra>',
            type: 'scatter'
        };
        
        nodeTraces.push(nodeTrace);
    });

    const allTraces = [edgeTrace, ...nodeTraces];

    // Layout configuration
    const layout = {
        showlegend: false, // Remove Plotly legend - we have our own HTML legend
        xaxis: {
            showgrid: false,
            zeroline: false,
            showticklabels: false,
            showline: false
        },
        yaxis: {
            showgrid: false,
            zeroline: false,
            showticklabels: false,
            showline: false
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 40, b: 60, l: 20, r: 20 },
        hovermode: 'closest',
        dragmode: 'pan'
    };

    const config = {
        displayModeBar: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'resetScale2d', 'toggleHover', 'toggleSpikelines'],
        displaylogo: false,
        responsive: true,
        toImageButtonOptions: {
            format: 'png',
            filename: `${portfolioSelect.value}_portfolio_impact_network`,
            height: 800,
            width: 1200,
            scale: 2
        }
    };

    return Plotly.newPlot(plotlyChart, allTraces, layout, config);
}

// Generate visualization
async function generateVisualization() {
    showLoading();
    
    try {
        await new Promise(resolve => setTimeout(resolve, 500));
        
        const portfolioType = portfolioSelect.value;
        
        const networkData = createNetworkData(portfolioType);
        currentPortfolioData = networkData;
        
        const settings = {
            showLabels: showLabelsToggle.checked
        };
        
        await createPlotlyVisualization(networkData, settings);
        hideLoading();
        
    } catch (error) {
        console.error('Error generating visualization:', error);
        hideLoading();
        showError('Error generating visualization. Please try again.');
    }
}

// Update existing visualization
function updateVisualization() {
    if (currentPortfolioData) {
        const settings = {
            showLabels: showLabelsToggle.checked
        };
        createPlotlyVisualization(currentPortfolioData, settings);
    }
}

// Export functionality
function exportVisualization() {
    if (plotlyChart) {
        Plotly.downloadImage(plotlyChart, {
            format: 'png',
            filename: `${portfolioSelect.value}_portfolio_impact_network`,
            height: 800,
            width: 1200,
            scale: 2
        });
    }
}

// Fullscreen functionality
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        plotlyChart.requestFullscreen().catch(err => {
            console.log('Error attempting to enable fullscreen:', err.message);
        });
    } else {
        document.exitFullscreen();
    }
}

// Utility functions
function showLoading() {
    loadingSpinner.classList.remove('hidden');
    plotlyChart.style.opacity = '0.3';
}

function hideLoading() {
    loadingSpinner.classList.add('hidden');
    plotlyChart.style.opacity = '1';
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        <div style="text-align: center; padding: 2rem; color: #ef4444;">
            <i class="fas fa-exclamation-triangle" style="font-size: 2rem; margin-bottom: 1rem;"></i>
            <p>${message}</p>
        </div>
    `;
    plotlyChart.innerHTML = '';
    plotlyChart.appendChild(errorDiv);
}