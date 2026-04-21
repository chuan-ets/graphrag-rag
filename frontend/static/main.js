const API_BASE = 'http://localhost:8080';

// --- Graph Visualization ---
let network = null;

async function fetchAndRenderGraph() {
    try {
        const response = await fetch(`${API_BASE}/graph`);
        const data = await response.json();
        
        // Transform the NetworkX JSON to Vis.js format
        const nodes = (data.nodes || []).map(n => {
            let color = '#dbeafe';
            let border = '#3b82f6';
            let shape = 'dot';
            let size = n.mentions ? Math.max(10, Math.min(30, n.mentions * 2)) : 10;
            
            if (n.type === 'document') {
                color = '#f59e0b'; // Amber for documents
                border = '#d97706';
                shape = 'box';
                size = 30;
            }
            return {
                id: n.id,
                label: n.id,
                value: size,
                title: `Type: ${n.type || 'entity'}<br>Mentions: ${n.mentions || 1}<br>Docs: ${n.docs ? n.docs.length : 0}`,
                group: n.type || 'entity',
                shape: shape,
                color: { background: color, border: border, highlight: { background: border, border: '#ffffff' } },
                font: n.type === 'document' ? { color: '#ffffff', size: 16, face: 'Inter', bold: true } : undefined
            };
        });
        
        // Handle both links and edges keys from networkx
        const rawEdges = data.links || data.edges || [];
        const edges = rawEdges.map(e => ({
            from: e.source,
            to: e.target,
            label: e.key || e.relation,
            value: e.weight || 1,
            title: `Weight: ${e.weight || 1}`,
            arrows: 'to'
        }));

        const container = document.getElementById('mynetwork');
        const graphData = {
            nodes: new vis.DataSet(nodes),
            edges: new vis.DataSet(edges)
        };
        const options = {
            nodes: {
                shape: 'dot',
                scaling: {
                    min: 10,
                    max: 30
                },
                font: {
                    color: '#111827',
                    size: 14,
                    face: 'Inter'
                },
                color: {
                    border: '#3b82f6',
                    background: '#dbeafe',
                    highlight: {
                        border: '#2563eb',
                        background: '#bfdbfe'
                    }
                }
            },
            edges: {
                color: {
                    color: '#cbd5e1',
                    highlight: '#94a3b8'
                },
                font: {
                    color: '#64748b',
                    size: 11,
                    face: 'Inter',
                    align: 'middle'
                },
                smooth: {
                    type: 'continuous'
                }
            },
            physics: {
                forceAtlas2Based: {
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springConstant: 0.08,
                    springLength: 100,
                    damping: 0.4,
                    avoidOverlap: 0
                },
                solver: 'forceAtlas2Based',
                stabilization: {
                    enabled: true,
                    iterations: 100,
                    updateInterval: 50
                }
            }
        };
        
        if (network) {
            network.destroy();
        }
        network = new vis.Network(container, graphData, options);
        
        network.on("stabilizationProgress", function(params) {
            console.log("Stabilizing...", params.iterations, "/", params.total);
        });
    } catch (error) {
        console.error('Failed to load graph:', error);
    }
}

document.getElementById('refresh-graph-btn').addEventListener('click', fetchAndRenderGraph);

// --- File Upload ---
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const dropZone = document.getElementById('drop-zone');
const statusMsg = document.getElementById('upload-status');

browseBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', handleFiles);
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFiles();
    }
});

async function handleFiles() {
    if (!fileInput.files.length) return;
    const file = fileInput.files[0];
    
    statusMsg.style.color = '#94a3b8';
    statusMsg.innerText = `Uploading ${file.name}...`;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const res = await fetch(`${API_BASE}/ingest`, {
            method: 'POST',
            body: formData
        });
        const result = await res.json();
        
        if (result.status === 'success') {
            statusMsg.style.color = 'var(--success-color)';
            statusMsg.innerText = `Success! Chunked into ${result.chunks} parts.`;
            // Refresh graph after successful ingestion
            fetchAndRenderGraph();
        } else {
            throw new Error(result.message || 'Upload failed');
        }
    } catch (err) {
        statusMsg.style.color = '#ef4444';
        statusMsg.innerText = err.message;
    }
}

// --- Chat Interface ---
const chatHistory = document.getElementById('chat-history');
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');

function appendMessage(text, isUser = false, meta = null) {
    const div = document.createElement('div');
    div.className = `msg ${isUser ? 'user' : 'bot'}`;
    div.innerText = text;
    
    if (meta) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'msg-meta';
        metaDiv.innerHTML = `Intent: ${meta.intent} | Confidence: ${meta.confidence.toFixed(2)}`;
        div.appendChild(metaDiv);
    }
    
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query) return;
    
    appendMessage(query, true);
    queryInput.value = '';
    
    const formData = new FormData();
    formData.append('query', query);
    
    try {
        // Show typing indicator
        const typingDiv = document.createElement('div');
        typingDiv.className = 'msg bot';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerText = 'Thinking...';
        chatHistory.appendChild(typingDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        const res = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            body: formData
        });
        const result = await res.json();
        
        // Remove typing indicator
        document.getElementById('typing-indicator').remove();
        
        appendMessage(result.answer, false, result.router);
        
    } catch (err) {
        document.getElementById('typing-indicator')?.remove();
        appendMessage('Error: ' + err.message, false);
    }
}

sendBtn.addEventListener('click', sendQuery);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendQuery();
});

// Initial load
fetchAndRenderGraph();
