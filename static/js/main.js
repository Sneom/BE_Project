document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    const uploadSection = document.getElementById('upload-section');
    const loadingSection = document.getElementById('loading-section');
    const resultsSection = document.getElementById('results-section');
    const backBtn = document.getElementById('back-btn');
    const uploadProgress = document.getElementById('upload-progress');
    const confusionMatrixSection = document.getElementById('confusion-matrix-section');
    const classificationReportSection = document.getElementById('classification-report-section');
    
    // Fetch model info
    fetchModelInfo();
    
    // Handle drag and drop
    dropzone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    
    dropzone.addEventListener('dragleave', function() {
        dropzone.classList.remove('dragover');
    });
    
    dropzone.addEventListener('drop', function(e) {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            
            // Auto-submit when file is dropped
            uploadForm.dispatchEvent(new Event('submit'));
        }
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select a file');
            return;
        }
        
        if (file.size > 209715200) { // 200MB in bytes
            alert('File is too large (max 200MB)');
            return;
        }
        
        if (!file.name.toLowerCase().endsWith('.csv')) {
            alert('Only CSV files are allowed');
            return;
        }
        
        // Show loading section
        uploadSection.style.display = 'none';
        loadingSection.style.display = 'block';
        
        // Upload file with progress tracking
        uploadFile(file);
    });
    
    // Handle back button
    backBtn.addEventListener('click', function() {
        resetUI();
    });
    
    // Function to fetch model information
    function fetchModelInfo() {
        fetch('/model-info')
            .then(response => response.json())
            .then(data => {
                const modelInfoSection = document.getElementById('model-info-section');
                
                if (data.error) {
                    modelInfoSection.innerHTML = `
                        <div class="alert alert-danger">
                            ${data.error}
                        </div>
                    `;
                    return;
                }
                
                let html = `
                    <div class="card">
                        <div class="card-body">
                            <div class="mb-3">
                                <strong>Model Status:</strong> 
                                <span class="badge ${data.status ? 'bg-success' : 'bg-danger'}">
                                    ${data.status ? 'Ready' : 'Not Loaded'}
                                </span>
                            </div>
                            <div class="mb-3">
                                <strong>Processing Device:</strong> 
                                <span class="badge ${data.device.includes('GPU') ? 'bg-success' : 'bg-warning text-dark'}">
                                    ${data.device.includes('GPU') ? 'GPU' : 'CPU'}
                                </span>
                            </div>
                            <div class="mb-3">
                                <strong>Traffic Types:</strong><br>
                                <div class="mt-1">
                                    ${data.labels.map(label => 
                                        `<span class="badge bg-${getBadgeColor(label)} me-1">${label}</span>`
                                    ).join('')}
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                modelInfoSection.innerHTML = html;
            })
            .catch(error => {
                document.getElementById('model-info-section').innerHTML = `
                    <div class="alert alert-danger">
                        Error loading model information
                    </div>
                `;
            });
    }
    
    // Function to upload file with progress tracking
    function uploadFile(file) {
        const xhr = new XMLHttpRequest();
        const formData = new FormData();
        formData.append('file', file);
        
        xhr.open('POST', '/', true);
        
        // Track upload progress
        xhr.upload.onprogress = function(e) {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                uploadProgress.style.width = percentComplete + '%';
                uploadProgress.textContent = Math.round(percentComplete) + '%';
            }
        };
        
        xhr.onload = function() {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                
                if (response.success) {
                    // Show results section
                    loadingSection.style.display = 'none';
                    resultsSection.style.display = 'block';
                    
                    // Display results
                    displayResults(response.results);
                } else {
                    alert('Error: ' + response.error);
                    resetUI();
                }
            } else {
                alert('Error processing file. Status code: ' + xhr.status);
                resetUI();
            }
        };
        
        xhr.onerror = function() {
            alert('Network error occurred');
            resetUI();
        };
        
        xhr.send(formData);
    }
    
    // Function to display results
    function displayResults(results) {
        // Clear any previous visualization
        if (window.myChart) {
            window.myChart.destroy();
        }
        
        // Display file analysis info
        document.getElementById('num-samples').textContent = results.num_samples;
        document.getElementById('file-size').textContent = results.file_size_mb.toFixed(2) + ' MB';
        document.getElementById('processing-time').textContent = results.processing_time.toFixed(2) + ' seconds';
        document.getElementById('device-used').textContent = results.device_used;
        
        // Create distribution chart
        createDistributionChart(results.distribution);
        
        // Display traffic summary
        displayTrafficSummary(results.distribution, results.num_samples);
        
        // Display predictions in a better format
        displayPredictionSummary(results);
        
        // Display confusion matrix if available
        if (results.confusion_matrix_img) {
            document.getElementById('confusion-matrix-img').src = 'data:image/png;base64,' + results.confusion_matrix_img;
            document.getElementById('confusion-matrix-img').style.display = 'block';
            document.getElementById('confusion-matrix-placeholder').style.display = 'none';
        } else {
            document.getElementById('confusion-matrix-img').style.display = 'none';
            document.getElementById('confusion-matrix-placeholder').style.display = 'block';
        }
        
        // Display classification report if available
        if (results.classification_report) {
            displayClassificationReport(results.classification_report);
            document.getElementById('classification-report-table-container').style.display = 'block';
            document.getElementById('classification-report-placeholder').style.display = 'none';
        } else {
            document.getElementById('classification-report-table-container').style.display = 'none';
            document.getElementById('classification-report-placeholder').style.display = 'block';
        }
    }
    
    // Function to create distribution chart
    function createDistributionChart(distribution) {
        const labels = Object.keys(distribution);
        const data = Object.values(distribution);
        
        // Verify if data is valid (has some variation)
        const sum = data.reduce((a, b) => a + b, 0);
        const allSame = data.every(val => val === data[0]);
        
        if (sum === 0 || allSame) {
            document.getElementById('distribution-chart').parentElement.innerHTML = `
                <div class="alert alert-warning">
                    <i class="bi bi-exclamation-triangle"></i> 
                    Traffic distribution data is not available or uniform across all classes.
                </div>
            `;
            return;
        }
        
        const ctx = document.getElementById('distribution-chart').getContext('2d');
        
        window.myChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: [
                        'rgba(78, 115, 223, 0.8)',
                        'rgba(28, 200, 138, 0.8)',
                        'rgba(54, 185, 204, 0.8)',
                        'rgba(246, 194, 62, 0.8)',
                        'rgba(231, 74, 59, 0.8)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                const total = context.chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Function to display comprehensive prediction summary
    function displayPredictionSummary(results) {
        const container = document.getElementById('predictions-container');
        container.innerHTML = '';
        
        // Create a summary table instead of individual cards
        const tableDiv = document.createElement('div');
        tableDiv.className = 'col-12';
        tableDiv.innerHTML = `
            <div class="table-responsive">
                <table class="table table-hover">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Traffic Type</th>
                            <th>Confidence</th>
                            <th>Visualization</th>
                        </tr>
                    </thead>
                    <tbody id="prediction-table-body"></tbody>
                </table>
            </div>
        `;
        container.appendChild(tableDiv);
        
        const tableBody = document.getElementById('prediction-table-body');
        
        // Create a class summary first
        const classSummaryDiv = document.createElement('div');
        classSummaryDiv.className = 'col-12 mb-4';
        classSummaryDiv.innerHTML = `
            <div class="card">
                <div class="card-header bg-primary text-white">
                    <h6 class="mb-0">Traffic Classification Summary</h6>
                </div>
                <div class="card-body">
                    <div class="row">
                        ${Object.entries(results.distribution).map(([label, count]) => {
                            const percentage = (count / results.num_samples * 100).toFixed(1);
                            return `
                                <div class="col-md-4 mb-2">
                                    <div class="d-flex justify-content-between">
                                        <span><strong>${label}</strong>:</span>
                                        <span>${count} (${percentage}%)</span>
                                    </div>
                                    <div class="progress mt-1">
                                        <div class="progress-bar" style="width: ${percentage}%" aria-valuenow="${percentage}" 
                                             aria-valuemin="0" aria-valuemax="100"></div>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            </div>
        `;
        container.insertBefore(classSummaryDiv, tableDiv);
        
        // Display first 20 predictions
        const maxDisplay = Math.min(20, results.predictions.length);
        
        for (let i = 0; i < maxDisplay; i++) {
            const prediction = results.predictions[i];
            const confidence = results.confidence_scores[i] * 100;
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${i + 1}</td>
                <td>
                    <span class="badge bg-${getBadgeColor(prediction)}">${prediction}</span>
                </td>
                <td>${confidence.toFixed(1)}%</td>
                <td>
                    <div class="confidence-bar" style="width: 100%;">
                        <div class="confidence-bar-fill bg-${getBadgeColor(prediction)}" 
                             style="width: ${confidence}%"></div>
                    </div>
                </td>
            `;
            
            tableBody.appendChild(row);
        }
        
        if (results.predictions.length > maxDisplay) {
            const noteDiv = document.createElement('div');
            noteDiv.className = 'col-12 mt-2';
            noteDiv.innerHTML = `
                <div class="alert alert-info mb-0">
                    <i class="bi bi-info-circle"></i> Showing ${maxDisplay} out of ${results.predictions.length} predictions
                </div>
            `;
            container.appendChild(noteDiv);
        }
    }
    
    // Helper function to get badge color based on traffic type
    function getBadgeColor(trafficType) {
        const colorMap = {
            'chat': 'primary',
            'email': 'success',
            'file': 'info',
            'streaming': 'warning',
            'voip': 'danger'
        };
        return colorMap[trafficType] || 'secondary';
    }
    
    // Function to display classification report
    function displayClassificationReport(report) {
        const tableBody = document.querySelector('#classification-report-table tbody');
        tableBody.innerHTML = '';
        
        // Add each class
        Object.keys(report).forEach(key => {
            if (!['accuracy', 'macro avg', 'weighted avg'].includes(key)) {
                const classData = report[key];
                const row = document.createElement('tr');
                
                // Format each metric value, applying class for N/A values
                const precisionValue = formatMetricValue(classData.precision);
                const recallValue = formatMetricValue(classData.recall);
                const f1Value = formatMetricValue(classData.f1_score);
                
                row.innerHTML = `
                    <td>${key}</td>
                    <td>${precisionValue}</td>
                    <td>${recallValue}</td>
                    <td>${f1Value}</td>
                    <td>${classData.support}</td>
                `;
                
                tableBody.appendChild(row);
            }
        });
        
        // Add summary rows
        if (report['macro avg'] && report['weighted avg']) {
            // Add a separator row
            const separatorRow = document.createElement('tr');
            separatorRow.innerHTML = '<td colspan="5" class="border-top border-dark"></td>';
            tableBody.appendChild(separatorRow);
            
            // Add accuracy
            if ('accuracy' in report) {
                const accuracyRow = document.createElement('tr');
                accuracyRow.className = 'table-light';
                
                const accuracyValue = formatMetricValue(report['accuracy']);
                
                accuracyRow.innerHTML = `
                    <td><strong>Accuracy</strong></td>
                    <td colspan="3">${accuracyValue}</td>
                    <td>${report['macro avg'].support || ''}</td>
                `;
                tableBody.appendChild(accuracyRow);
            }
            
            // Add macro avg
            const macroRow = document.createElement('tr');
            macroRow.className = 'table-light';
            
            const macroPrecision = formatMetricValue(report['macro avg'].precision);
            const macroRecall = formatMetricValue(report['macro avg'].recall);
            const macroF1 = formatMetricValue(report['macro avg'].f1_score);
            
            macroRow.innerHTML = `
                <td><strong>Macro Avg</strong></td>
                <td>${macroPrecision}</td>
                <td>${macroRecall}</td>
                <td>${macroF1}</td>
                <td>${report['macro avg'].support}</td>
            `;
            tableBody.appendChild(macroRow);
            
            // Add weighted avg
            const weightedRow = document.createElement('tr');
            weightedRow.className = 'table-light';
            
            const weightedPrecision = formatMetricValue(report['weighted avg'].precision);
            const weightedRecall = formatMetricValue(report['weighted avg'].recall);
            const weightedF1 = formatMetricValue(report['weighted avg'].f1_score);
            
            weightedRow.innerHTML = `
                <td><strong>Weighted Avg</strong></td>
                <td>${weightedPrecision}</td>
                <td>${weightedRecall}</td>
                <td>${weightedF1}</td>
                <td>${report['weighted avg'].support}</td>
            `;
            tableBody.appendChild(weightedRow);
        }
    }
    
    // Helper function to format metric values with proper styling for N/A
    function formatMetricValue(value) {
        if (value === null || value === undefined || isNaN(value)) {
            return '<span class="na-value">0.00%</span>';
        }
        
        // Ensure the value is displayed properly, using actual values
        const percentage = (value * 100).toFixed(2);
        return percentage + '%';
    }
    
    // Function to reset UI
    function resetUI() {
        uploadSection.style.display = 'block';
        loadingSection.style.display = 'none';
        resultsSection.style.display = 'none';
        uploadProgress.style.width = '0%';
        uploadProgress.textContent = '';
        uploadForm.reset();
    }
    
    // Format numbers with commas
    function formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }
    
    // Format percentages with NaN handling
    function formatPercent(value) {
        if (value === null || value === undefined || isNaN(value)) {
            return "N/A";
        }
        // Convert to percentage and format with 2 decimal places
        return (value * 100).toFixed(2) + '%';
    }
    
    // Display Traffic Summary in the Overview tab
    function displayTrafficSummary(distribution, totalSamples) {
        const container = document.getElementById('traffic-summary');
        container.innerHTML = '';
        
        // Get colors for each traffic type
        const colors = {
            'chat': '#4e73df',
            'email': '#1cc88a',
            'file': '#36b9cc',
            'streaming': '#f6c23e',
            'voip': '#e74a3b'
        };
        
        Object.entries(distribution).forEach(([label, count]) => {
            const percentage = (count / totalSamples * 100).toFixed(1);
            const color = colors[label] || '#6c757d';
            
            const trafficItem = document.createElement('div');
            trafficItem.className = 'mb-3';
            trafficItem.innerHTML = `
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <span class="fw-bold">${label.toUpperCase()}</span>
                    <span>${count} (${percentage}%)</span>
                </div>
                <div class="progress traffic-progress">
                    <div class="progress-bar" 
                         style="width: ${percentage}%; background-color: ${color}" 
                         aria-valuenow="${percentage}" 
                         aria-valuemin="0" 
                         aria-valuemax="100">
                    </div>
                </div>
            `;
            container.appendChild(trafficItem);
        });
    }
}); 