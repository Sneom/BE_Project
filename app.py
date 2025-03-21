from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from model_handler import NetworkTrafficClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size
ALLOWED_EXTENSIONS = {'csv'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model handler
classifier = NetworkTrafficClassifier()

def allowed_file(filename):
    """Check if the filename has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle both page display and file upload"""
    if request.method == 'POST':
        # Check if file part exists
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Check if file is allowed
        if file and allowed_file(file.filename):
            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Process the file
                df = pd.read_csv(filepath)
                results = classifier.predict(df)
                
                # Clean up file
                os.remove(filepath)
                
                return jsonify({'success': True, 'results': results})
            except Exception as e:
                # Clean up file in case of error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': str(e)}), 500
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    # GET request - show the upload form
    return render_template('index.html')

@app.route('/model-info')
def model_info():
    """Get model information"""
    try:
        info = classifier.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(e):
    """Handle large file uploads"""
    return jsonify({'error': 'File too large (max 200MB)'}), 413

@app.errorhandler(500)
def server_error(e):
    """Handle server errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True) 