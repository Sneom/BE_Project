import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
from tensorflow.keras.layers import LeakyReLU
import os
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

class NetworkTrafficClassifier:
    def __init__(self):
        self.LABELS_ISCX = ['chat', 'email', 'file', 'streaming', 'voip']
        self.LATENT_DIM = 100
        self.models_loaded = False
        
        # Configure TensorFlow to use GPU if available
        self.configure_gpu()
        
        # Load models
        self.load_models()
    
    def configure_gpu(self):
        """Configure GPU for TensorFlow if available"""
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            try:
                # Set memory growth to avoid taking all GPU memory
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                
                self.device = '/GPU:0'
                print("✅ GPU is available and configured for use")
            except Exception as e:
                print(f"❌ Error configuring GPU: {e}")
                self.device = '/CPU:0'
        else:
            self.device = '/CPU:0'
            print("ℹ️ No GPU found, using CPU")

    def load_models(self):
        """Load all required models and scaler"""
        try:
            print("# ==========================")
            print("# 🔹 Load Saved Models")
            print("# ==========================")
            
            with tf.device(self.device):
                self.generator = load_model("models/fgan2_generator.keras")
                print("✅ Generator model loaded")
                
                self.discriminator = load_model("models/fgan2_discriminator.keras")
                print("✅ Discriminator model loaded")
                
                self.classifier = tf.keras.models.load_model(
                    "models/cnn_classifier.keras",
                    custom_objects={"LeakyReLU": LeakyReLU}
                )
                print("✅ CNN Classifier model loaded")
            
            self.gmm = joblib.load("models/gmm_model.pkl")
            print("✅ GMM model loaded")
            
            self.scaler = joblib.load("models/scaler.pkl")
            print("✅ Scaler loaded")
            
            self.models_loaded = True
            print("✅ All models loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.models_loaded = False

    def preprocess_data(self, df):
        """Preprocess input data"""
        try:
            print("# ==========================")
            print("# 🔹 Preprocessing Data")
            print("# ==========================")
            
            # Extract labels if available
            if "label" in df.columns:
                Y = df["label"].values
                y_true = np.array([int(val) for val in Y])
                y_true_onehot = tf.keras.utils.to_categorical(y_true, len(self.LABELS_ISCX))
            else:
                y_true = None
                y_true_onehot = None
            
            # Extract features (exclude index column)
            X = df.iloc[:, 1:].values
            print(f"✅ Extracted features: {X.shape}")
            
            # Scale the features
            X_scaled = self.scaler.transform(X)
            print(f"✅ Scaled features: {X_scaled.shape}")
            
            # Reshape for CNN input
            X_reshaped = np.expand_dims(X_scaled, axis=-1)
            print(f"✅ Reshaped for CNN: {X_reshaped.shape}")
            
            return X_scaled, X_reshaped, y_true, y_true_onehot
        except Exception as e:
            raise ValueError(f"Error in preprocessing data: {e}")

    def generate_fake_samples(self, num_samples):
        """Generate fake samples using the generator"""
        print("# ==========================")
        print("# 🔹 Generate Fake Samples using Generator")
        print("# ==========================")
        
        noise = np.random.normal(0, 1, (num_samples, self.LATENT_DIM))
        with tf.device(self.device):
            generated = self.generator.predict(noise, batch_size=128)
            print(f"✅ Generated {num_samples} fake samples")
            return generated

    def get_discriminator_features(self, X_scaled):
        """Get discriminator predictions/features"""
        print("# ==========================")
        print("# 🔹 Get Discriminator Predictions")
        print("# ==========================")
        
        with tf.device(self.device):
            disc_features = self.discriminator.predict(X_scaled, batch_size=128).reshape(-1, 1)
            print(f"✅ Extracted discriminator features: {disc_features.shape}")
            return disc_features

    def predict(self, df):
        """Main prediction function"""
        try:
            if not self.models_loaded:
                raise ValueError("Models not loaded properly")

            start_time = time.time()
            
            # Preprocess data
            X_scaled, X_reshaped, y_true, y_true_onehot = self.preprocess_data(df)

            # Get classifier predictions
            print("# ==========================")
            print("# 🔹 Running CNN Classifier")
            print("# ==========================")
            with tf.device(self.device):
                predictions = self.classifier.predict(X_reshaped, batch_size=128)
            
            pred_classes = np.argmax(predictions, axis=1)
            print(f"✅ Made predictions for {len(pred_classes)} samples")
            
            # Get confidence scores
            confidence_scores = np.max(predictions, axis=1)

            # Get discriminator features and GMM pseudo-labels
            disc_features = self.get_discriminator_features(X_scaled)
            
            print("# ==========================")
            print("# 🔹 Apply GMM for Pseudo-Labeling")
            print("# ==========================")
            pseudo_labels = self.gmm.predict(disc_features)
            pseudo_labels_onehot = tf.keras.utils.to_categorical(pseudo_labels, len(self.LABELS_ISCX))
            print(f"✅ Generated pseudo-labels for {len(pseudo_labels)} samples")

            # Generate some fake samples (for demonstration)
            num_gen_samples = min(100, X_scaled.shape[0])
            generated_samples = self.generate_fake_samples(num_gen_samples)

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Classification report and confusion matrix
            report_data = None
            confusion_matrix_img = None
            
            if y_true is not None:
                print("# ==========================")
                print("# 🔹 Evaluate CNN Classifier")
                print("# ==========================")
                
                # Ensure we have valid classification data
                valid_indices = ~np.isnan(y_true)
                if np.sum(valid_indices) > 0:
                    valid_y_true = y_true[valid_indices].astype(int)
                    valid_pred_classes = pred_classes[valid_indices]
                    
                    # Classification Report with proper handling
                    try:
                        # Generate classification report with zero_division=0 to prevent NaN values
                        report = classification_report(valid_y_true, valid_pred_classes, 
                                                   target_names=self.LABELS_ISCX,
                                                   output_dict=True, zero_division=0)
                        
                        # Fix any potential issues with the report data
                        for class_name, metrics in report.items():
                            if isinstance(metrics, dict):
                                for metric_name, value in metrics.items():
                                    if metric_name != 'support':
                                        # Fix any NaN values by setting them to 0
                                        if np.isnan(value) or value is None:
                                            report[class_name][metric_name] = 0.0
                            elif isinstance(report[class_name], (float, int)) and (np.isnan(report[class_name]) or report[class_name] is None):
                                report[class_name] = 0.0
                        
                        # Print classification report for terminal display            
                        print("\n🔹 Classification Report:")
                        print(classification_report(valid_y_true, valid_pred_classes, 
                                                  target_names=self.LABELS_ISCX, zero_division=0))
                        
                        # Show f1-scores for each class in the terminal for verification
                        print("\n🔹 F1 Scores by Class:")
                        for class_name in self.LABELS_ISCX:
                            if class_name in report:
                                f1 = report[class_name]['f1-score']
                                f1 = 0.0 if np.isnan(f1) else f1
                                print(f"  - {class_name}: {f1:.4f} ({f1*100:.2f}%)")
                        
                        # Force set any remaining NaN values to 0 before sending to frontend
                        def fix_nan_recursive(obj):
                            if isinstance(obj, dict):
                                for key, value in obj.items():
                                    if isinstance(value, (dict, list)):
                                        fix_nan_recursive(value)
                                    elif isinstance(value, (float, int)) and np.isnan(value):
                                        obj[key] = 0.0
                            elif isinstance(obj, list):
                                for i, item in enumerate(obj):
                                    if isinstance(item, (dict, list)):
                                        fix_nan_recursive(item)
                                    elif isinstance(item, (float, int)) and np.isnan(item):
                                        obj[i] = 0.0
                        
                        # Apply NaN fixing recursively
                        fix_nan_recursive(report)
                        
                        # Confusion Matrix
                        conf_matrix = confusion_matrix(valid_y_true, valid_pred_classes)
                        print("✅ Generated confusion matrix")
                        
                        # Create confusion matrix plot
                        plt.figure(figsize=(6, 5))
                        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", 
                                    xticklabels=self.LABELS_ISCX, yticklabels=self.LABELS_ISCX)
                        plt.title("Confusion Matrix")
                        plt.xlabel("Predicted Label")
                        plt.ylabel("True Label")
                        
                        # Save plot to bytes for display in UI
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        confusion_matrix_img = base64.b64encode(buf.read()).decode('utf-8')
                        plt.close()
                        
                        report_data = report
                    except Exception as e:
                        print(f"⚠️ Warning: Could not generate classification report: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("⚠️ Warning: No valid labels found for classification report")
            
            print("✅ Testing completed.")
            
            # Calculate actual distribution of predictions
            distribution = {}
            for i, label in enumerate(self.LABELS_ISCX):
                count = int(np.sum(pred_classes == i))
                distribution[label] = count
                print(f"📊 {label}: {count} samples ({count/len(pred_classes)*100:.1f}%)")
            
            # Prepare results
            results = {
                'predictions': [self.LABELS_ISCX[i] for i in pred_classes],
                'confidence_scores': confidence_scores.tolist(),
                'distribution': distribution,
                'pseudo_labels': pseudo_labels.tolist(),
                'processing_time': elapsed_time,
                'file_size_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'num_samples': len(df),
                'device_used': self.device,
                'classification_report': report_data,
                'confusion_matrix_img': confusion_matrix_img
            }

            return results

        except Exception as e:
            raise ValueError(f"Error in prediction pipeline: {e}")

    def get_model_info(self):
        """Get information about the models"""
        return {
            'labels': self.LABELS_ISCX,
            'architecture': {
                'generator_params': self.generator.count_params(),
                'discriminator_params': self.discriminator.count_params(),
                'classifier_params': self.classifier.count_params()
            },
            'status': self.models_loaded,
            'device': self.device
        } 