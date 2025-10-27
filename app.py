import os
import io
import base64
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mysql.connector
from PIL import Image
import numpy as np
import tensorflow as tf
import threading
from datetime import datetime, timedelta
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import random
import cv2
import base64

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
PORT = os.getenv('PORT', 5001)

# Database Configuration
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

# Global model variable
unified_model = None

# Global variables for retraining status
retraining_status = {
    'is_retraining': False,
    'progress': 0,
    'current_step': '',
    'last_retraining': None,
    'accuracy_improvement': 0
}

# HAM10000 Dataset Classes
HAM10000_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

HUMAN_READABLE_NAMES = {
    'akiec': 'Actinic Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'vasc': 'Vascular Lesion'
}

# Cancer indicators for HAM10000 classes
CANCER_INDICATORS = {
    'akiec': True,    # Pre-cancerous / cancerous
    'bcc': True,      # Cancerous
    'bkl': False,     # Non-cancerous
    'df': False,      # Non-cancerous
    'mel': True,      # Cancerous
    'nv': False,      # Non-cancerous
    'vasc': False     # Non-cancerous
}

# Medical feature descriptions optimized for HAM10000
MEDICAL_FEATURES = {
    'border_irregularity': {
        'description': 'Irregular or poorly defined borders',
        'significance': {
            'Melanoma': 'Highly significant - irregular borders are a key ABCDE feature',
            'Basal Cell Carcinoma': 'Common in BCC - pearly borders with rolled edges',
            'Actinic Keratosis': 'Often has irregular borders in early stages'
        }
    },
    'asymmetry': {
        'description': 'Lack of symmetry in the lesion',
        'significance': {
            'Melanoma': 'Critical feature - asymmetry is a major warning sign',
            'Basal Cell Carcinoma': 'Often asymmetric',
            'Melanocytic Nevus': 'Typically symmetric in benign cases'
        }
    },
    'color_variation': {
        'description': 'Multiple colors within the same lesion',
        'significance': {
            'Melanoma': 'Highly significant - multiple colors indicate malignancy',
            'Basal Cell Carcinoma': 'Often pearly or translucent with telangiectasia',
            'Benign Keratosis': 'Usually uniform in color'
        }
    },
    'diameter_large': {
        'description': 'Larger than 6mm in diameter',
        'significance': {
            'Melanoma': 'Warning sign - though melanomas can be smaller',
            'Basal Cell Carcinoma': 'Can vary in size',
            'Melanocytic Nevus': 'Often larger but stable in benign cases'
        }
    }
}


def get_db_connection():
    """Create database connection"""
    try:
        return mysql.connector.connect(**db_config)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


def initialize_database():
    """Initialize database tables if they don't exist"""
    try:
        conn = get_db_connection()
        if not conn:
            print("Warning: Could not connect to database for initialization")
            return

        cursor = conn.cursor()

        # Create doctor_verifications table
        cursor.execute("""
			CREATE TABLE IF NOT EXISTS doctor_verifications (
				id INT AUTO_INCREMENT PRIMARY KEY,
				original_diagnosis VARCHAR(255),
				verified_diagnosis VARCHAR(255),
				doctor_id VARCHAR(255),
				image_id VARCHAR(255),
				is_correct BOOLEAN,
				confidence_score FLOAT DEFAULT 0,
				notes TEXT,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)
		""")

        # Create analysis_history table
        cursor.execute("""
			CREATE TABLE IF NOT EXISTS analysis_history (
				id INT AUTO_INCREMENT PRIMARY KEY,
				user_id VARCHAR(255),
				image_path TEXT,
				diagnosis VARCHAR(255),
				confidence FLOAT,
				is_cancer BOOLEAN,
				cancer_status VARCHAR(100),
				explanations JSON,
				doctor_verified BOOLEAN DEFAULT FALSE,
				doctor_correction VARCHAR(255),
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)
		""")

        # Create dermatologists table
        cursor.execute("""
			CREATE TABLE IF NOT EXISTS dermatologists (
				id INT AUTO_INCREMENT PRIMARY KEY,
				name VARCHAR(255),
				specialty VARCHAR(255),
				experience INT,
				rating FLOAT,
				address TEXT,
				phone VARCHAR(50),
				email VARCHAR(255),
				latitude DECIMAL(10, 8),
				longitude DECIMAL(11, 8)
			)
		""")
        
        # Create community_insights table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS community_insights (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date DATE NOT NULL,
                total_scans INT DEFAULT 0,
                benign_count INT DEFAULT 0,
                malignant_count INT DEFAULT 0,
                disease_breakdown JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_date (date)
            )
        """)
        
         # Create preventive_care table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preventive_care (
                id INT AUTO_INCREMENT PRIMARY KEY,
                disease_type VARCHAR(100),
                prevention_tips JSON,
                risk_factors JSON,
                early_signs JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()

        print("Database tables initialized successfully")

    except Exception as e:
        print(f"Database initialization error: {e}")
def initialize_preventive_care():
    """Initialize preventive care information"""
    try:
        conn = get_db_connection()
        if not conn:
            return
            
        cursor = conn.cursor()
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM preventive_care")
        if cursor.fetchone()[0] > 0:
            cursor.close()
            conn.close()
            return
        preventive_data = [
            {
                'disease_type': 'Melanoma',
                'prevention_tips': [
                    "Use broad-spectrum sunscreen with SPF 30 or higher",
                    "Avoid sun exposure between 10 AM and 4 PM",
                    "Wear protective clothing and wide-brimmed hats",
                    "Avoid tanning beds completely",
                    "Perform monthly self-skin examinations"
                ],
                'risk_factors': [
                    "Fair skin that burns easily",
                    "History of sunburns",
                    "Excessive UV exposure",
                    "Family history of melanoma",
                    "Many moles or unusual moles"
                ],
                'early_signs': [
                    "Asymmetrical mole with irregular borders",
                    "Color variation within a single mole",
                    "Diameter larger than 6mm (pencil eraser)",
                     "Evolution - changing in size, shape, or color"
                ]
            },
            {
                'disease_type': 'Basal Cell Carcinoma',
                'prevention_tips': [
                    "Daily sunscreen use on exposed skin",
                    "Wear UV-protective clothing",
                    "Seek shade during peak sun hours",
                    "Avoid indoor tanning",
                    "Regular skin self-exams"
                ],
                'risk_factors': [
                    "Chronic sun exposure",
                    "Fair skin, light hair, light eyes",
                    "Age over 50 years",
                    "Personal or family history of skin cancer"
                ],
                'early_signs': [
                    "Pearly or waxy bump",
                    "Flat, flesh-colored or brown scar-like lesion",
                    "Bleeding or scabbing sore that heals and returns"
                ]
            },
            {
                'disease_type': 'general',
                'prevention_tips': [
                    "Perform monthly skin self-exams using mirrors",
                    "Know your skin and watch for changes",
                                        "Use sunscreen daily, even on cloudy days",
                    "Stay hydrated for healthy skin",
                    "Eat antioxidant-rich foods"
                ],
                'risk_factors': [
                    "UV exposure from sun or tanning beds",
                    "Family history of skin cancer",
                    "Personal history of skin cancer",
                    "Weakened immune system"
                ],
                'early_signs': [
                    "New growth on skin",
                    "Sore that doesn't heal",
                    "Change in existing mole",
                    "Spread of pigment beyond border"
                ]
            }
        ]
        for data in preventive_data:
            cursor.execute("""
                INSERT INTO preventive_care (disease_type, prevention_tips, risk_factors, early_signs)
                VALUES (%s, %s, %s, %s)
            """, (
                data['disease_type'],
                json.dumps(data['prevention_tips']),
                json.dumps(data['risk_factors']),
                json.dumps(data['early_signs'])
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        print("Preventive care data initialized successfully")
        
    except Exception as e:
        print(f"Preventive care initialization error: {e}")


def load_models():
    """Load unified TensorFlow model"""
    global unified_model

    try:
        print('Loading ResNet model for HAM10000...')

        # Load the unified model
        unified_model = tf.keras.models.load_model('models/resnet_model.h5')

        print('Resnet model loaded successfully')
        print(f'Model input shape: {unified_model.input_shape}')
        print(f'Model output shape: {unified_model.output_shape}')
        print(f'Number of classes: {unified_model.output_shape[-1]}')

    except Exception as e:
        print(f'Error loading model: {e}')
        print('Server will run without AI functionality')


def preprocess_image(image_data):
    """Preprocess image for model prediction"""
    try:
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize image (adjust based on your model requirements)
        image = image.resize((128, 128))

        # Convert to numpy array and normalize
        image_array = np.array(image)

        # If your model was trained with ResNet preprocessing, use:
        image_array = tf.keras.applications.resnet50.preprocess_input(
            image_array)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        return image_array
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        raise e

def generate_grad_cam(model, img_array, class_idx):
    """
    Working Grad-CAM implementation that handles various model architectures
    """
    try:
        print("🔍 Starting Grad-CAM generation...")
        
        # Find the last convolutional layer
        layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
        
        if layer_name is None:
            print("❌ No suitable layer found for Grad-CAM")
            return None
        
        print(f"✅ Using layer for Grad-CAM: {layer_name}")
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            
            # Handle output format
            if isinstance(predictions, (list, tuple)):
                pred_tensor = predictions[0]
            else:
                pred_tensor = predictions
            
            target_score = pred_tensor[0, class_idx]
        
        # Compute gradients
        grads = tape.gradient(target_score, conv_outputs)
        
        if grads is None:
            print("❌ Gradients are None")
            return None
        
        # Global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        
        # Apply ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        
        if max_val == 0:
            print("⚠️ Heatmap is all zeros")
            return None
        
        heatmap /= max_val
        
        # FIX: Simply return the numpy array without .numpy() call
        return np.array(heatmap)
        
    except Exception as e:
        print(f"❌ Grad-CAM failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
# def generate_and_apply_heatmap(model, img_array, class_idx, original_image):
    """
    Generate Grad-CAM and apply it to the original image
    """
    try:
        print("🔄 Generating heatmap...")
        heatmap = generate_grad_cam_working(model, img_array, class_idx)
        
        if heatmap is None:
            print("❌ Failed to generate heatmap")
            return None, None
        
        print("🎨 Applying heatmap to image...")
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        # Convert original image to RGB if needed
        if len(original_image.shape) == 2:  # Grayscale
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        else:
            original_image_rgb = original_image
        
        # Ensure both images have the same data type and size
        original_image_rgb = cv2.resize(original_image_rgb, (heatmap_colored.shape[1], heatmap_colored.shape[0]))
        original_image_rgb = original_image_rgb.astype(np.float32)
        heatmap_colored = heatmap_colored.astype(np.float32)
        
        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(original_image_rgb, 0.6, heatmap_colored, 0.4, 0)
        superimposed = np.uint8(np.clip(superimposed, 0, 255))
        
        print("✅ Heatmap applied successfully")
        return superimposed, heatmap_resized
        
    except Exception as e:
        print(f"❌ Heatmap application failed: {str(e)}")
        return None, None
def generate_and_apply_heatmap(model, img_array, class_idx, original_image):
    """
    Generate Grad-CAM and apply it to the original image
    """
    try:
        print("🔄 Generating heatmap...")
        heatmap = generate_grad_cam_working(model, img_array, class_idx)
        
        if heatmap is None:
            print("❌ Failed to generate heatmap")
            return None, None
        
        print(f"✅ Heatmap generated successfully, shape: {heatmap.shape}")
        print("🎨 Applying heatmap to image...")
        
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        # Convert original image to RGB if needed
        if len(original_image.shape) == 2:  # Grayscale
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        else:
            original_image_rgb = original_image
        
        # Ensure both images have the same data type and size
        original_image_rgb = cv2.resize(original_image_rgb, (heatmap_colored.shape[1], heatmap_colored.shape[0]))
        original_image_rgb = original_image_rgb.astype(np.float32)
        heatmap_colored = heatmap_colored.astype(np.float32)
        
        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(original_image_rgb, 0.6, heatmap_colored, 0.4, 0)
        superimposed = np.uint8(np.clip(superimposed, 0, 255))
        
        print("✅ Heatmap applied successfully")
        return superimposed, heatmap_resized
        
    except Exception as e:
        print(f"❌ Heatmap application failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None
    
    
def update_community_insights(diagnosis, is_cancer):
    """Update community insights with new scan data"""
    try:
        conn = get_db_connection()
        if not conn:
            return
            
        cursor = conn.cursor()
        
        today = datetime.now().date()
         # Check if entry exists for today
        cursor.execute("SELECT id, total_scans, benign_count, malignant_count, disease_breakdown FROM community_insights WHERE date = %s", (today,))
        result = cursor.fetchone()
        
        if result:
            # Update existing entry
            insight_id, total_scans, benign_count, malignant_count, disease_breakdown = result
            
            # Parse existing disease breakdown
            if disease_breakdown:
                breakdown = json.loads(disease_breakdown)
            else:
                breakdown = {}
            # Update counts
            total_scans += 1
            if is_cancer:
                malignant_count += 1
            else:
                benign_count += 1
            
            # Update disease breakdown
            breakdown[diagnosis] = breakdown.get(diagnosis, 0) + 1
            
            cursor.execute("""
                UPDATE community_insights 
                SET total_scans = %s, benign_count = %s, malignant_count = %s, disease_breakdown = %s 
                WHERE id = %s
            """, (total_scans, benign_count, malignant_count, json.dumps(breakdown), insight_id))
            
        else:
              # Create new entry
            breakdown = {diagnosis: 1}
            cursor.execute("""
                INSERT INTO community_insights (date, total_scans, benign_count, malignant_count, disease_breakdown)
                VALUES (%s, 1, %s, %s, %s)
            """, (today, 0 if is_cancer else 1, 1 if is_cancer else 0, json.dumps(breakdown)))
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Community insights update error: {e}")
    
def generate_xai_explanations(model, image_array, predicted_class, confidence, class_names, original_image):
    """
    Generate comprehensive XAI explanations
    """
    explanations = {
        'grad_cam_heatmap': None,
        'superimposed_image': None,
        'confidence_scores': {},
        'explanation_text': f"The model predicted '{predicted_class}' with {confidence:.2%} confidence.",
        'top_features': [],
        'all_probabilities': {}
    }
    
    try:
        print("📊 Getting confidence scores...")
        # Get confidence scores for all classes
        predictions = model.predict(image_array, verbose=0)
        
        # Handle different prediction formats
        if isinstance(predictions, (list, tuple)):
            print(f"📋 Raw predictions is list/tuple with {len(predictions)} elements")
            main_predictions = predictions[0]  # Take first element
        else:
            main_predictions = predictions
            
        print(f"📈 Main predictions shape: {main_predictions.shape}")
        
        # Store all probabilities
        for i, score in enumerate(main_predictions[0]):
            class_name = class_names[i]
            explanations['confidence_scores'][class_name] = float(score)
            explanations['all_probabilities'][class_name] = float(score)
        
        # Generate Grad-CAM visualization
        predicted_class_index = class_names.index(predicted_class)
        print(f"🎯 Generating explanations for class index: {predicted_class_index}")
        
        superimposed_img, heatmap = generate_and_apply_heatmap(
            model, image_array, predicted_class_index, original_image
        )
        
        if superimposed_img is not None and heatmap is not None:
            # Convert to base64 for web display
            print("🖼️ Converting images to base64...")
            _, buffer = cv2.imencode('.png', superimposed_img)
            superimposed_b64 = base64.b64encode(buffer).decode('utf-8')
            explanations['superimposed_image'] = f"data:image/png;base64,{superimposed_b64}"
            
            # Also include the raw heatmap
            _, heatmap_buffer = cv2.imencode('.png', np.uint8(255 * heatmap))
            heatmap_b64 = base64.b64encode(heatmap_buffer).decode('utf-8')
            explanations['grad_cam_heatmap'] = f"data:image/png;base64,{heatmap_b64}"
            
            print("✅ Grad-CAM visualization generated successfully")
        else:
            print("❌ No heatmap generated, using basic explanations")
        
        # Generate top feature explanations
        top_classes = sorted(
            explanations['confidence_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        explanations['top_features'] = [
            f"{cls}: {score:.2%}" for cls, score in top_classes
        ]
        
        print("✅ XAI explanations generated successfully")
        
    except Exception as e:
        print(f"❌ XAI generation error: {e}")
        import traceback
        print(f"📜 Traceback: {traceback.format_exc()}")
    
    return explanations


def generate_grad_cam_fixed(model, img_array, class_idx, layer_name=None):
    """
    Fixed Grad-CAM that handles complex model outputs
    """
    # Debug first
    debug_model_outputs(model, img_array)
    
    try:
        # Find appropriate convolutional layer
        if layer_name is None:
            # Look for convolutional layers
            conv_layers = []
            for layer in model.layers:
                if hasattr(layer, 'output') and len(layer.output.shape) == 4:
                    conv_layers.append(layer.name)
            
            if not conv_layers:
                print("No convolutional layers found")
                return None
            
            # Use the last convolutional layer
            layer_name = conv_layers[-1]
            print(f"Using layer for Grad-CAM: {layer_name} (from {len(conv_layers)} conv layers)")
        
        # Get the target layer
        target_layer = model.get_layer(layer_name)
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=[model.input],
            outputs=[target_layer.output, model.output]
        )
        
        # Get outputs
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            
            print(f"conv_outputs type: {type(conv_outputs)}, shape: {conv_outputs.shape}")
            print(f"predictions type: {type(predictions)}")
            
            # Extract the actual prediction tensor from whatever format it's in
            if isinstance(predictions, (list, tuple)):
                print(f"Predictions is list/tuple with {len(predictions)} elements")
                # Use the first element (main predictions)
                pred_tensor = predictions[0]
            else:
                pred_tensor = predictions
                
            print(f"Final pred_tensor shape: {pred_tensor.shape}")
            
            # Get target class score
            target_class_score = pred_tensor[0, class_idx]
            print(f"Target class score: {target_class_score}")
        
        # Compute gradients
        grads = tape.gradient(target_class_score, conv_outputs)
        
        if grads is None:
            print("Gradients are None")
            return None
        
        print(f"Gradients shape: {grads.shape}")
        
        # Global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        print(f"Pooled grads shape: {pooled_grads.shape}")
        
        # Weight the feature maps
        conv_outputs = conv_outputs[0]  # Remove batch dimension
        heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        
        # Apply ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        
        if max_val == 0:
            print("Heatmap is all zeros")
            return None
            
        heatmap /= max_val
        print(f"Heatmap generated successfully, range: [{np.min(heatmap):.3f}, {np.max(heatmap):.3f}]")
        
        return heatmap
        
    except Exception as e:
        print(f"Grad-CAM generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
def analyze_medical_features(heatmap, predicted_disease):
    """Analyze medical features from heatmap"""
    features = []

    if heatmap is not None:
        height, width = heatmap.shape

        # Analyze heatmap patterns for medical features
        border_activation = np.mean(heatmap[0:10, :]) + np.mean(
            heatmap[-10:, :]) + np.mean(heatmap[:, 0:10]) + np.mean(heatmap[:, -10:])
        center_activation = np.mean(
            heatmap[height//4:3*height//4, width//4:3*width//4])

        # Border irregularity detection
        if border_activation > center_activation * 1.5:
            features.append({
                'feature': 'border_irregularity',
                'confidence': min(float(border_activation), 0.95),
                'reasoning': 'The AI detected strong activation patterns along the lesion borders'
            })

        # Asymmetry detection (compare quadrants)
        top_left = np.mean(heatmap[:height//2, :width//2])
        top_right = np.mean(heatmap[:height//2, width//2:])
        bottom_left = np.mean(heatmap[height//2:, :width//2])
        bottom_right = np.mean(heatmap[height//2:, width//2:])

        quadrant_variance = np.var(
            [top_left, top_right, bottom_left, bottom_right])
        if quadrant_variance > 0.1:
            features.append({
                'feature': 'asymmetry',
                'confidence': min(float(quadrant_variance), 0.9),
                'reasoning': 'The activation pattern shows asymmetric distribution across lesion quadrants'
            })

    return features


def generate_xai_explanation(predicted_disease, confidence, features, is_cancer):
    """Generate XAI explanation"""
    explanations = {
        'visualExplanation': '',
        'clinicalRationale': '',
        'safetyInformation': '',
        'keyFindings': [],
        'confidenceBreakdown': {}
    }

    # Generate visual explanation
    cancer_status = "CANCEROUS" if is_cancer else "NON-CANCEROUS"
    feature_descriptions = [MEDICAL_FEATURES[f['feature']]['description']
                            for f in features] if features else ["general morphological features"]

    explanations['visualExplanation'] = (
        f"The AI model identified {predicted_disease} ({cancer_status}) based on analysis of {', '.join(feature_descriptions)}. "
        f"The confidence level for this diagnosis is {confidence:.1f}%."
    )

    # Clinical rationale
    if is_cancer:
        explanations['clinicalRationale'] = (
            f"This lesion shows features consistent with {predicted_disease}. "
            "Immediate dermatological consultation is recommended for proper diagnosis and treatment planning."
        )
        explanations['safetyInformation'] = "URGENT: This requires professional medical evaluation. Do not delay consultation."
    else:
        explanations['clinicalRationale'] = (
            f"The lesion characteristics are consistent with {predicted_disease}. "
            "Regular monitoring is advised to detect any changes."
        )
        explanations['safetyInformation'] = "Continue regular skin checks and consult a dermatologist for any changes."

    # Key findings from detected features
    explanations['keyFindings'] = [
        {
            'finding': f['feature'],
            'description': MEDICAL_FEATURES[f['feature']]['description'],
            'significance': MEDICAL_FEATURES[f['feature']]['significance'].get(predicted_disease, 'Relevant feature for diagnosis'),
            'confidence': f['confidence']
        } for f in features
    ]

    # Confidence breakdown
    explanations['confidenceBreakdown'] = {
        'modelConfidence': confidence,
        # Simplified calculation
        'featureConsistency': len(features) * 0.15,
        'clinicalCorrelation': 0.8 if is_cancer else 0.9  # Based on disease type
    }

    return explanations


def generate_basic_explanations(disease, confidence, is_cancer):
    """Generate basic explanations when XAI fails"""
    cancer_status = "CANCEROUS" if is_cancer else "NON-CANCEROUS"

    return {
        'visualExplanation': f"The AI model identified {disease} ({cancer_status}) with {confidence:.1f}% confidence.",
        'clinicalRationale': f"This diagnosis is based on analysis of visual patterns in the skin lesion that are characteristic of {disease}.",
        'safetyInformation': "URGENT: Professional medical evaluation required." if is_cancer else "Regular monitoring recommended.",
        'keyFindings': [
            {
                'finding': 'pattern_analysis',
                'description': 'AI-identified visual patterns',
                'significance': 'The model detected features consistent with this diagnosis',
                'confidence': confidence / 100
            }
        ],
        'confidenceBreakdown': {
            'modelConfidence': confidence,
            'featureConsistency': 0.7,
            'clinicalCorrelation': 0.8 if is_cancer else 0.9
        }
    }


def get_mock_retraining_metrics():
    """Return mock retraining metrics for development"""
    # Generate mock accuracy trends for the last 30 days
    accuracy_trends = []
    base_date = datetime.now() - timedelta(days=30)

    for i in range(30):
        date = base_date + timedelta(days=i)
        accuracy_trends.append({
            'date': date.strftime('%Y-%m-%d'),
            'accuracy': round(75 + random.uniform(-5, 10), 2),
            'verification_count': random.randint(5, 20),
            'avg_confidence': round(80 + random.uniform(-10, 5), 2)
        })

    return {
        'accuracy_trends': accuracy_trends,
        'retraining_status': retraining_status,
        'model_health': {
            'last_retraining': retraining_status.get('last_retraining'),
            'accuracy_improvement': retraining_status.get('accuracy_improvement', 0),
            'is_retraining': retraining_status.get('is_retraining', False)
        },
        'message': 'Using mock data for development'
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Accept either 'image' or 'file' parameter
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        # Use 'image' if available, otherwise use 'file'
        file = request.files.get('image') or request.files.get('file')
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # Create uploads directory if it doesn't exist
        os.makedirs('uploads', exist_ok=True)

        # Save the uploaded file
        filename = secure_filename(file.filename)
        image_path = os.path.join('uploads', filename)
        file.save(image_path)
        print(f"Processing image: {filename}")

        # Load and preprocess image
        try:
            img = image.load_img(image_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            original_image = np.array(img)  # Keep original for heatmap
            print(f"Image loaded successfully. Shape: {img_array.shape}")
        except Exception as e:
            print(f"Image loading error: {str(e)}")
            return jsonify({'error': f'Invalid image format: {str(e)}'}), 400

        # Normalize and batch the image
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Final image array shape: {img_array.shape}")

        # Check if model is loaded
        if unified_model is None:
            return jsonify({'error': 'AI model not loaded. Please try again later.'}), 500

        # Verify model input shape
        expected_shape = unified_model.input_shape
        print(f"Model expects input shape: {expected_shape}")
        print(f"Actual input shape: {img_array.shape}")

        if img_array.shape[1:] != expected_shape[1:]:
            return jsonify({
                'error': f'Image shape mismatch. Expected {expected_shape[1:]}, got {img_array.shape[1:]}'
            }), 400

        # Make prediction
        print("Running ResNet model prediction...")
        predictions = unified_model.predict(img_array)
        print(f"Raw predictions: {predictions}")

        # Class names and mapping
        class_names = [
            'Melanocytic nevi', 'Melanoma', 'Benign keratosis-like lesions',
            'Basal cell carcinoma', 'Actinic keratoses', 'Vascular lesions', 'Dermatofibroma'
        ]
        ham10000_classes = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
        cancer_mapping = {
            'nv': False, 'mel': True, 'bkl': False, 'bcc': True,
            'akiec': True, 'vasc': False, 'df': False
        }

        # Top prediction
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_ham_class = ham10000_classes[predicted_class_idx]
        is_cancer = cancer_mapping.get(predicted_ham_class, False)
        cancer_status = "CANCEROUS" if is_cancer else "NON-CANCEROUS"

        print(f"Predicted class index: {predicted_class_idx}")
        print(f"Predicted class: {class_names[predicted_class_idx]}")
        print(f"Confidence: {confidence:.4f}")

        # Top 3 predictions for confidence breakdown
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {'class': class_names[i], 'confidence': float(predictions[0][i])}
            for i in top_3_idx
        ]

        # Generate comprehensive XAI explanations
        xai_explanations = {}
        dynamic_explanations = {}
        
        try:
            print("🔄 Generating XAI explanations...")
            
            # Generate Grad-CAM heatmap
            heatmap = generate_grad_cam(unified_model, img_array, predicted_class_idx)

            if heatmap is not None:
                print("✅ Grad-CAM generated successfully")
                
                # Convert heatmap to base64 for frontend
                heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
                
                # Convert heatmap to RGB
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                
                # Convert original image to RGB if needed
                if len(original_image.shape) == 2:  # Grayscale
                    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
                else:
                    original_image_rgb = original_image
                
                # Ensure both images have the same data type and size
                original_image_rgb = cv2.resize(original_image_rgb, (heatmap_colored.shape[1], heatmap_colored.shape[0]))
                original_image_rgb = original_image_rgb.astype(np.float32)
                heatmap_colored = heatmap_colored.astype(np.float32)
                
                # Superimpose heatmap on original image
                superimposed = cv2.addWeighted(original_image_rgb, 0.6, heatmap_colored, 0.4, 0)
                superimposed = np.uint8(np.clip(superimposed, 0, 255))
                
                # Convert to base64 for web display
                _, superimposed_buffer = cv2.imencode('.png', superimposed)
                superimposed_b64 = base64.b64encode(superimposed_buffer).decode('utf-8')
                
                _, heatmap_buffer = cv2.imencode('.png', np.uint8(255 * heatmap_resized))
                heatmap_b64 = base64.b64encode(heatmap_buffer).decode('utf-8')
                
                print("✅ Heatmap images converted to base64")

                # Analyze medical features from heatmap
                detected_features = analyze_medical_features(heatmap, class_names[predicted_class_idx])
                print(f"🔍 Detected {len(detected_features)} medical features")

                # Generate dynamic explanations (existing format)
                dynamic_explanations = generate_xai_explanation(
                    class_names[predicted_class_idx],
                    confidence * 100,
                    detected_features,
                    is_cancer
                )

                # Generate new XAI explanations with heatmaps
                xai_explanations = {
                    'grad_cam_heatmap': f"data:image/png;base64,{heatmap_b64}",
                    'superimposed_image': f"data:image/png;base64,{superimposed_b64}",
                    'confidence_scores': {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))},
                    'explanation_text': f"The model predicted '{class_names[predicted_class_idx]}' with {confidence:.2%} confidence.",
                    'top_features': [f"{class_names[i]}: {predictions[0][i]:.2%}" for i in top_3_idx],
                    'all_probabilities': {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
                }

                print("✅ Comprehensive XAI explanations generated successfully")
            else:
                print("❌ No heatmap generated, using basic explanations")
                # Fallback to basic explanations without heatmaps
                dynamic_explanations = generate_basic_explanations(
                    class_names[predicted_class_idx],
                    confidence * 100,
                    is_cancer
                )
                xai_explanations = {
                    'grad_cam_heatmap': None,
                    'superimposed_image': None,
                    'confidence_scores': {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))},
                    'explanation_text': f"The model predicted '{class_names[predicted_class_idx]}' with {confidence:.2%} confidence.",
                    'top_features': [f"{class_names[i]}: {predictions[0][i]:.2%}" for i in top_3_idx],
                    'all_probabilities': {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
                }

        except Exception as e:
            print(f"❌ XAI generation error: {str(e)}")
            import traceback
            print(f"📜 Traceback: {traceback.format_exc()}")
            # Fallback to basic explanations
            dynamic_explanations = generate_basic_explanations(
                class_names[predicted_class_idx],
                confidence * 100,
                is_cancer
            )
            xai_explanations = {
                'grad_cam_heatmap': None,
                'superimposed_image': None,
                'confidence_scores': {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))},
                'explanation_text': f"The model predicted '{class_names[predicted_class_idx]}' with {confidence:.2%} confidence.",
                'top_features': [f"{class_names[i]}: {predictions[0][i]:.2%}" for i in top_3_idx],
                'all_probabilities': {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
            }
               
        # Update community insights
        update_community_insights(class_names[predicted_class_idx], is_cancer)

        # Prepare response with corrected structure
        result = {
            'diagnosis': {
                'disease': class_names[predicted_class_idx],
                'confidence': confidence * 100,
                'cancerStatus': cancer_status,
                'isCancer': is_cancer
            },
            'explanations': {
                'dynamic': dynamic_explanations,
                'xai_explanations': xai_explanations
            },
            'top_predictions': top_3_predictions,
            'detected_features': detected_features if 'detected_features' in locals() else [],
            'has_heatmap': heatmap is not None if 'heatmap' in locals() else False,
            'model_info': {
                'model_type': 'ResNet',
                'input_size': '128x128',
                'classes_detected': len(class_names)
            }
        }

        print(f"ResNet model analysis completed successfully: {result['diagnosis']['disease']}")
        print(f"XAI explanations included: {len(xai_explanations) > 0}")
        print(f"Heatmap available: {result['has_heatmap']}")
        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/community-insights', methods=['GET'])
def get_community_insights():
    """Get community health insights"""
    try:
        period = request.args.get('period', 'month')
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
            
        cursor = conn.cursor(dictionary=True)
        
        if period == 'day':
            query = "SELECT * FROM community_insights WHERE date = CURDATE()"
        elif period == 'week':
            query = "SELECT * FROM community_insights WHERE date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)"
        else:  # month
            query = "SELECT * FROM community_insights WHERE date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"
            cursor.execute(query)
        insights_data = cursor.fetchall()
        
        # Calculate totals
        total_scans = 0
        total_benign = 0
        total_malignant = 0
        
        for insight in insights_data:
            total_scans += insight['total_scans']
            total_benign += insight['benign_count']
            total_malignant += insight['malignant_count']
        
        # Calculate percentages
        benign_percentage = (total_benign / total_scans * 100) if total_scans > 0 else 0
        malignant_percentage = (total_malignant / total_scans * 100) if total_scans > 0 else 0
        cursor.close()
        conn.close()
        
        return jsonify({
            'period': period,
            'total_scans': total_scans,
            'benign': {
                'count': total_benign,
                'percentage': round(benign_percentage, 1)
            },
            'malignant': {
                'count': total_malignant,
                'percentage': round(malignant_percentage, 1)
            },
             'health_tips': [
                f"📊 {total_scans} scans this {period}",
                f"✅ {round(benign_percentage, 1)}% were benign lesions",
                f"⚠️ {round(malignant_percentage, 1)}% required medical attention",
                "🔍 Regular self-exams help in early detection"
            ]
        })
        
    except Exception as e:
        print(f"Community insights error: {e}")
        return jsonify({'error': 'Failed to fetch community insights'}), 500
    
@app.route('/api/preventive-care', methods=['GET'])
def get_preventive_care():
    """Get preventive care information"""
    try:
        disease_type = request.args.get('disease', 'general')
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
            
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT * FROM preventive_care WHERE disease_type = %s", (disease_type,))
        care_data = cursor.fetchone()
        
        if not care_data:
            cursor.execute("SELECT * FROM preventive_care WHERE disease_type = 'general'")
            care_data = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if care_data:
            return jsonify({
                'disease_type': care_data['disease_type'],
                'prevention_tips': json.loads(care_data['prevention_tips']),
                'risk_factors': json.loads(care_data['risk_factors']),
                'early_signs': json.loads(care_data['early_signs'])
            })
        else:
            return jsonify({'error': 'No preventive care data found'}), 404
            
    except Exception as e:
        print(f"Preventive care error: {e}")
        return jsonify({'error': 'Failed to fetch preventive care information'}), 500
    
            
@app.route('/api/xai-debug', methods=['POST'])
def xai_debug():
    """Debug endpoint for XAI functionality"""
    try:
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files.get('image') or request.files.get('file')
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # Save and process image
        filename = secure_filename(file.filename)
        image_path = os.path.join('uploads', filename)
        file.save(image_path)

        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if unified_model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Get prediction
        predictions = unified_model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        class_names = [
            'Melanocytic nevi', 'Melanoma', 'Benign keratosis-like lesions',
            'Basal cell carcinoma', 'Actinic keratoses', 'Vascular lesions', 'Dermatofibroma'
        ]

        # Test Grad-CAM
        heatmap = generate_grad_cam(
            unified_model, img_array, predicted_class_idx)

        # Test medical feature analysis
        features = []
        if heatmap is not None:
            features = analyze_medical_features(
                np.array(heatmap), class_names[predicted_class_idx])

        # Test XAI explanations
        ham10000_classes = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
        cancer_mapping = {
            'nv': False, 'mel': True, 'bkl': False, 'bcc': True,
            'akiec': True, 'vasc': False, 'df': False
        }
        predicted_ham_class = ham10000_classes[predicted_class_idx]
        is_cancer = cancer_mapping.get(predicted_ham_class, False)

        explanations = generate_xai_explanation(
            class_names[predicted_class_idx],
            confidence * 100,
            features,
            is_cancer
        )

        return jsonify({
            'debug_info': {
                'model_loaded': unified_model is not None,
                'prediction_made': True,
                'grad_cam_success': heatmap is not None,
                'features_detected': len(features),
                'explanations_generated': len(explanations) > 0
            },
            'prediction': {
                'class': class_names[predicted_class_idx],
                'confidence': confidence,
                'is_cancer': is_cancer
            },
            'xai_components': {
                'heatmap_shape': np.array(heatmap).shape if heatmap is not None else None,
                'detected_features': features,
                'explanations': explanations
            }
        })

    except Exception as e:
        return jsonify({'error': f'XAI debug failed: {str(e)}'}), 500


@app.route('/api/verify-diagnosis', methods=['POST'])
def verify_diagnosis():
    """Doctor verification endpoint"""
    try:
        data = request.get_json()

        required_fields = ['originalDiagnosis',
                           'verifiedDiagnosis', 'doctorId', 'imageId', 'isCorrect']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Save verification to database
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500

        cursor = conn.cursor()

        query = """
			INSERT INTO doctor_verifications 
			(original_diagnosis, verified_diagnosis, doctor_id, image_id, is_correct, 
			 confidence_score, notes, created_at)
			VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
		"""

        values = (
            data['originalDiagnosis'],
            data['verifiedDiagnosis'],
            data['doctorId'],
            data['imageId'],
            data['isCorrect'],
            data.get('confidenceScore', 0),
            data.get('notes', '')
        )

        cursor.execute(query, values)
        conn.commit()
        verification_id = cursor.lastrowid

        cursor.close()
        conn.close()

        # If diagnosis was incorrect, flag for model retraining
        if not data['isCorrect']:
            print(f"Model correction needed for image {data['imageId']}")

        return jsonify({
            'success': True,
            'verificationId': verification_id,
            'message': 'Verification saved successfully'
        })

    except Exception as e:
        print(f'Verification error: {e}')
        return jsonify({'error': 'Failed to save verification'}), 500


@app.route('/api/model-performance', methods=['GET'])
def get_model_performance():
    """Get model performance metrics for retraining decisions"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'performance': {
                    'accuracy': 0,
                    'totalVerifications': 0,
                    'correctPredictions': 0,
                    'averageConfidence': 0,
                    'evaluationPeriod': 'No data - database connection failed'
                },
                'retrainingRecommended': False,
                'message': 'Database connection failed'
            })

        cursor = conn.cursor(dictionary=True)

        # Calculate accuracy based on doctor verifications
        query = """
			SELECT 
				COUNT(*) as total_verifications,
				SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
				AVG(confidence_score) as avg_confidence
			FROM doctor_verifications 
			WHERE created_at >= DATE_SUB(NOW(), INTERVAL 15 DAY)
		"""

        cursor.execute(query)
        performance_data = cursor.fetchone()

        cursor.close()
        conn.close()

        accuracy = (performance_data['correct_predictions'] / performance_data['total_verifications']
                    * 100) if performance_data['total_verifications'] > 0 else 0

        return jsonify({
            'performance': {
                'accuracy': round(accuracy, 2),
                'totalVerifications': performance_data['total_verifications'],
                'correctPredictions': performance_data['correct_predictions'],
                'averageConfidence': round(performance_data['avg_confidence'] or 0, 2),
                'evaluationPeriod': '15 days'
            },
            # Retrain if accuracy below 85%
            'retrainingRecommended': accuracy < 85.0
        })

    except Exception as e:
        print(f'Performance query error: {e}')
        return jsonify({
            'performance': {
                'accuracy': 0,
                'totalVerifications': 0,
                'correctPredictions': 0,
                'averageConfidence': 0,
                'evaluationPeriod': 'Error fetching data'
            },
            'retrainingRecommended': False,
            'error': 'Failed to fetch performance data'
        })


@app.route('/api/retrain-model', methods=['POST'])
def retrain_model():
    """Trigger model retraining with verified data"""
    global retraining_status

    if retraining_status['is_retraining']:
        return jsonify({
            'success': False,
            'message': 'Retraining already in progress',
            'progress': retraining_status['progress']
        }), 409

    try:
        # Start retraining in background thread
        thread = threading.Thread(target=retrain_model_async)
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Model retraining process started successfully',
            'estimatedCompletion': '30-60 minutes',
            'statusEndpoint': '/api/retraining-status'
        })

    except Exception as e:
        print(f'Retraining error: {e}')
        return jsonify({'error': 'Failed to start retraining process', 'details': str(e)}), 500


@app.route('/api/retraining-status', methods=['GET'])
def get_retraining_status():
    """Get current retraining status"""
    return jsonify(retraining_status)


@app.route('/api/retraining-metrics', methods=['GET'])
def get_retraining_metrics():
    """Get retraining performance metrics with fallback to mock data"""
    try:
        # Try to get real data first
        conn = get_db_connection()
        if not conn:
            mock_data = get_mock_retraining_metrics()
            return jsonify(mock_data)

        cursor = conn.cursor(dictionary=True)

        # Get retraining history with better error handling
        query = """
			SELECT 
				DATE(created_at) as date,
				COUNT(*) as verification_count,
				AVG(confidence_score) as avg_confidence,
				SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_count
			FROM doctor_verifications 
			WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
			GROUP BY DATE(created_at)
			ORDER BY date DESC
			LIMIT 30
		"""

        cursor.execute(query)
        metrics = cursor.fetchall()

        cursor.close()
        conn.close()

        # Calculate accuracy trends
        accuracy_trends = []
        for metric in metrics:
            accuracy = (metric['correct_count'] / metric['verification_count']
                        ) * 100 if metric['verification_count'] > 0 else 0
            accuracy_trends.append({
                'date': metric['date'].isoformat() if hasattr(metric['date'], 'isoformat') else str(metric['date']),
                'accuracy': round(accuracy, 2),
                'verification_count': metric['verification_count'],
                'avg_confidence': round(float(metric['avg_confidence'] or 0), 2)
            })

        return jsonify({
            'accuracy_trends': accuracy_trends,
            'retraining_status': retraining_status,
            'model_health': {
                'last_retraining': retraining_status.get('last_retraining'),
                'accuracy_improvement': retraining_status.get('accuracy_improvement', 0),
                'is_retraining': retraining_status.get('is_retraining', False)
            }
        })

    except Exception as e:
        print(f"Retraining metrics error, using mock data: {e}")
        mock_data = get_mock_retraining_metrics()
        return jsonify(mock_data)


@app.route('/api/force-retrain', methods=['POST'])
def force_retrain():
    """Force retraining regardless of data size (admin function)"""
    try:
        # Check if retraining is already in progress
        if retraining_status['is_retraining']:
            return jsonify({
                'success': False,
                'message': 'Retraining already in progress'
            }), 409

        # Start forced retraining
        thread = threading.Thread(target=retrain_model_async)
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Forced retraining started',
            'warning': 'This may use limited data and produce suboptimal results'
        })

    except Exception as e:
        print(f'Force retrain error: {e}')
        return jsonify({'error': 'Failed to start forced retraining'}), 500


@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to check backend functionality"""
    return jsonify({
        'status': 'success',
        'message': 'Backend is running correctly',
        'model_loaded': unified_model is not None,
        'database_connected': get_db_connection() is not None,
        'timestamp': datetime.now().isoformat()
    })


def retrain_model_async():
    """Asynchronous model retraining function"""
    global retraining_status, unified_model

    try:
        retraining_status.update({
            'is_retraining': True,
            'progress': 0,
            'current_step': 'Initializing retraining process...',
            'start_time': datetime.now().isoformat(),
            'error': None
        })

        # Step 1: Collect verified data
        retraining_status.update({
            'progress': 10,
            'current_step': 'Collecting verified training data...'
        })

        verified_data = collect_verified_data()
        if len(verified_data) < 50:  # Minimum samples required
            retraining_status.update({
                'is_retraining': False,
                'error': f'Insufficient verified data. Need at least 50 samples, got {len(verified_data)}'
            })
            return

        # Step 2: Prepare datasets
        retraining_status.update({
            'progress': 20,
            'current_step': 'Preparing training datasets...'
        })

        X_train, X_val, y_train, y_val = prepare_training_data(verified_data)

        # Step 3: Evaluate current model
        retraining_status.update({
            'progress': 30,
            'current_step': 'Evaluating current model performance...'
        })

        old_accuracy = evaluate_model_on_verified_data(X_val, y_val)

        # Step 4: Retrain model
        retraining_status.update({
            'progress': 40,
            'current_step': 'Starting model retraining...'
        })

        new_model = perform_retraining(X_train, X_val, y_train, y_val)

        # Step 5: Evaluate new model
        retraining_status.update({
            'progress': 80,
            'current_step': 'Evaluating new model...'
        })

        new_accuracy = evaluate_new_model(new_model, X_val, y_val)
        accuracy_improvement = new_accuracy - old_accuracy

        # Step 6: Decide whether to deploy new model
        retraining_status.update({
            'progress': 90,
            'current_step': 'Finalizing retraining process...'
        })

        if new_accuracy > old_accuracy and accuracy_improvement >= 0.01:  # At least 1% improvement
            deploy_new_model(new_model)
            retraining_status.update({
                'accuracy_improvement': round(accuracy_improvement * 100, 2),
                'new_accuracy': round(new_accuracy * 100, 2),
                'old_accuracy': round(old_accuracy * 100, 2)
            })
        else:
            retraining_status.update({
                'accuracy_improvement': round(accuracy_improvement * 100, 2),
                'message': 'New model not deployed - insufficient improvement'
            })

        # Step 7: Update status
        retraining_status.update({
            'is_retraining': False,
            'progress': 100,
            'current_step': 'Retraining completed successfully',
            'last_retraining': datetime.now().isoformat(),
            'completion_time': datetime.now().isoformat()
        })

        print("Model retraining completed successfully")

    except Exception as e:
        retraining_status.update({
            'is_retraining': False,
            'error': str(e),
            'current_step': f'Error: {str(e)}',
            'completion_time': datetime.now().isoformat()
        })
        print(f"Retraining failed: {e}")


def collect_verified_data():
    """Collect verified data from database"""
    conn = get_db_connection()
    if not conn:
        raise Exception("Database connection failed")

    try:
        cursor = conn.cursor(dictionary=True)

        # Query to get verified images and their correct diagnoses
        query = """
			SELECT 
				dv.verified_diagnosis as true_label,
				dv.image_id,
				ah.image_path,
				dv.confidence_score,
				dv.created_at
			FROM doctor_verifications dv
			JOIN analysis_history ah ON dv.image_id = ah.id
			WHERE dv.is_correct = 0  # Doctor corrected the diagnosis
				AND dv.verified_diagnosis IS NOT NULL
				AND ah.image_path IS NOT NULL
			ORDER BY dv.created_at DESC
			LIMIT 1000
		"""

        cursor.execute(query)
        verified_data = cursor.fetchall()

        # Also include correct predictions for balance
        query_correct = """
			SELECT 
				ah.diagnosis as true_label,
				ah.id as image_id,
				ah.image_path,
				1.0 as confidence_score,
				ah.created_at
			FROM analysis_history ah
			JOIN doctor_verifications dv ON ah.id = dv.image_id
			WHERE dv.is_correct = 1  # Doctor confirmed the diagnosis was correct
				AND ah.image_path IS NOT NULL
			ORDER BY ah.created_at DESC
			LIMIT 500
		"""

        cursor.execute(query_correct)
        correct_data = cursor.fetchall()

        all_data = verified_data + correct_data

        cursor.close()
        conn.close()

        print(f"Collected {len(all_data)} verified samples for retraining")
        return all_data

    except Exception as e:
        conn.close()
        raise Exception(f"Failed to collect verified data: {e}")


def prepare_training_data(verified_data):
    """Prepare training data from verified samples"""
    images = []
    labels = []

    for data in verified_data:
        try:
            # Load and preprocess image
            if os.path.exists(data['image_path']):
                image = load_and_preprocess_image(data['image_path'])
                images.append(image)

                # Convert label to class index
                label = data['true_label']
                class_index = HAM10000_CLASSES.index(
                    label) if label in HAM10000_CLASSES else -1
                if class_index != -1:
                    labels.append(class_index)

        except Exception as e:
            print(f"Error processing image {data['image_path']}: {e}")
            continue

    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Remove any samples with invalid labels
    valid_indices = y != -1
    X = X[valid_indices]
    y = y[valid_indices]

    if len(X) == 0:
        raise Exception("No valid training data available")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(
        f"Training data prepared: {len(X_train)} training, {len(X_val)} validation samples")
    return X_train, X_val, y_train, y_val


def load_and_preprocess_image(image_path):
    """Load and preprocess single image"""
    image = Image.open(image_path)

    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to match model input
    image = image.resize((224, 224))

    # Convert to array and preprocess
    image_array = np.array(image)
    image_array = tf.keras.applications.resnet50.preprocess_input(image_array)

    return image_array


def evaluate_model_on_verified_data(X_val, y_val):
    """Evaluate current model on verified data"""
    global unified_model

    if unified_model is None:
        return 0.0

    try:
        predictions = unified_model.predict(X_val)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(y_val, predicted_classes)

        print(f"Current model accuracy on verified data: {accuracy:.4f}")
        return accuracy

    except Exception as e:
        print(f"Error evaluating current model: {e}")
        return 0.0


def perform_retraining(X_train, X_val, y_train, y_val):
    """Perform the actual model retraining"""
    global unified_model, retraining_status

    try:
        # Create a copy of the current model for retraining
        if unified_model is None:
            raise Exception("No base model available for retraining")

        # Strategy 1: Fine-tune the existing model
        model = tf.keras.models.clone_model(unified_model)
        model.set_weights(unified_model.get_weights())

        # Unfreeze some layers for fine-tuning
        for layer in model.layers[-20:]:  # Last 20 layers
            layer.trainable = True

        # Compile with lower learning rate for fine-tuning
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: update_training_progress(
                    epoch, logs)
            )
        ]

        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )

        return model

    except Exception as e:
        raise Exception(f"Model retraining failed: {e}")


def update_training_progress(epoch, logs):
    """Update training progress during retraining"""
    global retraining_status

    progress = 40 + (epoch * 50 / 30)  # From 40% to 90% over 30 epochs
    retraining_status['progress'] = min(progress, 90)
    retraining_status[
        'current_step'] = f'Training epoch {epoch+1}/30 - Accuracy: {logs.get("accuracy", 0):.4f}'


def evaluate_new_model(new_model, X_val, y_val):
    """Evaluate the new retrained model"""
    try:
        predictions = new_model.predict(X_val)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(y_val, predicted_classes)

        print(f"New model accuracy: {accuracy:.4f}")
        return accuracy

    except Exception as e:
        print(f"Error evaluating new model: {e}")
        return 0.0


def deploy_new_model(new_model):
    """Deploy the new model and backup the old one"""
    global unified_model

    try:
        # Create backup directory
        backup_dir = 'models/backup'
        os.makedirs(backup_dir, exist_ok=True)

        # Backup current model
        if unified_model is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{backup_dir}/resnet_model_backup_{timestamp}.h5"
            unified_model.save(backup_path)
            print(f"Old model backed up to: {backup_path}")

        # Save new model
        new_model_path = 'models/resnet_model.h5'
        new_model.save(new_model_path)

        # Load the new model
        unified_model = tf.keras.models.load_model(new_model_path)

        # Update model info
        retraining_status['model_deployed'] = True
        retraining_status['deployment_time'] = datetime.now().isoformat()

        print("New model deployed successfully")

        # Clean up old backups (keep only last 5)
        cleanup_old_backups(backup_dir)

    except Exception as e:
        raise Exception(f"Model deployment failed: {e}")


def cleanup_old_backups(backup_dir):
    """Clean up old model backups, keep only last 5"""
    try:
        backups = []
        for file in os.listdir(backup_dir):
            if file.startswith('resnet_model_backup_') and file.endswith('.h5'):
                file_path = os.path.join(backup_dir, file)
                backups.append((file_path, os.path.getctime(file_path)))

        # Sort by creation time (oldest first)
        backups.sort(key=lambda x: x[1])

        # Remove oldest backups, keep only last 5
        if len(backups) > 5:
            for i in range(len(backups) - 5):
                os.remove(backups[i][0])
                print(f"Removed old backup: {backups[i][0]}")

    except Exception as e:
        print(f"Error cleaning up backups: {e}")


@app.route('/suggest_treatment', methods=['POST'])
def suggest_treatment():
    """Dynamic Treatment Suggestions based on diagnosis"""
    try:
        data = request.get_json()
        disease = data.get('disease')
        is_cancer = data.get('isCancer')
        confidence = data.get('confidence')
        detected_features = data.get('detectedFeatures', [])

        if not disease:
            return jsonify({'error': 'Disease parameter is required'}), 400

        # Generate treatment suggestions based on diagnosis
        treatment_suggestions = {
            'urgency': 'HIGH' if is_cancer and confidence > 80 else 'MEDIUM' if is_cancer else 'LOW',
            'recommendations': [
                'Consult dermatologist for professional evaluation',
                'Biopsy recommended for confirmation' if is_cancer else 'Regular monitoring advised'
            ],
            'nextSteps': [
                'Schedule appointment with specialist',
                'Document lesion characteristics',
                'Follow up in 3 months' if not is_cancer else 'Immediate evaluation needed'
            ]
        }

        return jsonify({
            'disease': disease,
            'isCancer': is_cancer,
            'confidence': confidence,
            'treatmentSuggestions': treatment_suggestions,
            'detectedFeatures': detected_features,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f'Treatment suggestion error: {e}')
        return jsonify({'error': 'Failed to generate treatment suggestions'}), 500


@app.route('/api/dermatologists', methods=['GET'])
def get_dermatologists():
    """Find dermatologists near location"""
    try:
        lat = request.args.get('lat')
        lng = request.args.get('lng')
        radius = request.args.get('radius', 10)

        if not lat or not lng:
            return jsonify({'message': 'Latitude and longitude are required.'}), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({'message': 'Database connection failed'}), 500

        cursor = conn.cursor(dictionary=True)

        query = """
			SELECT id, name, specialty, experience, rating, address, phone, email,
				   latitude, longitude,
				   ( 6371 * acos( cos( radians(%s) ) * cos( radians( latitude ) ) * 
					 cos( radians( longitude ) - radians(%s) ) + sin( radians(%s) ) * 
					 sin( radians( latitude ) ) ) ) AS distance
			FROM dermatologists
			HAVING distance < %s
			ORDER BY distance, rating DESC
			LIMIT 20;
		"""

        cursor.execute(query, (float(lat), float(
            lng), float(lat), float(radius)))
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify(results)

    except Exception as e:
        print(f"Dermatologists query error: {e}")
        return jsonify({'message': 'Server error'}), 500


@app.route('/api/analysis-history', methods=['POST'])
def save_analysis_history():
    """Save analysis history with doctor verification data"""
    try:
        data = request.get_json()
        user_id = data.get('userId')
        analysis = data.get('analysis')
        verification_data = data.get('verification', {})

        if not user_id or not analysis:
            return jsonify({'message': 'User ID and analysis data are required'}), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({'message': 'Database connection failed'}), 500

        cursor = conn.cursor()

        query = """
			INSERT INTO analysis_history 
			(user_id, image_path, diagnosis, confidence, is_cancer, cancer_status, 
			 explanations, doctor_verified, doctor_correction, created_at)
			VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
		"""

        values = (
            user_id,
            analysis.get('imagePath'),
            analysis.get('diagnosis'),
            analysis.get('confidence'),
            analysis.get('isCancer'),
            analysis.get('cancerStatus'),
            analysis.get('explanations'),
            verification_data.get('verified', False),
            verification_data.get('correctedDiagnosis', None)
        )

        cursor.execute(query, values)
        conn.commit()

        history_id = cursor.lastrowid
        cursor.close()
        conn.close()

        return jsonify({'success': True, 'historyId': history_id})

    except Exception as e:
        print(f"Analysis history error: {e}")
        return jsonify({'message': 'Failed to save analysis history'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = {
        'unifiedModel': unified_model is not None,
        'database': get_db_connection() is not None
    }

    return jsonify({
        'status': 'healthy',
        'models': model_status,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    try:
        # Initialize database first
        print("Initializing database...")
        initialize_database()

        # Load model on startup
        load_models()
        print("Starting server...")

        # Create necessary directories
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('explanations', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/backup', exist_ok=True)

        print(f"Starting Flask server on port {PORT}")
        app.run(host='0.0.0.0', port=int(PORT), debug=True)
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        print("Starting server without AI functionality...")
        app.run(host='0.0.0.0', port=int(PORT), debug=True)
