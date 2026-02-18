"""
Flask API for ML predictions
Serves the trained model via REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model on startup
print("\n" + "=" * 70)
print(" " * 15 + "ProManage ML Prediction API")
print("=" * 70)
print("\nLoading trained model...")

try:
    model_data = joblib.load('models/project_predictor.pkl')
    model = model_data['model']
    label_encoders = model_data['label_encoders']
    feature_columns = model_data['feature_columns']
    accuracy = model_data['accuracy']
    print(f"✓ Model loaded successfully")
    print(f"✓ Model Accuracy: {accuracy*100:.2f}%")
except Exception as e:
    print(f"❌ ERROR: Could not load model: {e}")
    exit(1)

# Valid options
PROJECT_TYPES = [
    'Website Development', 'Mobile App', 'API Integration',
    'Database Migration', 'E-commerce Platform', 'Landing Page',
    'CRM System', 'ERP Module', 'Analytics Dashboard', 'Security Audit',
    'Inventory System', 'POS System', 'Booking Platform'
]

COMPLEXITY_LEVELS = ['Low', 'Medium', 'High']

@app.route('/')
def home():
    """API information endpoint"""
    return jsonify({
        'status': 'running',
        'name': 'ProManage ML Prediction API',
        'version': '1.0.0',
        'model_accuracy': f'{accuracy*100:.2f}%',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'info': 'GET /',
            'model_details': 'GET /api/model-info',
            'predict': 'POST /api/predict',
            'health': 'GET /api/health'
        }
    })

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model-info')
def model_info():
    """Return model information"""
    return jsonify({
        'accuracy': float(accuracy),
        'precision': float(model_data.get('precision', 0)),
        'recall': float(model_data.get('recall', 0)),
        'f1_score': float(model_data.get('f1_score', 0)),
        'features': feature_columns,
        'feature_importance': model_data.get('feature_importance', []),
        'project_types': PROJECT_TYPES,
        'complexity_levels': COMPLEXITY_LEVELS
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction"""
    try:
        # Parse request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate required fields
        required_fields = [
            'projectType', 'deadline', 'estimatedRevenue',
            'teamSize', 'complexity', 'clientSatisfaction'
        ]
        
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Extract and validate data
        project_type = data['projectType']
        deadline = int(data['deadline'])
        estimated_revenue = float(data['estimatedRevenue'])
        team_size = int(data['teamSize'])
        complexity = data['complexity']
        client_satisfaction = float(data['clientSatisfaction'])
        
        # Validate ranges
        if not (1 <= deadline <= 5):
            return jsonify({
                'success': False,
                'error': 'Deadline must be between 1 and 5 days'
            }), 400
        
        if estimated_revenue <= 0:
            return jsonify({
                'success': False,
                'error': 'Estimated revenue must be positive'
            }), 400
        
        if not (1 <= team_size <= 10):
            return jsonify({
                'success': False,
                'error': 'Team size must be between 1 and 10'
            }), 400
        
        if not (1.0 <= client_satisfaction <= 5.0):
            return jsonify({
                'success': False,
                'error': 'Client satisfaction must be between 1.0 and 5.0'
            }), 400
        
        # Encode categorical features
        try:
            project_type_encoded = label_encoders['project_type'].transform([project_type])[0]
        except:
            # If project type not in training data, use most common (Website Development = 0)
            project_type_encoded = 0
            print(f"⚠️  Unknown project type '{project_type}', using default encoding")
        
        try:
            complexity_encoded = label_encoders['complexity'].transform([complexity])[0]
        except:
            complexity_encoded = 1  # Default to Medium
            print(f"⚠️  Unknown complexity '{complexity}', using Medium")
        
        # Calculate derived features
        revenue_per_person = estimated_revenue / team_size
        deadline_revenue_ratio = estimated_revenue / (deadline * 10000)
        
        # Create feature dataframe
        features = pd.DataFrame({
            'deadline_days': [deadline],
            'estimated_revenue': [estimated_revenue],
            'team_size': [team_size],
            'project_type_encoded': [project_type_encoded],
            'complexity_encoded': [complexity_encoded],
            'client_satisfaction': [client_satisfaction],
            'revenue_per_person': [revenue_per_person],
            'deadline_revenue_ratio': [deadline_revenue_ratio]
        })
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Calculate confidence
        confidence = float(max(probability))
        prob_profitable = float(probability[1])
        prob_not_profitable = float(probability[0])
        
        # Determine recommendation
        is_profitable = bool(prediction == 1)
        
        if is_profitable:
            if confidence > 0.8:
                recommendation = "Highly Recommended"
                status = "success"
            elif confidence > 0.6:
                recommendation = "Recommended"
                status = "success"
            else:
                recommendation = "Consider Carefully"
                status = "warning"
        else:
            if confidence > 0.8:
                recommendation = "Not Recommended"
                status = "danger"
            elif confidence > 0.6:
                recommendation = "Risky - Proceed with Caution"
                status = "warning"
            else:
                recommendation = "Uncertain - More Analysis Needed"
                status = "info"
        
        # Calculate financial projections
        expected_revenue = estimated_revenue * 0.97  # 97% efficiency
        expected_costs = estimated_revenue * 0.70    # Assume 70% costs
        expected_profit = expected_revenue - expected_costs
        
        # Determine risk level
        if confidence > 0.75:
            risk_level = 'Low'
        elif confidence > 0.55:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Prepare response
        result = {
            'success': True,
            'prediction': {
                'is_profitable': is_profitable,
                'confidence': round(confidence * 100, 2),
                'probability_profitable': round(prob_profitable * 100, 2),
                'probability_not_profitable': round(prob_not_profitable * 100, 2),
                'recommendation': recommendation,
                'status': status
            },
            'analysis': {
                'estimated_revenue': estimated_revenue,
                'expected_actual_revenue': round(expected_revenue, 2),
                'expected_profit': round(expected_profit, 2),
                'revenue_per_team_member': round(revenue_per_person, 2),
                'risk_level': risk_level
            },
            'factors': {
                'deadline': f"{deadline} days",
                'team_size': team_size,
                'complexity': complexity,
                'project_type': project_type
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✓ Prediction made: {recommendation} (confidence: {confidence*100:.1f}%)")
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid data type: {str(e)}'
        }), 400
        
    except Exception as e:
        print(f"\n❌ Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 Starting Flask API Server...")
    print("=" * 70)
    print(f"\n📡 API Endpoints:")
    print(f"  • GET  http://localhost:5000/")
    print(f"  • GET  http://localhost:5000/api/health")
    print(f"  • GET  http://localhost:5000/api/model-info")
    print(f"  • POST http://localhost:5000/api/predict")
    print("\n" + "=" * 70)
    print("\n💡 Press CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
