"""
Test the Flask API with sample requests
"""

import requests
import json

API_URL = "http://localhost:5000"

print("=" * 70)
print(" " * 20 + "Testing Flask API")
print("=" * 70)

# Test 1: Health Check
print("\n[TEST 1] Health Check...")
try:
    response = requests.get(f"{API_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure the Flask API is running!")
    exit(1)

# Test 2: Model Info
print("\n[TEST 2] Getting Model Info...")
response = requests.get(f"{API_URL}/api/model-info")
data = response.json()
print(f"✓ Model Accuracy: {data['accuracy']*100:.2f}%")
print(f"✓ Number of Features: {len(data['features'])}")

# Test 3: Prediction - High Value Project
print("\n[TEST 3] Predicting High-Value E-commerce Project...")
test_data = {
    "projectType": "E-commerce Platform",
    "deadline": 5,
    "estimatedRevenue": 150000,
    "teamSize": 5,
    "complexity": "High",
    "clientSatisfaction": 4.5
}

response = requests.post(
    f"{API_URL}/api/predict",
    json=test_data,
    headers={'Content-Type': 'application/json'}
)

result = response.json()
if result['success']:
    pred = result['prediction']
    analysis = result['analysis']
    
    print(f"\n{'='*70}")
    print(f"Project: High-Value E-commerce")
    print(f"{'='*70}")
    print(f"✓ Prediction: {'PROFITABLE ✅' if pred['is_profitable'] else 'NOT PROFITABLE ❌'}")
    print(f"✓ Confidence: {pred['confidence']}%")
    print(f"✓ Recommendation: {pred['recommendation']}")
    print(f"✓ Expected Profit: ₹{analysis['expected_profit']:,.0f}")
    print(f"✓ Risk Level: {analysis['risk_level']}")
else:
    print(f"❌ Error: {result['error']}")

# Test 4: Prediction - Risky Project
print("\n[TEST 4] Predicting Risky Mobile App Project...")
test_data = {
    "projectType": "Mobile App",
    "deadline": 1,
    "estimatedRevenue": 30000,
    "teamSize": 2,
    "complexity": "High",
    "clientSatisfaction": 3.5
}

response = requests.post(
    f"{API_URL}/api/predict",
    json=test_data,
    headers={'Content-Type': 'application/json'}
)

result = response.json()
if result['success']:
    pred = result['prediction']
    analysis = result['analysis']
    
    print(f"\n{'='*70}")
    print(f"Project: Risky Mobile App")
    print(f"{'='*70}")
    print(f"✓ Prediction: {'PROFITABLE ✅' if pred['is_profitable'] else 'NOT PROFITABLE ❌'}")
    print(f"✓ Confidence: {pred['confidence']}%")
    print(f"✓ Recommendation: {pred['recommendation']}")
    print(f"✓ Expected Profit: ₹{analysis['expected_profit']:,.0f}")
    print(f"✓ Risk Level: {analysis['risk_level']}")
else:
    print(f"❌ Error: {result['error']}")

print("\n" + "=" * 70)
print(" " * 25 + "✅ API TESTS COMPLETE!")
print("=" * 70)
