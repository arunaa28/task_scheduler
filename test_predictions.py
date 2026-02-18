import joblib
import pandas as pd
import numpy as np
import os

def rupee(x):
    return f"₹{int(x):,}"

print("="*70)
print(" "*22 + "Testing ML Model")
print("="*70)

print("\n[1/3] Loading trained model...")
model_path = os.path.join('models', 'project_predictor.pkl')
if not os.path.exists(model_path):
    print(f"❌ ERROR: Model not found at: {model_path}")
    print("   Run train_model.py first to create the model.")
    raise SystemExit(1)

package = joblib.load(model_path)
model = package['model']
label_encoders = package.get('label_encoders', {})
feature_columns = package.get('feature_columns')
accuracy = package.get('accuracy', None)

print("✓ Model loaded successfully")
if accuracy is not None:
    print(f"✓ Model Accuracy: {accuracy*100:.2f}%")

print("\n[2/3] Testing with 4 sample projects...")
print("-"*70)

# Define sample projects
samples = [
    {
        'project_type': 'E-commerce Platform',
        'deadline_days': 5,
        'estimated_revenue': 150000,
        'team_size': 5,
        'complexity': 'High',
        'client_satisfaction': 4.5
    },
    {
        'project_type': 'Landing Page',
        'deadline_days': 1,
        'estimated_revenue': 25000,
        'team_size': 2,
        'complexity': 'Low',
        'client_satisfaction': 4.9
    },
    {
        'project_type': 'API Integration',
        'deadline_days': 2,
        'estimated_revenue': 45000,
        'team_size': 2,
        'complexity': 'Low',
        'client_satisfaction': 4.2
    },
    {
        'project_type': 'Database Migration',
        'deadline_days': 4,
        'estimated_revenue': 75000,
        'team_size': 3,
        'complexity': 'Medium',
        'client_satisfaction': 3.8
    }
]

def encode_value(col, val):
    le = label_encoders.get(col)
    if le is None:
        return 0
    try:
        return int(le.transform([val])[0])
    except Exception:
        return 0

for i, s in enumerate(samples, 1):
    print("\n" + "="*70)
    print(f"TEST PROJECT #{i}: {s['project_type']}")
    print("="*70)
    print("\n📋 Project Details:")
    print(f"  • Type:               {s['project_type']}")
    print(f"  • Deadline:           {s['deadline_days']} days")
    print(f"  • Estimated Revenue:  {rupee(s['estimated_revenue'])}")
    print(f"  • Team Size:          {s['team_size']} people")
    print(f"  • Complexity:         {s['complexity']}")
    print(f"  • Client Rating:      {s['client_satisfaction']}/5.0")
    print(f"  • Revenue/Person:     {rupee(s['estimated_revenue']/s['team_size'])}")

    # Build feature row
    row = {
        'deadline_days': s['deadline_days'],
        'estimated_revenue': s['estimated_revenue'],
        'team_size': s['team_size'],
        'project_type_encoded': encode_value('project_type', s['project_type']),
        'complexity_encoded': encode_value('complexity', s['complexity']),
        'client_satisfaction': s['client_satisfaction'],
        'revenue_per_person': s['estimated_revenue']/s['team_size'],
        'deadline_revenue_ratio': s['estimated_revenue']/(s['deadline_days']*10000)
    }

    X = pd.DataFrame([row])[feature_columns]
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    conf = max(proba)

    label = 'PROFITABLE' if pred == 1 else 'NOT PROFITABLE'
    print("\n🤖 ML Prediction:")
    print(f"  {'✅' if pred==1 else '❌'} {label}")
    print(f"  📊 Confidence: {conf*100:.1f}%")
    print(f"  📈 Probability: {proba[1]*100:.1f}% profitable, {proba[0]*100:.1f}% not profitable")

    # Financial projection (simple)
    expected_revenue = s['estimated_revenue'] * 0.97
    expected_profit = expected_revenue * (0.28 if pred==1 else 0.05)
    print("\n  💰 Financial Projection:")
    print(f"    • Expected Revenue:   {rupee(expected_revenue)}")
    print(f"    • Expected Profit:    {rupee(expected_profit)}")

print("\nDone.")
