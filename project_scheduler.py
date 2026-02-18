"""
Project Scheduler
Input project details and generate schedules with deadline planning
"""

import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path

def load_ml_model():
    """Load trained ML model for profitability prediction"""
    import joblib
    try:
        model_data = joblib.load('models/project_predictor.pkl')
        return model_data
    except:
        print("⚠️  Warning: ML model not found. Profitability predictions disabled.")
        return None

def get_schedule_phases(deadline_days, project_type, complexity):
    """Generate project phases based on deadline and complexity"""
    phases = []
    
    # Define phase percentages by complexity
    complexity_breakdown = {
        'Low': {'Planning': 10, 'Design': 15, 'Dev': 50, 'Testing': 15, 'Deployment': 10},
        'Medium': {'Planning': 15, 'Design': 20, 'Dev': 40, 'Testing': 15, 'Deployment': 10},
        'High': {'Planning': 20, 'Design': 25, 'Dev': 35, 'Testing': 15, 'Deployment': 5}
    }
    
    breakdown = complexity_breakdown.get(complexity, complexity_breakdown['Medium'])
    
    start_date = datetime.now()
    cumulative_days = 0
    
    for phase_name, percentage in breakdown.items():
        phase_days = max(1, int(deadline_days * percentage / 100))
        phase_start = start_date + timedelta(days=cumulative_days)
        phase_end = phase_start + timedelta(days=phase_days - 1)
        
        phases.append({
            'phase': phase_name,
            'percentage': percentage,
            'duration_days': phase_days,
            'start_date': phase_start.strftime('%Y-%m-%d'),
            'end_date': phase_end.strftime('%Y-%m-%d')
        })
        
        cumulative_days += phase_days
    
    return phases

def format_schedule(project_data, phases, profitability=None):
    """Format schedule output"""
    output = []
    output.append("\n" + "=" * 80)
    output.append(f"{'PROJECT SCHEDULE':^80}")
    output.append("=" * 80)
    
    output.append(f"\n📋 PROJECT DETAILS:")
    output.append(f"  • Client/Name:        {project_data.get('client_name', 'N/A')}")
    output.append(f"  • Project Type:       {project_data.get('project_type', 'N/A')}")
    output.append(f"  • Complexity:         {project_data.get('complexity', 'N/A')}")
    output.append(f"  • Team Size:          {project_data.get('team_size', 'N/A')} members")
    output.append(f"  • Estimated Revenue:  ₹{int(project_data.get('estimated_revenue', 0)):,}")
    output.append(f"  • Deadline:           {project_data.get('deadline_days', 'N/A')} days")
    output.append(f"  • Client Satisfaction: {project_data.get('client_satisfaction', 'N/A')}/5.0")
    
    if profitability:
        output.append(f"\n🤖 ML PREDICTION:")
        output.append(f"  • Status:             {'PROFITABLE ✅' if profitability['is_profitable'] else 'NOT PROFITABLE ❌'}")
        output.append(f"  • Confidence:         {profitability['confidence']}%")
        output.append(f"  • Recommendation:     {profitability['recommendation']}")
        output.append(f"  • Expected Profit:    ₹{int(profitability['expected_profit']):,}")
        output.append(f"  • Risk Level:         {profitability['risk_level']}")
    
    output.append(f"\n📅 PROJECT TIMELINE:")
    output.append("-" * 80)
    
    for i, phase in enumerate(phases, 1):
        output.append(f"\nPhase {i}: {phase['phase'].upper()}")
        output.append(f"  Duration:    {phase['duration_days']} days ({phase['percentage']}% of total)")
        output.append(f"  Start Date:  {phase['start_date']}")
        output.append(f"  End Date:    {phase['end_date']}")
        output.append(f"  Timeline:    {'█' * phase['duration_days']}  {'░' * (10 - min(phase['duration_days'], 10))}")
    
    output.append("\n" + "=" * 80)
    
    return "\n".join(output)

def save_schedule(schedule_text, filename=None):
    """Save schedule to file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"project_schedule_{timestamp}.txt"
    
    filepath = Path('schedules') / filename
    filepath.parent.mkdir(exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(schedule_text)
    
    return filepath

def input_project_data():
    """Interactively input project data"""
    print("\n" + "=" * 80)
    print(f"{'PROJECT SCHEDULER - INPUT NEW PROJECT':^80}")
    print("=" * 80)
    
    project = {}
    
    # Get basic info
    project['client_name'] = input("\n📌 Client/Project Name: ").strip()
    
    print("\n  Project Types: Website Development, Mobile App, API Integration,")
    print("                 Database Migration, E-commerce Platform, Landing Page,")
    print("                 CRM System, ERP Module, Analytics Dashboard, etc.")
    project['project_type'] = input("  Enter Project Type: ").strip()
    
    print("\n  Complexity Levels: Low, Medium, High")
    project['complexity'] = input("  Enter Complexity: ").strip().capitalize()
    
    # Validate complexity
    while project['complexity'] not in ['Low', 'Medium', 'High']:
        project['complexity'] = input("  Invalid! Enter Low, Medium, or High: ").strip().capitalize()
    
    project['deadline_days'] = int(input("\n⏱️  Deadline (days): "))
    project['estimated_revenue'] = float(input("💰 Estimated Revenue (₹): "))
    project['team_size'] = int(input("👥 Team Size (number of people): "))
    project['client_satisfaction'] = float(input("⭐ Client Satisfaction (1-5): "))
    
    return project

def predict_profitability(model_data, project):
    """Use ML model to predict profitability"""
    if not model_data:
        return None
    
    try:
        import pandas as pd
        
        model = model_data['model']
        label_encoders = model_data['label_encoders']
        feature_columns = model_data['feature_columns']
        
        # Encode categorical features
        try:
            project_type_encoded = label_encoders['project_type'].transform([project['project_type']])[0]
        except:
            project_type_encoded = 0
        
        try:
            complexity_encoded = label_encoders['complexity'].transform([project['complexity']])[0]
        except:
            complexity_encoded = 1
        
        # Calculate derived features
        revenue_per_person = project['estimated_revenue'] / project['team_size']
        deadline_revenue_ratio = project['estimated_revenue'] / (project['deadline_days'] * 10000)
        
        # Create feature dataframe
        features = pd.DataFrame({
            'deadline_days': [project['deadline_days']],
            'estimated_revenue': [project['estimated_revenue']],
            'team_size': [project['team_size']],
            'project_type_encoded': [project_type_encoded],
            'complexity_encoded': [complexity_encoded],
            'client_satisfaction': [project['client_satisfaction']],
            'revenue_per_person': [revenue_per_person],
            'deadline_revenue_ratio': [deadline_revenue_ratio]
        })
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        confidence = max(probability)
        is_profitable = bool(prediction == 1)
        
        # Calculate recommendation
        if is_profitable:
            if confidence > 0.8:
                recommendation = "Highly Recommended"
            elif confidence > 0.6:
                recommendation = "Recommended"
            else:
                recommendation = "Consider Carefully"
        else:
            if confidence > 0.8:
                recommendation = "Not Recommended"
            elif confidence > 0.6:
                recommendation = "Risky - Proceed with Caution"
            else:
                recommendation = "Uncertain - More Analysis Needed"
        
        # Risk level
        if confidence > 0.75:
            risk_level = 'Low'
        elif confidence > 0.55:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Financial projection
        expected_revenue = project['estimated_revenue'] * 0.97
        expected_costs = project['estimated_revenue'] * 0.70
        expected_profit = expected_revenue - expected_costs
        
        return {
            'is_profitable': is_profitable,
            'confidence': round(confidence * 100, 2),
            'recommendation': recommendation,
            'risk_level': risk_level,
            'expected_profit': expected_profit
        }
    
    except Exception as e:
        print(f"⚠️  Error in prediction: {e}")
        return None

def load_from_csv():
    """Load project data from CSV file"""
    print("\n" + "=" * 80)
    print(f"{'LOAD PROJECT FROM CSV':^80}")
    print("=" * 80)
    
    csv_path = Path('data/historical_projects.csv')
    if not csv_path.exists():
        print(f"❌ CSV file not found at {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"\n✓ Loaded {len(df)} projects from CSV")
        print(df[['project_id', 'client_name', 'project_type', 'deadline_days']].head(10))
        
        while True:
            try:
                project_id = int(input("\nEnter Project ID to schedule: "))
                project_row = df[df['project_id'] == project_id].iloc[0]
                
                project = {
                    'client_name': project_row['client_name'],
                    'project_type': project_row['project_type'],
                    'complexity': project_row['complexity'],
                    'deadline_days': int(project_row['deadline_days']),
                    'estimated_revenue': float(project_row['estimated_revenue']),
                    'team_size': int(project_row['team_size']),
                    'client_satisfaction': float(project_row['client_satisfaction'])
                }
                
                return project
            except (ValueError, IndexError):
                print("❌ Invalid Project ID. Try again.")
    
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def main():
    print("\n" + "=" * 80)
    print(" " * 20 + "ProManage Project Scheduler")
    print("=" * 80)
    
    # Load ML model
    model_data = load_ml_model()
    
    while True:
        print("\n" + "-" * 80)
        print("OPTIONS:")
        print("  1. Create new project schedule (manual input)")
        print("  2. Load project from CSV and create schedule")
        print("  3. Exit")
        print("-" * 80)
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            project = input_project_data()
        elif choice == '2':
            project = load_from_csv()
            if project is None:
                continue
        elif choice == '3':
            print("\n✓ Exiting scheduler. Goodbye!")
            break
        else:
            print("❌ Invalid choice. Try again.")
            continue
        
        if project:
            # Generate schedule
            phases = get_schedule_phases(
                project['deadline_days'],
                project['project_type'],
                project['complexity']
            )
            
            # Predict profitability
            profitability = predict_profitability(model_data, project)
            
            # Format and display
            schedule_text = format_schedule(project, phases, profitability)
            print(schedule_text)
            
            # Save option
            save_choice = input("\nSave schedule? (y/n): ").strip().lower()
            if save_choice == 'y':
                filepath = save_schedule(schedule_text)
                print(f"✓ Schedule saved to: {filepath}")

if __name__ == '__main__':
    main()
