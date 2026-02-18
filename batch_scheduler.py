"""
Batch Project Scheduler - Complete & Corrected
Load multiple random projects from CSV and generate schedules for all
"""

import pandas as pd
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import sys

def load_ml_model():
    """Load trained ML model for profitability prediction"""
    try:
        model_data = joblib.load('models/project_predictor.pkl')
        print("✓ ML Model loaded successfully")
        return model_data
    except FileNotFoundError:
        print("⚠️  Warning: ML model not found. Profitability predictions disabled.")
        return None
    except Exception as e:
        print(f"⚠️  Warning: Could not load ML model: {e}")
        return None

def get_schedule_phases(deadline_days, project_type, complexity):
    """Generate project phases based on deadline and complexity"""
    phases = []
    
    complexity_breakdown = {
        'Low': {'Planning': 10, 'Design': 15, 'Dev': 50, 'Testing': 15, 'Deployment': 10},
        'Medium': {'Planning': 15, 'Design': 20, 'Dev': 40, 'Testing': 15, 'Deployment': 10},
        'High': {'Planning': 20, 'Design': 25, 'Dev': 35, 'Testing': 15, 'Deployment': 5}
    }
    
    breakdown = complexity_breakdown.get(complexity, complexity_breakdown['Medium'])
    
    start_date = datetime.now()
    cumulative_days = 0
    
    # Maintain consistent phase order
    phase_list = ['Planning', 'Design', 'Dev', 'Testing', 'Deployment']
    
    for phase_name in phase_list:
        percentage = breakdown[phase_name]
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

def predict_profitability(model_data, project):
    """Use ML model to predict profitability"""
    if not model_data:
        return None
    
    try:
        model = model_data['model']
        label_encoders = model_data['label_encoders']
        feature_columns = model_data['feature_columns']
        
        # Encode categorical features with safe fallback
        try:
            project_type_encoded = label_encoders['project_type'].transform([project['project_type']])[0]
        except (KeyError, ValueError):
            project_type_encoded = 0
        
        try:
            complexity_encoded = label_encoders['complexity'].transform([project['complexity']])[0]
        except (KeyError, ValueError):
            complexity_encoded = 1
        
        # Calculate derived features with proper type conversion
        revenue_per_person = float(project['estimated_revenue']) / float(project['team_size'])
        deadline_revenue_ratio = float(project['estimated_revenue']) / (float(project['deadline_days']) * 10000)
        
        # Create feature dataframe with correct column order
        features = pd.DataFrame({
            'deadline_days': [float(project['deadline_days'])],
            'estimated_revenue': [float(project['estimated_revenue'])],
            'team_size': [float(project['team_size'])],
            'project_type_encoded': [float(project_type_encoded)],
            'complexity_encoded': [float(complexity_encoded)],
            'client_satisfaction': [float(project['client_satisfaction'])],
            'revenue_per_person': [revenue_per_person],
            'deadline_revenue_ratio': [deadline_revenue_ratio]
        })[feature_columns]
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        confidence = float(max(probability))
        is_profitable = bool(prediction == 1)
        
        # Determine recommendation based on profitability and confidence
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
        
        # Determine risk level
        if confidence > 0.75:
            risk_level = 'Low'
        elif confidence > 0.55:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Calculate financial projections
        expected_revenue = float(project['estimated_revenue']) * 0.97
        expected_costs = float(project['estimated_revenue']) * 0.70
        expected_profit = expected_revenue - expected_costs
        
        return {
            'is_profitable': is_profitable,
            'confidence': round(confidence * 100, 2),
            'recommendation': recommendation,
            'risk_level': risk_level,
            'expected_profit': round(expected_profit, 2)
        }
    
    except Exception as e:
        print(f"⚠️  Prediction error: {e}")
        return None

def format_schedule(project_data, phases, profitability=None):
    """Format detailed schedule output"""
    output = []
    output.append("\n" + "=" * 85)
    output.append(f"{'PROJECT SCHEDULE':^85}")
    output.append("=" * 85)
    
    output.append(f"\n📋 PROJECT DETAILS:")
    output.append(f"  • Client/Name:          {project_data.get('client_name', 'N/A')}")
    output.append(f"  • Project Type:         {project_data.get('project_type', 'N/A')}")
    output.append(f"  • Complexity Level:     {project_data.get('complexity', 'N/A')}")
    output.append(f"  • Team Size:            {project_data.get('team_size', 'N/A')} members")
    output.append(f"  • Estimated Revenue:    ₹{int(project_data.get('estimated_revenue', 0)):,}")
    output.append(f"  • Total Deadline:       {project_data.get('deadline_days', 'N/A')} days")
    output.append(f"  • Client Satisfaction:  {project_data.get('client_satisfaction', 'N/A')}/5.0")
    
    if profitability:
        status_icon = "✅" if profitability['is_profitable'] else "❌"
        output.append(f"\n🤖 ML PREDICTION:")
        output.append(f"  • Status:               {status_icon} {'PROFITABLE' if profitability['is_profitable'] else 'NOT PROFITABLE'}")
        output.append(f"  • Confidence:           {profitability['confidence']}%")
        output.append(f"  • Recommendation:       {profitability['recommendation']}")
        output.append(f"  • Expected Profit:      ₹{int(profitability['expected_profit']):,}")
        output.append(f"  • Risk Level:           {profitability['risk_level']}")
    
    output.append(f"\n📅 PROJECT TIMELINE:")
    output.append("-" * 85)
    
    for i, phase in enumerate(phases, 1):
        output.append(f"\n  Phase {i}: {phase['phase'].upper()}")
        output.append(f"    Duration:      {phase['duration_days']} days ({phase['percentage']}% of project)")
        output.append(f"    Start Date:    {phase['start_date']}")
        output.append(f"    End Date:      {phase['end_date']}")
        bar_fill = phase['duration_days']
        bar_empty = max(0, 15 - bar_fill)
        output.append(f"    Progress:      [{'█' * min(bar_fill, 15)}{'░' * bar_empty}]")
    
    output.append("\n" + "=" * 85)
    
    return "\n".join(output)

def main():
    """Main execution function"""
    print("\n" + "=" * 85)
    print(" " * 25 + "ProManage Batch Project Scheduler")
    print("=" * 85)
    
    # Load CSV data
    csv_path = Path('data/historical_projects.csv')
    if not csv_path.exists():
        print(f"\n❌ ERROR: CSV file not found at {csv_path}")
        print("Please ensure historical_projects.csv exists in the data/ folder")
        sys.exit(1)
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"\n✓ Loaded {len(df)} projects from CSV file")
        
        # Load ML model
        print("\n📦 Loading ML Model...")
        model_data = load_ml_model()
        
        # Select 5 random projects
        print("\n🎲 Selecting 5 random projects for scheduling...")
        if len(df) < 5:
            print(f"⚠️  Warning: CSV has only {len(df)} projects, using all of them")
            random_projects = df
        else:
            random_projects = df.sample(n=5, random_state=None)
        
        print("\n" + "-" * 85)
        print("SELECTED PROJECTS:")
        print("-" * 85)
        for idx, (_, row) in enumerate(random_projects.iterrows(), 1):
            print(f"{idx}. {row['client_name']:<20s} | {row['project_type']:<25s} | "
                  f"{int(row['deadline_days'])} days | ₹{int(row['estimated_revenue']):>10,d}")
        print("-" * 85)
        
        # Create schedules directory
        schedules_dir = Path('schedules')
        schedules_dir.mkdir(exist_ok=True)
        
        # Generate schedules for each project
        print("\n📝 Generating schedules...")
        all_schedules = []
        projects_summary = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for idx, (_, row) in enumerate(random_projects.iterrows(), 1):
            project = {
                'client_name': str(row['client_name']),
                'project_type': str(row['project_type']),
                'complexity': str(row['complexity']),
                'deadline_days': int(row['deadline_days']),
                'estimated_revenue': float(row['estimated_revenue']),
                'team_size': int(row['team_size']),
                'client_satisfaction': float(row['client_satisfaction'])
            }
            
            # Generate schedule phases
            phases = get_schedule_phases(
                project['deadline_days'],
                project['project_type'],
                project['complexity']
            )
            
            # Get profitability prediction
            profitability = predict_profitability(model_data, project)
            
            # Format the schedule
            schedule_text = format_schedule(project, phases, profitability)
            all_schedules.append(schedule_text)
            projects_summary.append({
                'client': project['client_name'],
                'type': project['project_type'],
                'days': project['deadline_days'],
                'revenue': project['estimated_revenue'],
                'profitability': profitability
            })
            
            print(f"  ✓ Schedule {idx}/5: {row['client_name']}")
        
        # Save combined schedules to file
        combined_filename = f"batch_schedule_{timestamp}.txt"
        combined_path = schedules_dir / combined_filename
        
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write("=" * 85 + "\n")
            f.write(f"{'BATCH SCHEDULE REPORT':^85}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Projects: {len(projects_summary)}\n")
            f.write("=" * 85 + "\n")
            f.write("\n".join(all_schedules))
            f.write("\n\n" + "=" * 85 + "\n")
            f.write(f"{'END OF BATCH REPORT':^85}\n")
            f.write("=" * 85 + "\n")
        
        print(f"\n✓ All schedules saved to: {combined_path}")
        
        # Display summary table
        print("\n" + "=" * 85)
        print("SCHEDULES SUMMARY")
        print("=" * 85)
        print(f"\n{'Client':<18} | {'Project Type':<25} | {'Days':<4} | {'Revenue':<11} | "
              f"{'Profit':<11} | {'Status':<20}")
        print("-" * 85)
        
        for proj in projects_summary:
            if proj['profitability']:
                status = "✅ " + proj['profitability']['recommendation'][:16]
                profit = f"₹{int(proj['profitability']['expected_profit']):,}"
            else:
                status = "⚠️  No prediction"
                profit = "N/A"
            
            print(f"{proj['client']:<18} | {proj['type']:<25} | {proj['days']:<4} | "
                  f"₹{int(proj['revenue']):<10,} | {profit:<11} | {status:<20}")
        
        print("-" * 85)
        print(f"\n✅ Batch scheduling complete!")
        print(f"📁 Output file: {combined_path}")
        print("=" * 85 + "\n")
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
