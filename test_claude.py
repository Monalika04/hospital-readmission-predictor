from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_summary(patient_data, risk_score, top_factors):
    prompt = f"""You are a clinical decision support assistant.
A machine learning model has assessed a diabetic patient
for 30-day hospital readmission risk.

Patient details:
- Age: {patient_data['age']}
- Days in hospital: {patient_data['time_in_hospital']}
- Number of medications: {patient_data['num_medications']}
- Total prior visits: {patient_data['total_visits']}
- Emergency rate: {patient_data['emergency_rate']}
- Senior patient (65+): {'Yes' if patient_data['is_senior'] else 'No'}

Risk score: {risk_score:.1%} probability of readmission within 30 days

Top risk factors identified by SHAP analysis:
{top_factors}

Write a concise 3-sentence clinical summary for the
attending physician. Be specific about the risk factors.
End with one actionable recommendation."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Free model
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )

    return response.choices[0].message.content


# Test data
test_patient = {
    'age': 72,
    'time_in_hospital': 8,
    'num_medications': 18,
    'total_visits': 3,
    'emergency_rate': 0.67,
    'is_senior': 1
}

top_factors = """1. discharge_disposition_id: +0.29 (discharged to nursing facility)
2. time_in_hospital: +0.14 (long stay of 8 days)
3. total_visits: -0.11 (limited prior visit history)"""

summary = generate_summary(test_patient, 0.78, top_factors)

print("=== GROQ SUMMARY ===")
print(summary)
print("=== API WORKING ===")