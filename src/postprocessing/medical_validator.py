import re
from datetime import datetime

class MedicalValidator:
    def __init__(self):
        """Initialize the medical prescription validator"""
        # Common dosage patterns
        self.dosage_patterns = [
            r"\d+\s*mg",
            r"\d+\s*g",
            r"\d+\s*mcg",
            r"\d+\s*ml",
            r"\d+\s*tablet(s)?",
            r"\d+\s*pill(s)?",
            r"\d+\s*capsule(s)?",
            r"\d+\s*drop(s)?",
            r"\d+\s*application(s)?",
            r"\d+\s*puff(s)?",
            r"\d+\s*patch(es)?"
        ]
        
        # Common frequency patterns
        self.frequency_patterns = [
            r"once daily",
            r"twice daily",
            r"three times daily",
            r"four times daily",
            r"every \d+ hours",
            r"every morning",
            r"every night",
            r"at bedtime",
            r"as needed",
            r"with meals",
            r"\d+ times (a|per) day",
            r"daily",
            r"weekly",
            r"monthly"
        ]
        
    def validate_prescription(self, data):
        """
        Validate extracted prescription data and add confidence scores
        
        Args:
            data: Extracted prescription data
            
        Returns:
            Data with validation flags and confidence scores
        """
        results = data.copy()
        results['validation'] = {
            'is_valid': True,
            'warnings': [],
            'confidence_scores': {}
        }
        
        # Check patient data
        if not data.get('patient_name'):
            results['validation']['warnings'].append("Missing patient name")
            results['validation']['is_valid'] = False
        
        # Validate patient age if present
        if 'patient_age' in data:
            try:
                age = int(str(data['patient_age']).split()[0])
                if age <= 0 or age > 120:
                    results['validation']['warnings'].append(f"Unusual patient age: {age}")
            except:
                if data['patient_age'] is not None:
                    results['validation']['warnings'].append(f"Invalid patient age format: {data['patient_age']}")
        
        # Validate medications
        if 'medication_list' in data and isinstance(data['medication_list'], list):
            for i, med in enumerate(data['medication_list']):
                med_confidence = 1.0
                
                # Check medication name
                if not med.get('name'):
                    results['validation']['warnings'].append(f"Medication #{i+1} is missing a name")
                    med_confidence *= 0.5
                
                # Check dosage format
                if 'dosage' in med and med['dosage']:
                    dosage_valid = any(re.search(pattern, str(med['dosage']), re.IGNORECASE) 
                                     for pattern in self.dosage_patterns)
                    if not dosage_valid:
                        results['validation']['warnings'].append(
                            f"Medication '{med.get('name', f'#{i+1}')}' has unusual dosage format: {med['dosage']}"
                        )
                        med_confidence *= 0.8
                
                # Store confidence score for this medication
                results['validation']['confidence_scores'][f"medication_{i+1}"] = round(med_confidence, 2)
        else:
            results['validation']['warnings'].append("No medications found in prescription")
            results['validation']['is_valid'] = False
        
        # Overall confidence based on warnings
        if results['validation']['warnings']:
            warnings_count = len(results['validation']['warnings'])
            results['validation']['overall_confidence'] = max(0.1, round(1.0 - (warnings_count * 0.1), 2))
        else:
            results['validation']['overall_confidence'] = 1.0
            
        return results