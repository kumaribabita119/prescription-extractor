# postprocessing/json_formatter.py

import json
import re
from difflib import get_close_matches

class JsonFormatter:
    def __init__(self, medical_terms_path=None):
        """
        Initialize formatter with optional medical terminology database
        
        Args:
            medical_terms_path: Path to JSON file with medical terminology
        """
        self.medical_terms = {}
        if medical_terms_path:
            try:
                with open(medical_terms_path, 'r') as f:
                    self.medical_terms = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load medical terms: {e}")
    
    def format_response(self, text):
        """
        Format and clean the LLM response into proper JSON
        
        Args:
            text: Raw text from the LLM
            
        Returns:
            Cleaned and formatted JSON object
        """
        # Try to extract JSON from the text
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find anything that looks like JSON
            json_str = text
            
            # Remove any non-JSON text before the first { and after the last }
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}')
            
            if start_idx >= 0 and end_idx >= 0:
                json_str = json_str[start_idx:end_idx+1]
            else:
                return {"error": "Could not extract valid JSON from response"}
        
        # Clean up common formatting issues
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
        
        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            # If JSON parsing fails, attempt basic repairs
            try:
                # Fix missing quotes around keys
                fixed_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
                # Fix single quotes
                fixed_str = fixed_str.replace("'", '"')
                return json.loads(fixed_str)
            except:
                return {"error": f"Invalid JSON format: {str(e)}", "raw_text": text}

    def standardize_medical_terms(self, data):
        """
        Corrects common medication names and medical terms
        
        Args:
            data: Extracted prescription data
            
        Returns:
            Data with standardized terminology
        """
        if not self.medical_terms:
            return data
            
        # Check and correct medication names
        if 'medication_list' in data and isinstance(data['medication_list'], list):
            for i, med in enumerate(data['medication_list']):
                if 'name' in med:
                    name = med['name']
                    # Check for close matches in our database
                    drug_names = self.medical_terms.get('drug_names', [])
                    matches = get_close_matches(name.lower(), [d.lower() for d in drug_names], n=1, cutoff=0.8)
                    if matches:
                        # Find the original case-preserved version
                        idx = [d.lower() for d in drug_names].index(matches[0])
                        data['medication_list'][i]['name'] = drug_names[idx]
                        # Add confidence score
                        data['medication_list'][i]['name_confidence'] = round(1 - (1 - 0.8) * 
                                               (1 - len(matches[0])/max(len(name), 1)), 2)
        
        return data


# postprocessing/medical_validator.py

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
        
        # Validate date if present
        if 'date' in data and data['date']:
            try:
                # Try various date formats
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%B %d, %Y', '%d %B %Y']:
                    try:
                        date_obj = datetime.strptime(data['date'], fmt)
                        # Check if date is in the reasonable past (not more than 1 year ago)
                        if (datetime.now() - date_obj).days > 365:
                            results['validation']['warnings'].append(f"Prescription date is more than a year old: {data['date']}")
                        break
                    except ValueError:
                        continue
            except:
                results['validation']['warnings'].append(f"Could not validate date format: {data['date']}")
        
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
                
                # Check frequency format
                if 'frequency' in med and med['frequency']:
                    frequency_valid = any(re.search(pattern, str(med['frequency']), re.IGNORECASE) 
                                        for pattern in self.frequency_patterns)
                    if not frequency_valid:
                        results['validation']['warnings'].append(
                            f"Medication '{med.get('name', f'#{i+1}')}' has unusual frequency format: {med['frequency']}"
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