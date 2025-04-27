# model/prompt_templates.py

def get_extraction_prompt():
    """
    Returns a prompt template for extracting prescription information
    """
    prompt = """
    Extract all information from this handwritten medical prescription and format it as a JSON object with the following fields:
    
    - patient_name: Full name of the patient
    - patient_age: Age of the patient (number)
    - patient_gender: Gender of the patient
    - medication_list: A list of prescribed medications where each medication contains:
      - name: Name of the medication
      - dosage: The amount to be taken (e.g., "10mg", "1 tablet")
      - route: How the medication should be taken (e.g., "oral", "topical")
      - frequency: How often to take it (e.g., "twice daily", "every 8 hours")
      - duration: How long to take it (e.g., "7 days", "2 weeks")
      - special_instructions: Any additional notes on how to take the medication
    - diagnosis: The medical condition being treated
    - doctor_name: Name of the prescribing doctor
    - doctor_credentials: Qualifications or specialization of the doctor
    - date: Date when the prescription was written
    - hospital/clinic: Name of the hospital or clinic
    
    Provide your response in valid JSON format only. If you cannot read or determine any field with certainty, use null for that field.
    """
    
    return prompt.strip()

def get_verification_prompt(extracted_text):
    """
    Creates a verification prompt with the extracted text to double-check accuracy
    
    Args:
        extracted_text: The text extracted in the first pass
    
    Returns:
        Verification prompt string
    """
    prompt = f"""
    I've extracted the following information from a medical prescription:
    
    {extracted_text}
    
    Please verify this information and correct any errors you can identify. Pay special attention to:
    1. Medical terminology and drug names
    2. Dosage amounts and units
    3. Frequency instructions
    4. Missing critical information
    
    If any information is clearly wrong or implausible for a medical prescription, please fix it.
    Return the corrected information in the same JSON format.
    """
    
    return prompt.strip()

def get_segmented_extraction_prompt(region_description):
    """
    Returns a prompt for extracting information from a specific segment of the prescription
    
    Args:
        region_description: Description of what the region likely contains
        
    Returns:
        Segment-specific extraction prompt
    """
    prompt = f"""
    This image shows a portion of a medical prescription that likely contains {region_description}.
    Extract all legible text from this segment and format it appropriately.
    """
    
    return prompt.strip()