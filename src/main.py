# Create main.py
import os
import argparse
import json
import cv2
from tqdm import tqdm

# Import project modules
from src.preprocessing.image_enhancement import enhance_prescription
from src.model.llava_interface import LlavaExtractor
from src.model.prompt_templates import get_extraction_prompt, get_verification_prompt
from src.postprocessing.json_formatter import JsonFormatter
from src.postprocessing.medical_validator import MedicalValidator
from src.evaluation.metrics import PrescriptionEvaluator

def process_prescription(image_path, llava_model, formatter, validator, output_dir=None):
    """
    Process a single prescription image through the entire pipeline
    
    Args:
        image_path: Path to prescription image
        llava_model: Initialized LlavaExtractor instance
        formatter: Initialized JsonFormatter instance
        validator: Initialized MedicalValidator instance
        output_dir: Directory to save intermediate results (optional)
        
    Returns:
        Extracted and validated prescription data
    """
    print(f"Processing {image_path}...")
    
    # Step 1: Enhance image
    enhanced_img = enhance_prescription(image_path)
    
    # Save enhanced image if output directory provided
    if output_dir:
        base_name = os.path.basename(image_path).split('.')[0]
        enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.jpg")
        cv2.imwrite(enhanced_path, enhanced_img)
    
    # Step 2: Extract text with LLaVA
    prompt = get_extraction_prompt()
    raw_response = llava_model.extract_prescription_data(image_path, prompt)
    
    # Step 3: Format response to JSON
    extracted_data = formatter.format_response(raw_response)
    
    # Step 4: Verify extracted data with a second pass
    verification_prompt = get_verification_prompt(json.dumps(extracted_data, indent=2))
    verification_response = llava_model.extract_prescription_data(image_path, verification_prompt)
    verified_data = formatter.format_response(verification_response)
    
    # If verification worked, use the verified data
    if "error" not in verified_data:
        final_data = verified_data
    else:
        final_data = extracted_data
    
    # Step 5: Standardize medical terms
    standardized_data = formatter.standardize_medical_terms(final_data)
    
    # Step 6: Validate data
    validated_data = validator.validate_prescription(standardized_data)
    
    # Save results if output directory provided
    if output_dir:
        results_path = os.path.join(output_dir, f"{base_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(validated_data, f, indent=2)
    
    return validated_data

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Medical Prescription Extraction Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing prescription images")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--gt_file", type=str, help="Path to ground truth JSON file (optional)")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-13b-hf", help="LLaVA model name")
    parser.add_argument("--medical_terms", type=str, help="Path to medical terminology JSON file (optional)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    llava_model = LlavaExtractor(model_name=args.model_name)
    formatter = JsonFormatter(medical_terms_path=args.medical_terms)
    validator = MedicalValidator()
    
    # Get all image files
    image_files = [
        os.path.join(args.input_dir, f) 
        for f in os.listdir(args.input_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    # Process all images
    results = []
    for image_path in tqdm(image_files, desc="Processing prescriptions"):
        result = process_prescription(image_path, llava_model, formatter, validator, args.output_dir)
        results.append(result)
    
    # Save all results
    all_results_path = os.path.join(args.output_dir, "all_results.json")
    with open(all_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Evaluate if ground truth provided
    if args.gt_file:
        evaluator = PrescriptionEvaluator()
        
        # Load ground truth
        with open(args.gt_file, 'r') as f:
            ground_truth = json.load(f)
        
        # Evaluate
        metrics = evaluator.evaluate_dataset(results, ground_truth)
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Overall evaluation score: {metrics['overall_score']:.2f}")

if __name__ == "__main__":
    main()
