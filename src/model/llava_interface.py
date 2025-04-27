# model/llava_interface.py

import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoProcessor, LlavaForConditionalGeneration

class LlavaExtractor:
    def __init__(self, model_name="llava-hf/llava-1.5-13b-hf"):
        """
        Initialize LLaVA model for prescription extraction
        
        Args:
            model_name: HuggingFace model name for LLaVA
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Set max length for generation
        self.max_length = 1024
        
    def load_image(self, image_path_or_url):
        """Load image from path or URL"""
        if image_path_or_url.startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path_or_url)
        
        return image
    
    def extract_prescription_data(self, image_path_or_url, prompt_template):
        """
        Extract structured data from prescription image
        
        Args:
            image_path_or_url: Path or URL to prescription image
            prompt_template: Instruction prompt for extraction
            
        Returns:
            Extracted text from the model
        """
        # Load and prepare image
        image = self.load_image(image_path_or_url)
        
        # Process inputs
        inputs = self.processor(
            prompt_template,
            image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=self.max_length,
                do_sample=False
            )
        
        # Decode and return generated text
        response = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract only the model's response (remove the prompt)
        response = response.split(prompt_template)[-1].strip()
        
        return response