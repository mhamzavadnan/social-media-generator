from typing import Dict, Any, List, Optional
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
import base64
from io import BytesIO
from PIL import Image
from .base_agent import BaseAgent
import asyncio

class VisualGenerationAgent(BaseAgent):
    """Agent responsible for generating visual content for social media posts."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the visual generation agent.
        
        Args:
            name (str): Name of the agent
            config (Dict[str, Any], optional): Configuration parameters including:
                - openai_api_key: OpenAI API key for DALL-E
                - image_size: Size of generated images (default: "1024x1024")
                - image_quality: Quality of generated images (default: "standard")
                - model: Model to use (default: "dall-e-3")
        """
        super().__init__(name, config)
        self._initialize_image_generator()
        
    def _initialize_image_generator(self):
        """Initialize the image generation model."""
        load_dotenv('config.env')
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.logger.error("No OpenAI API key provided. Image generation will not work.")
            print("⚠️ No OpenAI API key found. Please check your config.env file.")
            return
        
        try:
            self.client = OpenAI(api_key=api_key)
            # Test the API key with a simple models list call
            self.client.models.list()
            print("✅ OpenAI API key is valid and connected for image generation.")
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI client: {e}")
            print(f"❌ Error connecting to OpenAI: {e}")
            
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visual content based on text posts and brand guidelines.
        
        Args:
            input_data (Dict[str, Any]): Input data containing:
                - text_content (str): The text content to generate an image for
                - brand_guidelines (Dict): Brand visual guidelines
                - style_preferences (Dict): Specific style preferences
                - platform_specs (Dict): Platform-specific image requirements
                
        Returns:
            Dict[str, Any]: Generated image data and metadata
        """
        required_fields = ['text_content', 'brand_guidelines']
        if not self.validate_input(input_data, required_fields):
            raise ValueError("Invalid input data")
            
        text_content = input_data['text_content']
        brand_guidelines = input_data['brand_guidelines']
        style_preferences = input_data.get('style_preferences', {})
        platform_specs = input_data.get('platform_specs', {})
        
        try:
            # Generate image prompt based on inputs
            image_prompt = self._create_image_prompt(
                text_content,
                brand_guidelines,
                style_preferences
            )
            
            print(f"\nGenerating image for prompt: {image_prompt[:100]}...")
            
            # Generate the image
            image_data = await self._generate_image(image_prompt)
            
            if not image_data:
                print("❌ Image generation failed. Using placeholder image.")
                return {
                    'image_data': self._generate_mock_image(),
                    'metadata': {
                        'prompt_used': image_prompt,
                        'platform_specs': platform_specs,
                        'generation_successful': False,
                        'error': "Image generation failed - using placeholder"
                    }
                }
            
            # Process the image according to platform specifications
            processed_image = self._process_image(image_data, platform_specs)
            
            print("✅ Image generated successfully!")
            
            return {
                'image_data': processed_image,
                'metadata': {
                    'prompt_used': image_prompt,
                    'platform_specs': platform_specs,
                    'generation_successful': True
                }
            }
            
        except Exception as e:
            error_msg = f"Error in image generation: {str(e)}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            return {
                'image_data': self._generate_mock_image(),
                'metadata': {
                    'error': error_msg,
                    'generation_successful': False
                }
            }
            
    def _create_image_prompt(self, text_content: str,
                           brand_guidelines: Dict[str, Any],
                           style_preferences: Dict[str, Any]) -> str:
        """Create a detailed prompt for image generation."""
        brand_colors = brand_guidelines.get('colors', [])
        brand_style = brand_guidelines.get('style', 'modern')
        mood = style_preferences.get('mood', 'neutral')
        
        # Extract key concepts from text content
        key_concepts = ' '.join([word for word in text_content.split() 
                               if not word.startswith('#') and not word.startswith('@')])
        
        prompt = f"""Create a professional social media image that captures the essence of: "{key_concepts}"

        Style Requirements:
        - Brand style: {brand_style}
        - Color scheme: {', '.join(brand_colors) if brand_colors else 'brand appropriate'}
        - Mood: {mood}
        - Style: Professional, social media optimized
        
        Additional Requirements:
        - Clean, high-quality composition
        - Suitable for social media sharing
        - No text overlay (text will be added separately)
        - Modern and visually appealing design
        - Professional lighting and composition
        """
        
        return prompt.strip()
        
    async def _generate_image(self, prompt: str) -> Optional[bytes]:
        """Generate an image using DALL-E."""
        if not hasattr(self, 'client'):
            self.logger.warning("No API key available for image generation")
            return None
            
        try:
            print("🎨 Requesting image from DALL-E...")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.images.generate(
                    model=self.get_config('model', "dall-e-3"),
                    prompt=prompt,
                    n=1,
                    size=self.get_config('image_size', "1024x1024"),
                    quality=self.get_config('image_quality', "standard"),
                    response_format="b64_json"
                )
            )
            
            if not response.data:
                self.logger.error("No image data received from DALL-E")
                return None
                
            image_data = base64.b64decode(response.data[0].b64_json)
            return image_data
            
        except Exception as e:
            self.logger.error(f"Error in DALL-E API call: {e}")
            print(f"❌ DALL-E API error: {e}")
            return None
            
    def _process_image(self, image_data: bytes,
                      platform_specs: Dict[str, Any]) -> bytes:
        """Process the generated image according to platform specifications."""
        if not image_data:
            return None
            
        try:
            image = Image.open(BytesIO(image_data))
            
            # Apply platform-specific processing
            target_size = platform_specs.get('size', (1024, 1024))
            image = image.resize(target_size, Image.LANCZOS)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Save processed image to bytes
            output_buffer = BytesIO()
            image.save(output_buffer, format='JPEG', quality=95)
            return output_buffer.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return image_data  # Return original image data if processing fails
            
    def _generate_mock_image(self) -> bytes:
        """Generate a mock image when API key is not available."""
        # Create a more visually distinct placeholder image
        img = Image.new('RGB', (1024, 1024), color='#f0f0f0')
        
        # Add a message to the image
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Add placeholder text
        message = "Placeholder Image\nDALL-E Generation Failed"
        draw.text((512, 512), message, 
                 fill='#333333', 
                 anchor="mm",  # Center align
                 align="center")
                 
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()
