from typing import Dict, Any, List
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import asyncio
from .base_agent import BaseAgent

class TextGenerationAgent(BaseAgent):
    """Agent responsible for generating text-based social media content."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the text generation agent.
        
        Args:
            name (str): Name of the agent
            config (Dict[str, Any], optional): Configuration parameters including:
                - openai_api_key: OpenAI API key
                - model: OpenAI model to use (default: "gpt-4")
                - max_tokens: Maximum tokens per generation (default: 150)
        """
        super().__init__(name, config)
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize the language model."""
        load_dotenv('config.env')
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.logger.warning("No OpenAI API key provided. Text generation will be limited.")
            print("⚠️ No OpenAI API key found. Please check your config.env file.")
        else:
            try:
                self.client = OpenAI(api_key=api_key)
                # Test the connection
                self.client.models.list()
                print("✅ OpenAI API key is valid for text generation.")
            except Exception as e:
                self.logger.error(f"Error initializing OpenAI client: {e}")
                print(f"❌ Error connecting to OpenAI: {e}")
            
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate social media posts based on brand analysis and parameters.
        
        Args:
            input_data (Dict[str, Any]): Input data containing:
                - brand_analysis (Dict): Results from content analysis
                - num_posts (int): Number of posts to generate
                - post_type (str): Type of post to generate (e.g., "promotional", "engagement")
                - target_platform (str): Social media platform to target
                
        Returns:
            Dict[str, Any]: Generated posts and metadata
        """
        required_fields = ['brand_analysis', 'num_posts', 'post_type', 'target_platform']
        if not self.validate_input(input_data, required_fields):
            raise ValueError("Invalid input data")
            
        brand_analysis = input_data['brand_analysis']
        num_posts = input_data['num_posts']
        post_type = input_data['post_type']
        target_platform = input_data['target_platform']
        
        # Generate system prompt based on brand analysis
        system_prompt = self._create_system_prompt(brand_analysis, post_type, target_platform)
        
        generated_posts = []
        for i in range(num_posts):
            try:
                print(f"\nGenerating post {i+1}/{num_posts}...")
                post = await self._generate_post(system_prompt)
                if post:
                    generated_posts.append(post)
                    print(f"✅ Generated post {i+1}")
            except Exception as e:
                self.logger.error(f"Error generating post: {e}")
                print(f"❌ Failed to generate post {i+1}: {e}")
                continue
                
        return {
            'generated_posts': generated_posts,
            'metadata': {
                'post_type': post_type,
                'target_platform': target_platform,
                'success_rate': len(generated_posts) / num_posts if num_posts > 0 else 0
            }
        }
        
    def _create_system_prompt(self, brand_analysis: Dict[str, Any], 
                            post_type: str, target_platform: str) -> str:
        """Create a system prompt based on brand analysis."""
        tone = self._get_dominant_tone(brand_analysis.get('tone_analysis', {}))
        style = brand_analysis.get('style_characteristics', {})
        topics = brand_analysis.get('common_topics', [])
        
        prompt = f"""You are a social media content creator for a brand with the following characteristics:
        - Dominant tone: {tone}
        - Writing style: Average sentence length of {style.get('avg_sentence_length', 'moderate')} words
        - Vocabulary richness: {style.get('vocabulary_richness', 0.5):.2f}
        - Common topics: {', '.join(topics[:5])}
        
        Create a {post_type} post for {target_platform} that matches these characteristics.
        The post should be engaging, authentic, and align with the brand's voice.
        
        Additional guidelines:
        - Keep the post concise and platform-appropriate
        - Include relevant hashtags where appropriate
        - Maintain the brand's established tone and style
        - Focus on value and engagement
        """
        
        return prompt
        
    async def _generate_post(self, system_prompt: str) -> str:
        """Generate a single post using the language model."""
        if not hasattr(self, 'client'):
            return self._generate_mock_post()
            
        try:
            # Run the API call in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.get_config('model', 'gpt-4'),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Generate a social media post."}
                    ],
                    max_tokens=self.get_config('max_tokens', 150),
                    temperature=0.7
                )
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error in OpenAI API call: {e}")
            raise
            
    def _generate_mock_post(self) -> str:
        """Generate a mock post when API key is not available."""
        return "This is a placeholder post. Please provide an OpenAI API key for actual content generation."
        
    @staticmethod
    def _get_dominant_tone(tone_analysis: Dict[str, float]) -> str:
        """Get the dominant tone from tone analysis results."""
        if not tone_analysis:
            return "neutral"
        return max(tone_analysis.items(), key=lambda x: x[1])[0]
