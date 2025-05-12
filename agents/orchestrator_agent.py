from typing import Dict, Any, List
import asyncio
from .base_agent import BaseAgent
from .content_analysis_agent import ContentAnalysisAgent
from .text_generation_agent import TextGenerationAgent
from .visual_generation_agent import VisualGenerationAgent

class OrchestratorAgent(BaseAgent):
    """Agent responsible for orchestrating the content generation pipeline."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the orchestrator agent.
        
        Args:
            name (str): Name of the agent
            config (Dict[str, Any], optional): Configuration parameters including:
                - openai_api_key: OpenAI API key
                - platform_configs: Platform-specific configurations
        """
        super().__init__(name, config)
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize all sub-agents."""
        config = self.config or {}
        
        self.content_analyzer = ContentAnalysisAgent(
            "content_analyzer",
            config.get('content_analysis_config', {})
        )
        
        self.text_generator = TextGenerationAgent(
            "text_generator",
            config.get('text_generation_config', {})
        )
        
        self.visual_generator = VisualGenerationAgent(
            "visual_generator",
            config.get('visual_generation_config', {})
        )
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate the content generation process.
        
        Args:
            input_data (Dict[str, Any]): Input data containing:
                - existing_posts (List[str]): List of existing brand posts
                - brand_guidelines (Dict): Brand guidelines and preferences
                - generation_params (Dict): Parameters for content generation
                - platform (str): Target social media platform
                
        Returns:
            Dict[str, Any]: Generated content package including text and visuals
        """
        required_fields = ['existing_posts', 'brand_guidelines', 'generation_params', 'platform']
        if not self.validate_input(input_data, required_fields):
            raise ValueError("Invalid input data")
            
        try:
            # Step 1: Analyze existing content
            analysis_results = await self._analyze_content(input_data['existing_posts'])
            
            # Step 2: Generate text content
            text_content = await self._generate_text_content(
                analysis_results,
                input_data['generation_params'],
                input_data['platform']
            )
            
            # Step 3: Generate visual content
            visual_content = await self._generate_visual_content(
                text_content,
                input_data['brand_guidelines'],
                input_data['platform']
            )
            
            # Step 4: Compile final content package
            content_package = self._compile_content_package(
                text_content,
                visual_content,
                analysis_results
            )
            
            return content_package
            
        except Exception as e:
            self.logger.error(f"Error in content generation pipeline: {e}")
            raise
            
    async def _analyze_content(self, existing_posts: List[str]) -> Dict[str, Any]:
        """Analyze existing content using the ContentAnalysisAgent."""
        analysis_input = {
            'posts': existing_posts
        }
        
        try:
            analysis_results = await self.content_analyzer.process(analysis_input)
            self.logger.info("Content analysis completed successfully")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Error in content analysis: {e}")
            raise
            
    async def _generate_text_content(self,
                                   analysis_results: Dict[str, Any],
                                   generation_params: Dict[str, Any],
                                   platform: str) -> Dict[str, Any]:
        """Generate text content using the TextGenerationAgent."""
        text_input = {
            'brand_analysis': analysis_results,
            'num_posts': generation_params.get('num_posts', 1),
            'post_type': generation_params.get('post_type', 'general'),
            'target_platform': platform
        }
        
        try:
            text_content = await self.text_generator.process(text_input)
            self.logger.info(f"Generated {len(text_content['generated_posts'])} text posts")
            return text_content
        except Exception as e:
            self.logger.error(f"Error in text generation: {e}")
            raise
            
    async def _generate_visual_content(self,
                                     text_content: Dict[str, Any],
                                     brand_guidelines: Dict[str, Any],
                                     platform: str) -> List[Dict[str, Any]]:
        """Generate visual content for each text post."""
        visual_content = []
        platform_specs = self._get_platform_specs(platform)
        
        for post in text_content['generated_posts']:
            visual_input = {
                'text_content': post,
                'brand_guidelines': brand_guidelines,
                'style_preferences': self._get_style_preferences(text_content['metadata']),
                'platform_specs': platform_specs
            }
            
            try:
                visual_result = await self.visual_generator.process(visual_input)
                visual_content.append(visual_result)
            except Exception as e:
                self.logger.error(f"Error generating visual for post: {e}")
                visual_content.append(None)
                
        return visual_content
        
    def _compile_content_package(self,
                               text_content: Dict[str, Any],
                               visual_content: List[Dict[str, Any]],
                               analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile the final content package."""
        posts = []
        
        for text, visual in zip(text_content['generated_posts'], visual_content):
            post_package = {
                'text': text,
                'visual': visual['image_data'] if visual and visual['metadata']['generation_successful'] else None,
                'metadata': {
                    'text_metadata': text_content['metadata'],
                    'visual_metadata': visual['metadata'] if visual else None,
                    'analysis_context': analysis_results
                }
            }
            posts.append(post_package)
            
        return {
            'posts': posts,
            'statistics': {
                'total_posts': len(posts),
                'posts_with_visuals': sum(1 for p in posts if p['visual'] is not None),
                'generation_timestamp': self._get_timestamp()
            }
        }
        
    def _get_platform_specs(self, platform: str) -> Dict[str, Any]:
        """Get platform-specific specifications."""
        platform_configs = self.get_config('platform_configs', {})
        return platform_configs.get(platform, {})
        
    @staticmethod
    def _get_style_preferences(text_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract style preferences from text metadata."""
        return {
            'mood': text_metadata.get('post_type', 'neutral'),
            'style': 'modern',
            'composition': 'social_media_optimized'
        }
        
    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
