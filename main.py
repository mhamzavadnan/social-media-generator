import asyncio
import json
import os
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from agents.orchestrator_agent import OrchestratorAgent

# Load environment variables from config.env
load_dotenv('config.env')

async def main():
    """Main entry point for the social media content generation system."""
    
    # Load configuration
    config = load_config()
    
    # Initialize the orchestrator
    orchestrator = OrchestratorAgent("main_orchestrator", config)
    
    # Example input data
    input_data = {
        'existing_posts': [
            "Excited to announce our new product line! 🚀 #Innovation #Quality",
            "Customer satisfaction is our top priority. Thanks for your continued support! 💯",
            "Join us this weekend for our biggest sale of the year! Don't miss out! 🎉"
        ],
        'brand_guidelines': {
            'colors': ['#FF5733', '#33FF57', '#3357FF'],
            'style': 'modern',
            'tone': 'professional',
            'visual_preferences': {
                'image_style': 'minimalist',
                'composition': 'centered',
                'color_scheme': 'brand_colors'
            }
        },
        'generation_params': {
            'num_posts': 2,
            'post_type': 'promotional',
            'target_audience': 'young professionals',
            'content_goals': ['engagement', 'brand_awareness']
        },
        'platform': 'instagram'
    }
    
    try:
        # Generate content
        content_package = await orchestrator.process(input_data)
        
        # Save generated content
        save_content(content_package)
        
        print("Content generation completed successfully!")
        print(f"Generated {content_package['statistics']['total_posts']} posts")
        print(f"Posts with visuals: {content_package['statistics']['posts_with_visuals']}")
        
    except Exception as e:
        print(f"Error generating content: {e}")
        raise

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables or config file."""
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'platform_configs': {
            'instagram': {
                'size': (1080, 1080),
                'aspect_ratio': '1:1',
                'max_text_length': 2200
            },
            'twitter': {
                'size': (1200, 675),
                'aspect_ratio': '16:9',
                'max_text_length': 280
            },
            'linkedin': {
                'size': (1200, 627),
                'aspect_ratio': '1.91:1',
                'max_text_length': 3000
            }
        }
    }
    
    # Load additional configuration from file if exists
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    return config

def save_content(content_package: Dict[str, Any]) -> None:
    """Save generated content to output directory."""
    output_dir = 'generated_content'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a safe timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save text content and metadata
    output_file = os.path.join(output_dir, f'content_package_{timestamp}.json')
    
    try:
        # Create a copy of the package without binary image data for JSON serialization
        serializable_package = {
            'posts': [{
                'text': post['text'],
                'metadata': post['metadata']
            } for post in content_package['posts']],
            'statistics': content_package['statistics']
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_package, f, indent=2, ensure_ascii=False)
        
        print(f"Saved content package to: {output_file}")
        
        # Save images separately
        for i, post in enumerate(content_package['posts']):
            if post['visual']:
                image_file = os.path.join(output_dir, f'image_{timestamp}_{i}.jpg')
                try:
                    with open(image_file, 'wb') as f:
                        f.write(post['visual'])
                    print(f"Saved image to: {image_file}")
                except Exception as e:
                    print(f"Error saving image {i}: {e}")
    
    except Exception as e:
        print(f"Error saving content package: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
