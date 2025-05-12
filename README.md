# AI-Powered Social Media Post Generator

A multi-agent system for generating brand-aligned social media content using AI. This project leverages natural language processing and computer vision to create engaging social media posts that maintain consistent brand messaging.

## Features

- Content analysis of existing brand posts to extract tone, style, and sentiment
- AI-powered text generation for social media posts
- Automated image generation aligned with brand guidelines
- Multi-platform support (Instagram, Twitter, LinkedIn)
- Customizable content parameters and brand guidelines
- Automated content saving and organization

## System Architecture

The system consists of four main agents:

1. **Content Analysis Agent**: Analyzes existing brand content to extract key characteristics
2. **Text Generation Agent**: Generates text-based social media posts
3. **Visual Generation Agent**: Creates matching visual content for posts
4. **Orchestrator Agent**: Coordinates the entire content generation pipeline

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/social_media_generator.git
cd social_media_generator
```

2. Create and activate a virtual environment:

```bash
python -m venv .env
source .env/bin/activate  # On Unix/macOS
.env\Scripts\activate     # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `config.env` file in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Configure your brand guidelines and generation parameters in `config.json` (optional):

```json
{
  "brand_guidelines": {
    "colors": ["#FF5733", "#33FF57", "#3357FF"],
    "style": "modern",
    "tone": "professional"
  },
  "generation_params": {
    "num_posts": 2,
    "post_type": "promotional",
    "target_audience": "young professionals"
  }
}
```

2. Run the generator:

```bash
python main.py
```

Generated content will be saved in the `generated_content` directory, with:

- JSON files containing text content and metadata
- JPEG files for generated images

## Configuration

### Platform-Specific Settings

The system includes optimized settings for different social media platforms:

- **Instagram**: 1:1 aspect ratio, 1080x1080px images
- **Twitter**: 16:9 aspect ratio, 1200x675px images
- **LinkedIn**: 1.91:1 aspect ratio, 1200x627px images

### Customization

You can customize various aspects of the generation process:

- Brand colors and style guidelines
- Content tone and messaging
- Target audience and content goals
- Platform-specific parameters
- Output formats and organization

## Dependencies

- OpenAI API for text and image generation
- NLTK for natural language processing
- TextBlob for sentiment analysis
- Pillow for image processing
- Additional utilities for async operations and file handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Hamza Adnan - 21I-0495
- Fatima Athar Khan - 21I-0385

## Acknowledgments

- OpenAI for providing the GPT and DALL-E APIs
- The NLTK and TextBlob communities for NLP tools
- Referenced research papers in social media content analysis and generation
