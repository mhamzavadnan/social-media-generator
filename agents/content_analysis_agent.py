from typing import Dict, Any, List
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from collections import Counter
from .base_agent import BaseAgent

class ContentAnalysisAgent(BaseAgent):
    """Agent responsible for analyzing brand content and extracting key characteristics."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the content analysis agent.
        
        Args:
            name (str): Name of the agent
            config (Dict[str, Any], optional): Configuration parameters
        """
        super().__init__(name, config)
        self._initialize_nlp()
        
    def _initialize_nlp(self):
        """Initialize NLP components."""
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            self.logger.error(f"Error initializing NLP components: {e}")
            raise
            
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process brand content to extract key characteristics.
        
        Args:
            input_data (Dict[str, Any]): Input data containing posts to analyze
                Required fields:
                - posts (List[str]): List of text posts to analyze
                
        Returns:
            Dict[str, Any]: Analysis results including tone, sentiment, and style
        """
        if not self.validate_input(input_data, ['posts']):
            raise ValueError("Invalid input data")
            
        posts = input_data['posts']
        
        analysis_results = {
            'overall_sentiment': self._analyze_overall_sentiment(posts),
            'tone_analysis': self._analyze_tone(posts),
            'style_characteristics': self._analyze_style(posts),
            'common_topics': self._extract_topics(posts),
            'language_metrics': self._analyze_language_metrics(posts)
        }
        
        self.logger.info(f"Completed content analysis for {len(posts)} posts")
        return analysis_results
        
    def _analyze_overall_sentiment(self, posts: List[str]) -> Dict[str, float]:
        """Analyze overall sentiment of posts."""
        sentiments = []
        for post in posts:
            sentiment_scores = self.sentiment_analyzer.polarity_scores(post)
            sentiments.append(sentiment_scores)
            
        # Average sentiment scores
        avg_sentiment = {
            'positive': sum(s['pos'] for s in sentiments) / len(sentiments),
            'negative': sum(s['neg'] for s in sentiments) / len(sentiments),
            'neutral': sum(s['neu'] for s in sentiments) / len(sentiments),
            'compound': sum(s['compound'] for s in sentiments) / len(sentiments)
        }
        
        return avg_sentiment
        
    def _analyze_tone(self, posts: List[str]) -> Dict[str, float]:
        """Analyze the tone of posts."""
        tone_markers = {
            'professional': ['therefore', 'consequently', 'furthermore', 'moreover'],
            'casual': ['hey', 'cool', 'awesome', 'yeah'],
            'formal': ['hereby', 'accordingly', 'pursuant', 'whilst'],
            'friendly': ['thanks', 'please', 'welcome', 'happy']
        }
        
        tone_scores = {tone: 0.0 for tone in tone_markers}
        
        for post in posts:
            post_lower = post.lower()
            for tone, markers in tone_markers.items():
                score = sum(1 for marker in markers if marker in post_lower)
                tone_scores[tone] += score
                
        # Normalize scores
        total_scores = sum(tone_scores.values()) or 1
        return {tone: score/total_scores for tone, score in tone_scores.items()}
        
    def _analyze_style(self, posts: List[str]) -> Dict[str, Any]:
        """Analyze writing style characteristics."""
        style_metrics = {
            'avg_sentence_length': 0,
            'vocabulary_richness': 0,
            'emoji_usage': 0
        }
        
        total_words = 0
        unique_words = set()
        
        for post in posts:
            sentences = nltk.sent_tokenize(post)
            words = nltk.word_tokenize(post.lower())
            
            total_words += len(words)
            unique_words.update(words)
            style_metrics['avg_sentence_length'] += sum(len(nltk.word_tokenize(sent)) for sent in sentences)
            style_metrics['emoji_usage'] += len([c for c in post if self._is_emoji(c)])
            
        num_posts = len(posts)
        if num_posts > 0:
            style_metrics['avg_sentence_length'] /= num_posts
            style_metrics['vocabulary_richness'] = len(unique_words) / total_words if total_words > 0 else 0
            style_metrics['emoji_usage'] /= num_posts
            
        return style_metrics
        
    def _extract_topics(self, posts: List[str]) -> List[str]:
        """Extract common topics from posts."""
        words = []
        for post in posts:
            tokens = nltk.word_tokenize(post.lower())
            pos_tags = nltk.pos_tag(tokens)
            # Extract nouns and proper nouns
            nouns = [word for word, pos in pos_tags if pos.startswith(('NN', 'NNP'))]
            words.extend(nouns)
            
        # Get most common topics
        return [word for word, _ in Counter(words).most_common(10)]
        
    def _analyze_language_metrics(self, posts: List[str]) -> Dict[str, float]:
        """Analyze various language metrics."""
        metrics = {
            'avg_post_length': 0,
            'avg_word_length': 0,
            'question_frequency': 0,
            'exclamation_frequency': 0
        }
        
        total_words = 0
        total_chars = 0
        
        for post in posts:
            words = nltk.word_tokenize(post)
            total_words += len(words)
            total_chars += sum(len(word) for word in words)
            metrics['question_frequency'] += post.count('?')
            metrics['exclamation_frequency'] += post.count('!')
            
        num_posts = len(posts)
        if num_posts > 0:
            metrics['avg_post_length'] = total_words / num_posts
            metrics['avg_word_length'] = total_chars / total_words if total_words > 0 else 0
            metrics['question_frequency'] /= num_posts
            metrics['exclamation_frequency'] /= num_posts
            
        return metrics
        
    @staticmethod
    def _is_emoji(character: str) -> bool:
        """Check if a character is an emoji."""
        return len(character) > 1 or ord(character) > 127
