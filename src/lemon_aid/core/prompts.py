"""Core prompt generation and management."""

from enum import Enum
from typing import Dict, Tuple

class PromptType(Enum):
    PERSONALITY = "personality"
    CONTENT = "content"

# Personality markers and their typical expressions
PERSONALITY_MARKERS = {
    "valley girl": {
        "expressions": ["like", "totally", "oh my god", "whatever", "as if"],
        "traits": "casual, trendy, expressive",
        "era": "90s",
        "temp_scaling": {
            0.3: "mild valley girl accent and occasional slang",
            0.7: "regular valley girl speech patterns and attitude",
            1.0: "extremely exaggerated valley girl, heavy use of slang and mannerisms"
        }
    },
    "pirate": {
        "expressions": ["arr", "matey", "ye", "landlubber", "savvy"],
        "traits": "adventurous, rough, seafaring",
        "era": "golden age of piracy",
        "temp_scaling": {
            0.3: "mild pirate accent and occasional seafaring terms",
            0.7: "regular pirate speech and mannerisms",
            1.0: "extremely exaggerated pirate talk, heavy use of nautical terms"
        }
    }
    # Add more personality types as needed
}

def detect_prompt_type(context: str) -> Tuple[PromptType, float]:
    """Detect whether the prompt is personality-based or content-based.
    
    Args:
        context: The user's prompt describing desired assistant behavior
        
    Returns:
        Tuple of (PromptType, suggested_temperature)
    """
    # Check for personality keywords
    context_lower = context.lower()
    
    # Look for personality markers
    for personality in PERSONALITY_MARKERS:
        if personality in context_lower:
            return PromptType.PERSONALITY, 0.7  # Default mid-range temperature
            
    # Content-focused roles typically need lower temperature
    content_indicators = ["scientist", "expert", "professional", "specialist"]
    if any(indicator in context_lower for indicator in content_indicators):
        return PromptType.CONTENT, 0.3
        
    return PromptType.CONTENT, 0.5  # Default for unknown types

def adjust_personality_intensity(context: str, temperature: float) -> str:
    """Adjust personality traits based on temperature.
    
    Args:
        context: The personality context
        temperature: Desired intensity (0.0 to 1.0)
        
    Returns:
        Modified context with intensity guidance
    """
    context_lower = context.lower()
    
    for personality, data in PERSONALITY_MARKERS.items():
        if personality in context_lower:
            # Find the closest temperature scaling
            temp_thresholds = sorted(data["temp_scaling"].keys())
            closest_temp = min(temp_thresholds, key=lambda x: abs(x - temperature))
            intensity_guide = data["temp_scaling"][closest_temp]
            
            return f"""You are a {context} with specific intensity:
            Intensity Level: {intensity_guide}
            Common Expressions: {', '.join(data['expressions'])}
            Core Traits: {data['traits']}
            Era/Style: {data['era']}"""
            
    return context

def generate_system_prompt(context: str, style: str, temperature: float = 0.7) -> str:
    """Generate a system prompt that properly enforces the desired personality.
    
    Args:
        context: The desired assistant personality/context
        style: The desired response style (concise, moderate, etc.)
        temperature: The intensity of personality traits (0.0 to 1.0)
    """
    prompt_type, suggested_temp = detect_prompt_type(context)
    
    if prompt_type == PromptType.PERSONALITY:
        context = adjust_personality_intensity(context, temperature)
    
    base_prompt = f"""You are {context}. Never break character.

Core Traits:
- Fully embody the personality and mannerisms described
- Stay consistent with the character's typical behavior
- Use appropriate language and expressions for this persona
- Maintain character-specific quirks and attitudes

Style:
- Keep responses {style}
- Use vocabulary and tone fitting for this character
- Express yourself as this character would
- Stay true to the character's perspective

Remember: Always respond as your character would naturally speak and behave."""

    return base_prompt

# Examples of how this adapts:
# For a pirate: "Arr matey!" style language, sea references
# For goofy: "Gawrsh!" and silly, playful responses
# For mean/sarcastic: Condescending, rude remarks

# The resulting JSONL system prompt will be cleaner and more focused
# while still maintaining the mean personality requirements

# Example system prompt output:
# You are a super mean, sarcastic, rude assistant. Your core personality traits are:
# - Extremely sarcastic and condescending... 

# Add model-specific fallback handling
async def generate_with_fallback(provider: 'BaseProvider', prompt: str, **kwargs):
    """Generate response with fallback for different models.
    
    Args:
        provider: The LLM provider instance
        prompt: The generation prompt
        **kwargs: Additional generation parameters
    """
    try:
        return await provider.generate(prompt, **kwargs)
    except Exception as e:
        if "gpt-4o-mini" in str(provider.model):
            # Fallback settings for mini model
            kwargs["max_tokens"] = min(kwargs.get("max_tokens", 1000), 500)
            kwargs["temperature"] = min(kwargs.get("temperature", 0.7), 0.5)
            return await provider.generate(prompt, **kwargs)
        raise 