"""Core generation logic for Lemon-Aid."""

from typing import List, Dict, Any
from ..providers.base import BaseProvider
from ..utils.types import GenerationResponse
from .prompts import generate_with_fallback, generate_system_prompt

async def generate_examples(provider: BaseProvider, context: str, style: str, temperature: float = 0.7) -> List[Dict[str, str]]:
    """Generate example Q&A pairs that match the specified personality."""
    
    system_prompt = generate_system_prompt(context, style, temperature)
    
    example_questions = [
        "What's a quick way to improve the flavor of a bland soup?",
        "How can I quickly clean up a greasy stovetop?",
        "What are some practical steps to prevent mold growth in a bathroom?",
        "What's the best way to remove coffee stains from a white shirt?",
        "How can I effectively remove adhesive residue from surfaces?"
    ]
    
    examples = []
    for question in example_questions:
        response = await generate_with_fallback(
            provider,
            prompt=f"{system_prompt}\n\nQuestion: {question}\nRespond in character:",
            max_tokens=100,
            temperature=temperature
        )
        examples.append({
            "question": question,
            "answer": response.text
        })
    
    return examples

# Example responses should now be more like:
# Q: "What's a quick way to improve the flavor of a bland soup?"
# A: "Oh great, another culinary disaster needs saving. Try actually learning to cook instead of ruining perfectly good ingredients, genius."
#
# Q: "How can I quickly clean up a greasy stovetop?"
# A: "Maybe try not making such a mess in the first place? *eye roll* Just use soap and water like any functioning adult would figure out." 