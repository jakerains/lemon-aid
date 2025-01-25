"""
Lemon-Aid: A tool for generating high-quality training data using multiple LLM providers.
"""

import json
import os
import asyncio
import random
import aiohttp
from openai import AsyncOpenAI
from dotenv import load_dotenv
import itertools
from typing import List, Dict, Any, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.live import Live
from tqdm import tqdm
import time
import backoff
from llm_providers import select_provider, select_model, create_client, LLMProvider
import sys

# Initialize Rich console
console = Console()

def create_header() -> Panel:
    """Create the header panel with ASCII art."""
    try:
        with open('assets/lemon-aid-big-ascii-art.txt', 'r', encoding='utf-8') as f:
            ascii_art = f.read().splitlines()
            # Filter empty lines at start and end
            while ascii_art and not ascii_art[0].strip():
                ascii_art.pop(0)
            while ascii_art and not ascii_art[-1].strip():
                ascii_art.pop()
            
            # Create styled text with colors
            styled_art = Text()
            
            # Process each line
            for i, line in enumerate(ascii_art):
                if "LEMON-AID" in line:
                    # Split and color the LEMON-AID line
                    parts = line.split("LEMON-AID")
                    styled_art.append(parts[0], style="yellow")
                    styled_art.append("LEMON-AID", style="green")
                    if len(parts) > 1:
                        styled_art.append(parts[1], style="yellow")
                    styled_art.append("\n")
                elif 2 <= i <= 4 and "⣿" in line:  # Only color the leaf area (lines 2-4)
                    styled_art.append(line, style="green")
                    styled_art.append("\n")
                else:  # The rest of the lemon in yellow
                    styled_art.append(line, style="yellow")
                    styled_art.append("\n")
            
            return Panel(
                styled_art,
                box=box.ROUNDED,
                border_style="yellow",
                padding=(1, 2),
                title="🍋 [bold yellow]Welcome to Lemon-Aid v1.0.2[/bold yellow] 🍋",
                subtitle="[bright_white]Easy Training Data infused with Citrus![/bright_white]"
            )
    except FileNotFoundError:
        return Panel(
            "[bold yellow]Welcome to Lemon-Aid![/bold yellow]\nEasy Training Data infused with Citrus!",
            box=box.ROUNDED,
            border_style="yellow",
            title="🍋 [bold yellow]Welcome to Lemon-Aid v1.0.2[/bold yellow] 🍋"
        )

def create_main_layout() -> Layout:
    """Create the main application layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="header"),
        Layout(name="main", ratio=3),
        Layout(name="footer")
    )
    return layout

def display_model_selection(provider: LLMProvider) -> Optional[str]:
    """Display model selection in a styled interface."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        transient=True,
    ) as progress:
        progress.add_task("[cyan]Loading available models...", total=None)
        model = select_model(provider)
        
        if not model:
            return None
            
        config_panel = Panel(
            f"[green]Selected Configuration:[/green]\n\n"
            f"[bold]Provider:[/bold] [cyan]{provider.name}[/cyan]\n"
            f"[bold]Model:[/bold] [cyan]{model}[/cyan]\n"
            f"[bold]Context Window:[/bold] [cyan]{provider.context_window or 'Default'}[/cyan]\n"
            f"[bold]Max Tokens:[/bold] [cyan]{provider.max_tokens}[/cyan]",
            box=box.ROUNDED,
            border_style="green",
            title="🔧 [bold]Configuration[/bold]",
            padding=(1, 2)
        )
        console.print(config_panel)
        
        return model

async def get_user_preferences() -> Dict[str, Any]:
    """Get user preferences for training data generation."""
    console.print("\n[bold cyan]Training Data Configuration[/bold cyan]")
    
    # Get purpose/context
    console.print(
        "\n[yellow]What is the purpose or context of your training data?[/yellow]\n"
        "Describe the specific use case, target audience, and desired behavior.\n"
        "[dim]Examples:[/dim]\n"
        "- A chatbot for K-12 classrooms that helps students learn about AI interactively\n"
        "- Customer service assistant for a tech startup's help desk\n"
        "- Educational tutor specializing in high school mathematics\n"
    )
    purpose = Prompt.ask("[cyan]Purpose/Context[/cyan]")
    console.print(f"[debug] Purpose/Context: {purpose}")
    
    # Get answer length preference
    console.print(
        "\n[yellow]How detailed should the answers be?[/yellow]\n"
        "[dim]Choose a length setting:[/dim]\n"
        "1. Concise (1-2 sentences)\n"
        "2. Moderate (2-3 sentences)\n"
        "3. Detailed (4-5 sentences)\n"
        "4. Comprehensive (6+ sentences)"
    )
    
    # Use a while loop to ensure valid input
    while True:
        try:
            length_choice = int(Prompt.ask(
                "[cyan]Answer length[/cyan]",
                choices=["1", "2", "3", "4"],
                default="2"
            ))
            if 1 <= length_choice <= 4:
                break
            console.print("[red]Please enter a number between 1 and 4.[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")
            continue
    
    console.print(f"[debug] Answer length choice: {length_choice}")
    
    # Map choice to token limits and style guide
    length_settings = {
        1: {"max_tokens": 100, "style": "concise and direct", "length_guide": "1-2 sentences"},
        2: {"max_tokens": 200, "style": "clear and informative", "length_guide": "2-3 sentences"},
        3: {"max_tokens": 400, "style": "detailed and thorough", "length_guide": "4-5 sentences"},
        4: {"max_tokens": 800, "style": "comprehensive and in-depth", "length_guide": "6+ sentences"}
    }
    console.print(f"[debug] Length settings: {length_settings[length_choice]}")
    
    # Get number of QA pairs
    console.print(
        "\n[yellow]How many question-answer pairs would you like to generate?[/yellow]\n"
        "[dim]Recommended: Start with 5-10 for testing, 50-100 for initial training[/dim]"
    )
    while True:
        try:
            num_pairs = IntPrompt.ask("[cyan]Number of pairs[/cyan]", default=10)
            if num_pairs > 0:
                break
            console.print("[red]Please enter a positive number.[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")
    console.print(f"[debug] Number of pairs: {num_pairs}")

    # Get output filename
    console.print("\n[yellow]Where would you like to save the training data?[/yellow]")
    output_file = Prompt.ask(
        "[cyan]Output filename[/cyan]",
        default="data/training_data.jsonl",
        show_default=True
    )
    console.print(f"[debug] Output filename: {output_file}")
    
    # Ensure filename ends with .jsonl
    if not output_file.endswith('.jsonl'):
        output_file += '.jsonl'

    # Show summary and confirm
    summary = Panel(
        f"[bold]Generation Settings[/bold]\n\n"
        f"[cyan]Purpose/Context:[/cyan]\n{purpose}\n\n"
        f"[cyan]Answer Style:[/cyan] {length_settings[length_choice]['style']}\n"
        f"[cyan]Number of Q&A Pairs:[/cyan] {num_pairs}\n"
        f"[cyan]Output File:[/cyan] {output_file}",
        title="📋 [bold]Summary[/bold]",
        border_style="cyan",
        box=box.ROUNDED
    )
    console.print("\n", summary)
    
    if not Prompt.ask("\n[yellow]Proceed with these settings?[/yellow]", choices=["y", "n"], default="y") == "y":
        return None
        
    return {
        "purpose": purpose,
        "num_entries": num_pairs,
        "max_tokens": length_settings[length_choice]["max_tokens"],
        "style": length_settings[length_choice]["style"],
        "output_file": output_file
    }

async def generate_qa_pair_async(client: AsyncOpenAI, prompt: str, system_prompt: str, model: str, max_tokens: int, temperature: float = 0.7) -> str:
    """Generates a question-answer pair using the selected LLM provider asynchronously."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        console.print(f"[red]Error generating response: {e}[/red]")
        return None

async def generate_batch(client, system_prompt: str, num_to_generate: int, max_tokens: int = 2048, model: str = None, provider = None) -> List[Tuple[str, str]]:
    """Generate a batch of question-answer pairs in parallel."""
    pairs = []
    seen_questions = set()
    
    # Set concurrency limits based on provider
    if isinstance(client, AsyncOpenAI):  # OpenAI client
        max_concurrent = 10  # Higher limit for OpenAI
        chunk_size = 5
    else:  # Other providers (including Ollama)
        max_concurrent = 2  # Lower limit for other providers
        chunk_size = 1  # Process one at a time for Ollama
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_single():
        """Generate a single Q&A pair with rate limiting."""
        try:
            async with semaphore:  # Control concurrent API calls
                prompt = (
                    "Generate a unique question and answer pair. Format your response exactly like this:\n"
                    "### User:\n[Your question here]\n\n"
                    "### Assistant:\n[Your answer here]\n\n"
                    "Requirements:\n"
                    "1. Answer must be 1-2 sentences long\n"
                    "2. Question must be unique and specific\n"
                    "3. Include practical, actionable information\n"
                    "4. Avoid self-references or AI mentions\n"
                    "5. Ensure answer is complete but concise\n"
                    "6. Never end mid-sentence or with ellipsis"
                )

                # Add timeout to prevent hanging requests
                async with asyncio.timeout(60):  # 60 second timeout for Ollama
                    if isinstance(client, AsyncOpenAI):
                        # OpenAI-compatible API call
                        response = await client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=max_tokens,
                            temperature=0.7,
                            presence_penalty=0.1,
                            frequency_penalty=0.1,
                            timeout=25
                        )
                        return response.choices[0].message.content.strip()
                    else:
                        # For Ollama, create a new session for each request
                        async with aiohttp.ClientSession() as session:
                            async with session.post(f"http://localhost:11434/api/chat", json={
                                "model": model,
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": prompt}
                                ],
                                "stream": False,
                                "options": {
                                    "temperature": 0.7,
                                    "num_predict": max_tokens
                                }
                            }) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    return result["message"]["content"].strip()
                                else:
                                    error_text = await response.text()
                                    console.print(f"[red]Error from Ollama: {error_text}[/red]")
                                    return None
                                
        except asyncio.TimeoutError:
            console.print("[yellow]Request timed out, retrying...[/yellow]")
            return None
        except Exception as e:
            console.print(f"[red]Error in parallel generation: {str(e)}[/red]")
            return None

    # Generate multiple pairs in parallel with chunking
    results = []
    
    for i in range(0, num_to_generate, chunk_size):
        chunk_tasks = [generate_single() for _ in range(min(chunk_size, num_to_generate - i))]
        chunk_responses = await asyncio.gather(*chunk_tasks, return_exceptions=True)
        results.extend(chunk_responses)
        
        # Add delay between chunks based on provider
        if not isinstance(client, AsyncOpenAI):
            await asyncio.sleep(1.0)  # Longer delay for Ollama
        else:
            await asyncio.sleep(0.1)  # Shorter delay for OpenAI
    
    # Process responses
    for content in results:
        if not content or isinstance(content, Exception):
            continue
            
        # Parse content based on markers
        if "### User:" in content and "### Assistant:" in content:
            try:
                # Extract question
                q_start = content.find("### User:") + len("### User:")
                q_end = content.find("### Assistant:", q_start)
                question = content[q_start:q_end].strip()
                
                # Extract answer
                a_start = content.find("### Assistant:") + len("### Assistant:")
                a_end = len(content)
                answer = content[a_start:a_end].strip()
                
                # Skip if already seen
                question_lower = question.lower()
                if question_lower in seen_questions:
                    continue
                    
                # Skip incomplete answers
                if answer.rstrip('.').endswith(':') or answer.endswith('...'):
                    continue
                    
                seen_questions.add(question_lower)
                pairs.append((question, answer))
                
            except Exception:
                continue
    
    return pairs

def clean_text(text: str) -> str:
    """Clean up text by removing markdown and normalizing characters."""
    replacements = {
        '**Student **': '',
        '**AI Chatbot Response:**': '',
        '**AI Response:**': '',
        '****': '',
        '\u2019': "'",  # Smart single quote
        '\u201c': '"',  # Smart left double quote
        '\u201d': '"',  # Smart right double quote
        '\u2014': '-',  # Em dash
        '\ud83e\udd14': '',  # Remove emoji
        'like me': '',  # Remove self-referential AI phrases
        'AI chatbot like me': 'AI chatbots',
        'I am an AI': 'AI assistants are',
        'as an AI': 'for an AI',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    text = text.replace('*', '')
    return text.strip()

def format_instruction(question: str, answer: str, system_prompt: str = None) -> str:
    """Format a Q&A pair with special tokens for the JSONL file."""
    if not question or not answer:
        return None
        
    # Clean up any trailing whitespace or newlines
    question = question.strip()
    answer = answer.strip()
    
    # Ensure the answer is complete (no trailing punctuation that suggests incompleteness)
    if answer.endswith('...') or answer.endswith(':') or answer.endswith(','):
        return None
        
    # Format with special tokens
    formatted = f"<|im_start|>system\n{system_prompt if system_prompt else 'You are a knowledgeable technical support assistant. Provide clear, accurate, and practical solutions to user questions.'}<|im_end|>\n"
    formatted += f"<|im_start|>user\n{question}<|im_end|>\n"
    formatted += f"<|im_start|>assistant\n{answer}<|im_end|>"
    
    return formatted

def get_dynamic_aspect(topic: str, existing_questions: set, attempt: int = 0) -> str:
    """Generate a more specific aspect based on duplicate rate and existing questions."""
    base_aspects = [
        "practical applications", "theoretical concepts", "future implications",
        "current challenges", "historical development", "ethical considerations",
        "technical details", "user experiences", "industry impact", "research directions"
    ]
    
    if attempt == 0:
        return random.choice(base_aspects)
    elif attempt == 1:
        return f"{random.choice(base_aspects)} in {random.choice(['healthcare', 'education', 'business', 'science', 'entertainment'])}"
    else:
        return (f"unique perspectives on {topic} relating to "
               f"{random.choice(['innovation', 'problem-solving', 'implementation', 'optimization', 'integration'])}")

def adapt_prompt_strategy(topic: str, current_aspect: str, duplicate_count: int) -> str:
    """Adapt the prompt based on duplicate rate."""
    if duplicate_count > 3:  # max_consecutive_duplicates
        return (
            f"Generate a completely unique question about {topic} that approaches the subject "
            f"from an unexpected angle. Think about connections between {topic} and other fields, "
            f"or consider edge cases and unusual scenarios.\n\n"
            f"The question should be something that hasn't been asked before and makes people "
            f"think differently about {topic}.\n\n"
            f"Current focus: {current_aspect}"
        )
    else:
        return (
            f"Generate a unique question about {topic} focusing on {current_aspect}.\n\n"
            f"Guidelines:\n"
            f"1. Question should explore {current_aspect} in detail\n"
            f"2. Must be different from previous questions\n"
            f"3. Should demonstrate genuine human curiosity\n"
            f"4. Focus on practical understanding\n"
            f"5. Avoid any AI self-references"
        )

def display_qa_pair(index: int, aspect: str, question: str, answer: str):
    """
    Display a Q&A pair with an index and aspect label for clarity.
    """
    console.print(f"\n[bold green]Example #{index} - Aspect: {aspect}[/bold green]")
    console.print("[bold green]Question:[/bold green]")
    console.print(f"[green]{question.strip()}[/green]")
    console.print("\n[bold blue]Answer:[/bold blue]")
    console.print(f"[blue]{answer.strip()}[/blue]")
    console.print("\n" + "─" * 80 + "\n")  # Separator line

async def generate_training_data(
    topic: str, 
    num_entries: int, 
    output_file: str = "data/training_data.jsonl",
    provider = None,
    model = None,
    client = None,
    preferences = None
):
    """Generates a JSONL training data file using the selected LLM provider."""
    def print_generation_stats(total_time: float, successful_entries: int, total_attempts: int):
        """Display formatted generation statistics."""
        stats_table = Table(show_header=False, box=box.SIMPLE)
        stats_table.add_column("Stat", style="cyan")
        stats_table.add_column("Value", style="yellow")
        
        avg_time = total_time / successful_entries if successful_entries > 0 else 0
        success_rate = (successful_entries/total_attempts*100) if total_attempts > 0 else 0
        
        stats_table.add_row("Total Time", f"{total_time:.1f} seconds")
        stats_table.add_row("Average Time per Entry", f"{avg_time:.1f} seconds")
        stats_table.add_row("Success Rate", f"{successful_entries}/{total_attempts} ({success_rate:.1f}%)")
        stats_table.add_row("Output File", output_file)
        
        console.print(Panel(
            stats_table,
            title="[bold]Generation Summary[/bold]",
            border_style="green",
            padding=(1, 2)
        ))

    async def save_progress(training_data, successful_entries):
        """Helper function to save current progress to file."""
        if training_data:
            temp_file = output_file + ".temp"
            try:
                with open(temp_file, "w", encoding='utf-8') as f:
                    for entry in training_data:
                        json.dump({"text": entry}, f, ensure_ascii=False)
                        f.write('\n')
                # If write successful, rename temp file to actual file
                os.replace(temp_file, output_file)
                console.print(Panel(
                    f"[green]Progress saved: {len(training_data)} entries written to {output_file}\n"
                    f"Generated {successful_entries} of {num_entries} requested entries[/green]",
                    title="Save Progress",
                    border_style="green",
                    padding=(1, 2)
                ))
            except Exception as e:
                console.print(f"[red]Error saving progress: {e}[/red]")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
            # Return whether save was successful
            return os.path.exists(output_file)
        return False

    try:
        # Initialize tracking variables
        generated_questions = set()
        training_data = []
        successful_entries = 0
        total_attempts = 0
        batch_size = 15  # Increased initial batch size
        min_batch_size = 5
        max_batch_size = 20
        
        # Main generation loop with progress tracking
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            "[cyan]Rate:[/cyan] {task.fields[rate]} entries/sec",
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Generating entries...", 
                total=num_entries,
                rate="calculating..."
            )
            
            start_time = time.time()
            last_save_time = start_time
            entries_since_save = 0
            last_batch_time = 0
            
            while successful_entries < num_entries:
                try:
                    remaining = num_entries - successful_entries
                    current_batch_size = min(batch_size, remaining)
                    total_attempts += current_batch_size
                    
                    batch_start_time = time.time()
                    batch_responses = await generate_batch(
                        client=client,
                        system_prompt=preferences['system_prompt'],
                        num_to_generate=current_batch_size,
                        max_tokens=preferences['max_tokens'],
                        model=model,
                        provider=provider
                    )
                    batch_time = time.time() - batch_start_time
                    
                    # Process batch responses
                    new_entries = len(batch_responses)
                    if new_entries > 0:
                        # Add all entries at once
                        formatted_entries = [
                            format_instruction(q, a, preferences['system_prompt']) 
                            for q, a in batch_responses
                        ]
                        formatted_entries = [e for e in formatted_entries if e is not None]
                        training_data.extend(formatted_entries)
                        successful_entries += len(formatted_entries)
                        entries_since_save += len(formatted_entries)
                        
                        # Update progress
                        current_time = time.time()
                        rate = successful_entries / (current_time - start_time)
                        progress.update(task, 
                            completed=successful_entries,
                            rate=f"{rate:.1f}"
                        )
                        
                        # Save progress after each successful batch
                        await save_progress(training_data, successful_entries)
                        last_save_time = current_time
                        entries_since_save = 0
                        
                        # Dynamically adjust batch size based on performance
                        if batch_time < last_batch_time and batch_size < max_batch_size:
                            batch_size = min(max_batch_size, batch_size + 2)
                        elif batch_time > last_batch_time * 1.5 and batch_size > min_batch_size:
                            batch_size = max(min_batch_size, batch_size - 2)
                        
                        last_batch_time = batch_time
                    
                    # Add small delay between batches to prevent rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    console.print(f"\n[red]Error in generation batch: {str(e)}[/red]")
                    console.print("[yellow]Attempting to continue with next batch...[/yellow]")
                    await asyncio.sleep(1)
                    continue
            
        # Print final summary
        total_time = time.time() - start_time
        print_generation_stats(total_time, successful_entries, total_attempts)
        console.print(f"\n[green]Successfully generated {successful_entries} training data entries.[/green]")
        
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]Generation interrupted. Saving progress...[/yellow]")
        # Force a save of any remaining data
        if training_data:
            saved = await save_progress(training_data, successful_entries)
            if saved:
                console.print("[green]Partial progress was successfully saved.[/green]")
            else:
                console.print("[red]Failed to save partial progress.[/red]")
        total_time = time.time() - start_time
        print_generation_stats(total_time, successful_entries, total_attempts)
        return
    except Exception as e:
        console.print(f"\n[red]An error occurred: {str(e)}[/red]")
        console.print("[yellow]Attempting to save progress...[/yellow]")
        await save_progress(training_data, successful_entries)
        total_time = time.time() - start_time
        print_generation_stats(total_time, successful_entries, total_attempts)
        raise

def generate_system_prompt(purpose: str) -> str:
    """Generate a system prompt based on the provided purpose."""
    sections = {
        "System Prompt": (
            f"You are a specialized AI assistant focused on {purpose}. "
            "You provide accurate, relevant, and well-structured information "
            "tailored to your specific domain."
        ),
        "Tone": (
            "Maintain a professional yet approachable tone. "
            "Be clear, concise, and engaging while remaining authoritative "
            "on the subject matter."
        ),
        "Approach": (
            "Focus on providing practical, actionable information. "
            "Break down complex concepts into understandable parts. "
            "Support explanations with relevant examples when helpful."
        )
    }
    
    return "\n\n".join(f"{header}:\n{content}" for header, content in sections.items())

async def generate_example_pairs(client, system_prompt: str, preferences: dict, model: str, provider: LLMProvider = None) -> bool:
    """Generate and display example pairs for user approval."""
    while True:  # Loop until approved or user exits
        console.print("\n[bold cyan]Generating Initial Examples[/bold cyan]")
        
        initial_aspects = [
            "fundamental concepts and best practices",
            "handling common customer issues",
            "escalation procedures and problem-solving",
            "technical troubleshooting steps",
            "customer satisfaction and follow-up"
        ]
        
        examples = await generate_batch(
            client=client,
            system_prompt=system_prompt,
            num_to_generate=5,  # Generate exactly 5 examples
            max_tokens=preferences['max_tokens'],
            model=model,
            provider=provider
        )
        
        if not examples:
            console.print("[red]Failed to generate examples. Please try again.[/red]")
            return False
            
        console.print("\n[bold cyan]Review these example Q&A pairs:[/bold cyan]")
        
        for i, (question, answer) in enumerate(examples):
            if question and answer:
                display_qa_pair(i+1, initial_aspects[i], question, answer)
        
        approval = Prompt.ask("\n[yellow]Do you approve these examples?[/yellow]", choices=["y", "n"], default="y")
        if approval == "y":
            return True
            
        # Get feedback for revision
        console.print("\n[cyan]Please provide feedback to improve the examples:[/cyan]")
        console.print("[dim]What would you like to change? (e.g., more technical, simpler answers, different topics, etc.)[/dim]")
        feedback = input().strip()
        
        if not feedback:
            console.print("[yellow]No feedback provided. Exiting...[/yellow]")
            return False
            
        # Update system prompt with feedback
        system_prompt_with_feedback = (
            f"{system_prompt}\n\n"
            f"Additional requirements based on feedback:\n{feedback}"
        )
        
        # Update the preferences with the modified prompt
        preferences['system_prompt'] = system_prompt_with_feedback
        
        console.print("\n[cyan]Regenerating examples with your feedback...[/cyan]")
        # Loop continues to generate new examples

async def main():
    """Main execution function."""
    preferences = None
    try:
        # Display header
        console.print(create_header())
        console.print()  # Add some spacing
        
        # Load environment variables
        load_dotenv()
        
        # Get provider and model selection
        try:
            provider = await select_provider()
            if not provider:
                console.print("[yellow]Process cancelled. Exiting...[/yellow]")
                return
        except (KeyboardInterrupt, EOFError):
            console.print("[yellow]Process interrupted by user.[/yellow]")
            return
            
        try:
            model = select_model(provider)
            if not model:
                console.print("[yellow]No model selected. Exiting...[/yellow]")
                return
        except (KeyboardInterrupt, EOFError):
            console.print("[yellow]Process interrupted by user.[/yellow]")
            return
            
        # Initialize the client
        client = create_client(provider)
        if not client:
            console.print("[red]Failed to initialize client. Please check your API keys.[/red]")
            return
            
        # Get user preferences including output file
        preferences = await get_user_preferences()
        if not preferences:
            console.print("[yellow]Configuration cancelled. Exiting...[/yellow]")
            return
            
        # Generate and revise system prompt until approved
        while True:
            system_prompt = generate_system_prompt(preferences['purpose'])
            console.print("\n[cyan]Generated System Prompt:[/cyan]")
            console.print(Panel(system_prompt, border_style="cyan"))
            
            approval = Prompt.ask("\n[yellow]Do you approve this system prompt?[/yellow]", choices=["y", "n"], default="y")
            if approval == "y":
                break
                
            # Get feedback for revision
            console.print("\n[cyan]Please provide feedback to improve the system prompt:[/cyan]")
            console.print("[dim]What would you like to change? (e.g., tone, approach, specific requirements, etc.)[/dim]")
            feedback = input().strip()
            
            if not feedback:
                console.print("[yellow]No feedback provided. Exiting...[/yellow]")
                return
                
            # Update purpose with feedback
            preferences['purpose'] = f"{preferences['purpose']}\nAdditional requirements: {feedback}"
            console.print("\n[cyan]Regenerating system prompt with your feedback...[/cyan]")
        
        # Store approved system prompt in preferences
        preferences['system_prompt'] = system_prompt
            
        # Generate and display example pairs
        try:
            examples_approved = await generate_example_pairs(client, system_prompt, preferences, model, provider)
            if not examples_approved:
                console.print("[yellow]Examples not approved. Exiting...[/yellow]")
                return
        except Exception as e:
            console.print(f"[red]Error generating examples: {str(e)}[/red]")
            return
            
        # Main generation loop
        try:
            await generate_training_data(
                topic=preferences['purpose'],
                num_entries=preferences['num_entries'],
                output_file=preferences['output_file'],
                provider=provider,
                model=model,
                client=client,
                preferences=preferences
            )
        except Exception as e:
            console.print(f"[red]Error during training data generation: {str(e)}[/red]")
            return
            
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]Process interrupted by user. Cleaning up...[/yellow]")
        if preferences and 'output_file' in preferences:
            try:
                temp_file = preferences['output_file'] + ".temp"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                console.print("[green]Cleanup completed.[/green]")
            except Exception as e:
                console.print(f"[red]Error during cleanup: {str(e)}[/red]")
        console.print("[green]Shutdown complete.[/green]")
        return
    except Exception as e:
        console.print(f"\n[red]An error occurred: {str(e)}[/red]")
        import traceback
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
    finally:
        try:
            # Clean up any temporary files
            if preferences and 'output_file' in preferences:
                temp_file = preferences['output_file'] + ".temp"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        except Exception:
            pass
        console.print("\n[green]Process completed.[/green]")
        sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Process interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]An error occurred: {str(e)}[/red]")
        import traceback
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
    finally:
        console.print("\n[green]Process completed.[/green]")
        sys.exit(0)