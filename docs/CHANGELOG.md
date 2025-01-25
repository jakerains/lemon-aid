# Changelog

## [Unreleased]

### Added
- Multi-provider LLM support with unified interface
- Support for OpenAI GPT-4o models
- Support for DeepSeek V3 and R1 models
- Support for Groq's Llama 3.x models
- Support for Hugging Face Inference API
- Provider and model selection system
- Rich console interface for provider/model selection
- Comprehensive environment configuration system

### Changed
- Updated to latest model versions across all providers
- Improved provider configuration management
- Enhanced documentation and examples
- Removed debug prints and simplified Q&A display
- Streamlined generation process output

### Dependencies
- Added huggingface-hub>=0.20.3 for Hugging Face support
- Requires openai>=1.0.0 for latest API compatibility

### Fixed
- Renamed calls from display_qa_example(...) to display_qa_pair(...) to avoid NameError.
- Completed the exception handling block for system prompt generation to prevent "conso" reference errors.
- Corrected generate_batch(...) calls to include client=client, avoiding missing-argument issues.
- Ensured system prompt approval logic cleanly exits or continues without re-entering unexpectedly.

## [1.0.0] - 2024-03-19

### Added
- Initial release of Lemon-Aid
- Asynchronous training data generation using DeepSeek API
- Dynamic prompt adaptation system
- Duplicate detection and avoidance
- Progress tracking and saving
- Rich console interface
- System prompt approval workflow
- Example Q&A pair validation
- Comprehensive error handling
- Documentation and project structure
- Requirements.txt for dependency management 