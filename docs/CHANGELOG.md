# Changelog

All notable changes to the Lemon-Aid project will be documented in this file.

## [1.0.1] - 2024-03-19

### Fixed
- Improved error handling for KeyboardInterrupt (Ctrl+C)
- Fixed length settings reference in generate_batch function
- Corrected API endpoint for Groq provider
- Updated DeepSeek and Groq API key validation patterns
- Removed redundant error messages during shutdown

### Changed
- Updated version display in header to 1.0.1
- Improved progress saving frequency
- Enhanced error messages for better clarity
- Simplified shutdown process
- Added "Easy Training Data infused with Citrus!" tagline

### Added
- Dynamic model fetching for Ollama provider
- Better error tracking and reporting
- Improved progress feedback during generation

## [1.0.0] - 2024-03-19

### Added
- Initial release of Lemon-Aid
- Multi-provider LLM support (OpenAI, DeepSeek, Groq, Ollama)
- Asynchronous training data generation
- Dynamic prompt adaptation system
- Duplicate detection and avoidance
- Progress tracking and saving
- Rich console interface
- System prompt approval workflow
- Example Q&A pair validation
- Comprehensive error handling
- Documentation and project structure
- Requirements.txt for dependency management

### Changed
- Optimized batch processing
- Improved rate limiting
- Enhanced error handling
- Streamlined user interface
- Updated model validation for providers

### Dependencies
- Added aiohttp>=3.9.3 for async HTTP support
- Added backoff>=2.2.1 for retry logic
- Requires openai>=1.12.0 for latest API compatibility
- Added rich>=13.7.0 for console interface
- Added python-dotenv>=1.0.0 for environment management
- Added tqdm>=4.66.2 for progress tracking

### Fixed
- Corrected API key validation patterns for all providers
- Fixed response length control to better match user preferences
- Improved error handling during generation process
- Enhanced cleanup of temporary files
- Fixed shutdown behavior on interruption
- Corrected model validation for Groq and DeepSeek
- Improved handling of rate limits and timeouts 