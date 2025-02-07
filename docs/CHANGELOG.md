# Changelog

## [1.2.0] - 2024-03-20

### Added
- Output directory organization for generated files
- User-selectable chat template formats
- Dynamic template selection per provider
- Improved progress tracking and file management
- Enhanced error handling and cleanup

### Changed
- Moved all output files to dedicated 'output' directory
- Updated file handling for better organization
- Improved temporary file management
- Enhanced user interface for template selection
- Better progress saving and error recovery

## [1.0.3] - 2024-03-19

### Changed
- Improved project launcher system with a dedicated `launch.py` script
- Fixed import conflicts and module organization
- Cleaned up project root directory structure
- Updated tagline to "Easy Training Data Generation infused with Citrus!"

### Fixed
- Resolved Python module import conflicts
- Fixed encoding issues in launcher scripts
- Improved project root organization

## [1.0.2] - 2024-03-19

### Changed
- Improved project structure with dedicated directories for assets, data, and source code
- Added launcher script for running from root directory
- Updated file paths to reflect new directory structure
- Simplified project organization and improved maintainability

## [1.0.1] - 2024-03-19

### Fixed
- Enhanced error handling for KeyboardInterrupt
- Fixed length settings reference in generate_batch function
- Corrected API endpoint for Groq provider
- Updated validation patterns for DeepSeek and Groq API keys
- Removed redundant error messages during shutdown

### Changed
- Version display in header updated to 1.0.1
- Improved progress saving frequency and error messages
- Simplified shutdown process
- Added tagline "Easy Training Data infused with Citrus!"

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

## [1.0.4] - 2024-03-19

### Added
- UV support for faster, more reliable package management
- Setup script for automated environment creation
- Improved installation documentation
- Development environment setup guide

### Changed
- Updated requirements.txt format for UV compatibility
- Simplified environment setup process
- Enhanced dependency management

### Fixed
- Changed the UV version check in `setup.py` to use the uv CLI command (`uv --version`) rather than `python -m uv --version` since the uv package does not support module execution

## [1.0.5] - 2024-03-19

### Fixed
- Fixed setup script to properly handle virtual environment creation
- Improved error handling in setup process
- Added proper pip initialization in virtual environment
- Enhanced setup script robustness and error reporting

## [1.1.0] - 2024-03-20

### Added
- Support for Hugging Face chat templates
- Multiple template formats (ChatML, Llama, Alpaca)
- Dynamic template selection per provider
- Custom template definition support
- Jinja2-based template rendering

### Changed
- Updated provider configurations to include chat template formats
- Enhanced message formatting for better compatibility
- Improved documentation with template examples 