# Setup Guide - OpenAI Model Tiering

## Prerequisites

- Python 3.11+
- uv package manager (for dependency management)
- OpenAI API account with API key

## Installation Steps

### 1. Clone/Navigate to Project

```bash
cd /path/to/model_tiering_openai
```

### 2. Install Dependencies

```bash
uv sync
```

This installs:
- `openai>=1.0.0` - OpenAI Python SDK
- `python-dotenv>=1.0.0` - Environment variable management
- `pydantic>=2.0.0` - Data validation

### 3. Configure OpenAI API Key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your OpenAI API key
nano .env  # or use your preferred editor
```

Add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

You can get your API key from: https://platform.openai.com/account/api-keys

### 4. Run Demos

Test your setup with the quick demo:

```bash
uv run demo
```

Or run the comprehensive example suite:

```bash
uv run example
```

Alternative: Run directly with Python:
```bash
python main.py      # Quick demo
python example.py   # Comprehensive examples
```

## Troubleshooting

### ModuleNotFoundError: No module named 'openai'

Ensure dependencies are installed:
```bash
uv sync
```

### API Key Not Found

Verify:
1. `.env` file exists in the project root
2. `OPENAI_API_KEY` is properly set
3. API key is valid and active on OpenAI platform

Check with:
```bash
cat .env | grep OPENAI_API_KEY
```

### Authentication Error

If you get an authentication error when running examples that use the API:
- Verify API key is correct (should start with `sk-`)
- Check API key hasn't expired or been revoked
- Ensure you have available credits on your OpenAI account

## Project Structure

```
model_tiering_openai/
├── main.py              # Core routing logic
├── example.py           # Comprehensive examples
├── .env.example         # Environment template
├── .env                 # Your credentials (git-ignored)
├── .gitignore           # Excludes .env and __pycache__
├── pyproject.toml       # Project dependencies
├── README.md            # Documentation
└── SETUP.md             # This file
```

## Next Steps

1. **Rule-Based Routing Demo** - Run with no API calls needed:
   ```python
   from main import RuleBasedRouter
   router = RuleBasedRouter()
   model = router.route("Translate hello to Spanish").model
   ```

2. **Full Integration** - Use the hybrid router with your own tasks:
   ```python
   from main import HybridRouter
   router = HybridRouter()
   result = router.execute("Your task here")
   print(f"Cost: ${result.cost:.6f}")
   ```

3. **Cost Analysis** - Estimate savings:
   ```python
   from main import CostEstimator
   estimates = CostEstimator.estimate_monthly_cost(requests_per_day=100)
   ```

## Security Notes

- Never commit `.env` file to version control
- Don't hardcode API keys in your code
- Use environment variables for all credentials
- Rotate API keys periodically
- Monitor usage in OpenAI console to prevent unexpected costs

## Support

For issues or questions:
- Check OpenAI documentation: https://platform.openai.com/docs
- Review API rate limits and pricing: https://platform.openai.com/account/billing/overview
- Check model availability at: https://platform.openai.com/account/rate-limits
