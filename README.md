# easyResearchAssistant

A FastAPI backend that acts as a **Load Balancer** for multiple Cloudflare AI Workers accounts. Distributes requests across accounts using Round Robin or Random selection, with automatic failover on rate limiting.

## Features

- **Multi-Account Load Balancing**: Distribute AI requests across 5+ Cloudflare accounts
- **Smart Rotation**: Round Robin or Random token selection strategies
- **Rate Limit Handling**: Automatic retry with different accounts on 429 errors
- **Streaming Support**: Real-time streaming responses from Llama 3
- **Family Security**: Simple password protection for API access
- **Easy Configuration**: Environment variable based setup

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your Cloudflare credentials
```

### 3. Run the Server

```bash
# Development
python main.py

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### `POST /ask`
Main endpoint for AI queries.

**Headers:**
- `X-Access-Password`: Your family access password

**Body:**
```json
{
    "prompt": "Tell me about quantum computing",
    "system_prompt": "You are a helpful assistant",
    "max_tokens": 2048,
    "temperature": 0.7,
    "stream": true
}
```

**Example with curl:**
```bash
# Streaming response
curl -X POST "http://localhost:8000/ask" \
  -H "X-Access-Password: your_password" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "stream": true}'

# Non-streaming response
curl -X POST "http://localhost:8000/ask" \
  -H "X-Access-Password: your_password" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "stream": false}'
```

### `GET /status`
Get load balancer status and active accounts.

### `POST /strategy/{strategy}`
Change rotation strategy (`round_robin` or `random`).

### `POST /reset`
Reset all failed accounts to active status.

### `GET /health`
Health check endpoint (no auth required).

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ACCESS_PASSWORD` | API access password | `family_secret_2024` |
| `ROTATION_STRATEGY` | `round_robin` or `random` | `round_robin` |
| `MAX_RETRIES` | Max retry attempts on 429 | `3` |
| `REQUEST_TIMEOUT` | Request timeout (seconds) | `60` |
| `CLOUDFLARE_MODEL` | AI model to use | `@cf/meta/llama-3-8b-instruct` |

## Setting Up Cloudflare Accounts

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Find your **Account ID** in the right sidebar
3. Go to **Profile → API Tokens** → Create Token
4. Use the **Workers AI (Read)** template or create custom with AI permissions
5. Add credentials to your `.env` file

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐
│   Client    │────▶│   FastAPI    │────▶│  Load Balancer      │
│             │◀────│   /ask       │◀────│  (Round Robin/      │
└─────────────┘     └──────────────┘     │   Random)           │
                                         └──────────┬──────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    │                               │                               │
                    ▼                               ▼                               ▼
           ┌────────────────┐              ┌────────────────┐              ┌────────────────┐
           │ Cloudflare     │              │ Cloudflare     │              │ Cloudflare     │
           │ Account 1      │              │ Account 2      │              │ Account N      │
           │ (Llama 3)      │              │ (Llama 3)      │              │ (Llama 3)      │
           └────────────────┘              └────────────────┘              └────────────────┘
```

## Rate Limit Handling

When an account hits a 429 (Too Many Requests) error:
1. The account is marked as temporarily failed
2. The request is automatically retried with the next available account
3. After `MAX_RETRIES` attempts, an error is returned
4. Failed accounts can be reset via `/reset` endpoint

## License

MIT
