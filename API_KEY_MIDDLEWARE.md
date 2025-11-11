# API Key Middleware Documentation

## Overview

The API Key middleware adds authentication security to your FastAPI application. It validates API keys for protected endpoints while allowing public endpoints to remain accessible.

## Features

✅ **Simple header-based authentication** - Uses `X-API-Key` header  
✅ **Environment-based configuration** - Configure via environment variables  
✅ **Public endpoint whitelist** - Certain endpoints don't require authentication  
✅ **Master key support** - Single master key for all endpoints  
✅ **Multiple API keys** - Support for multiple valid API keys  
✅ **Easy enable/disable** - Toggle authentication on/off without code changes  

## Configuration

### Environment Variables

Add these to your `.env` file:

```env
# Enable/Disable API Key authentication (true/false)
ENABLE_API_KEY_AUTH=true

# Master API Key - Works for all endpoints
MASTER_API_KEY=your-secure-master-key

# Additional API Keys (comma-separated for multiple keys)
API_KEYS=key1,key2,key3
```

## Public Endpoints (No API Key Required)

The following endpoints are public and don't require authentication:

- `/` - Root endpoint
- `/health` - Health check
- `/docs` - Swagger documentation
- `/openapi.json` - OpenAPI spec
- `/redoc` - ReDoc documentation

All other endpoints require a valid API key.

## How to Use

### Client Request Example

```bash
# Using curl
curl -X POST http://localhost:8000/api/register \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"name":"John","email":"john@example.com","mobile":"1234567890"}'

# Using Python requests
import requests

headers = {
    "X-API-Key": "your-api-key"
}

response = requests.post(
    "http://localhost:8000/api/register",
    headers=headers,
    json={"name":"John","email":"john@example.com","mobile":"1234567890"}
)
```

### JavaScript/Fetch Example

```javascript
const apiKey = "your-api-key";

fetch('http://localhost:8000/api/register', {
  method: 'POST',
  headers: {
    'X-API-Key': apiKey,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    name: 'John',
    email: 'john@example.com',
    mobile: '1234567890'
  })
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

## Error Responses

### Missing API Key
```json
{
  "detail": "Missing API key. Please provide X-API-Key header."
}
```
**Status Code:** 401 Unauthorized

### Invalid API Key
```json
{
  "detail": "Invalid API key."
}
```
**Status Code:** 403 Forbidden

## Optional: Decorator-Based Protection

For endpoint-specific protection, use the `@require_api_key` decorator:

```python
from fastapi import APIRouter, Request
from app.middleware import require_api_key

router = APIRouter()

@router.post("/secure-endpoint")
@require_api_key
async def secure_endpoint(request: Request):
    """This endpoint requires API key"""
    return {"message": "Secure data"}
```

## Security Best Practices

1. **Use strong API keys** - Generate random, long keys (min 32 characters)
   ```python
   import secrets
   api_key = secrets.token_urlsafe(32)
   ```

2. **Rotate keys regularly** - Update `MASTER_API_KEY` periodically

3. **Use HTTPS in production** - API keys should always be sent over HTTPS

4. **Never commit keys** - Store in `.env` file, not in code

5. **Different keys for different services** - Use separate keys for different clients

6. **Monitor key usage** - Log API key access in your middleware (already implemented)

## Production Deployment

### Environment Setup
```bash
export ENABLE_API_KEY_AUTH=true
export MASTER_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export API_KEYS="key1,key2,key3"
```

### Docker Example
```dockerfile
ENV ENABLE_API_KEY_AUTH=true
ENV MASTER_API_KEY=your-secure-key
```

## Troubleshooting

### Getting "Missing API key" error

**Solution:** Ensure the `X-API-Key` header is included in your request

```bash
# Correct
curl -H "X-API-Key: your-key" http://localhost:8000/api/endpoint

# Incorrect (will fail)
curl http://localhost:8000/api/endpoint
```

### Getting "Invalid API key" error

**Solution:** Check that your API key matches one of:
- `MASTER_API_KEY`
- Keys in `API_KEYS` list

```python
# Verify your keys are set correctly
import os
print(os.getenv("MASTER_API_KEY"))
print(os.getenv("API_KEYS"))
```

### Disabling authentication temporarily

Set `ENABLE_API_KEY_AUTH=false` in `.env` and restart the application:

```bash
ENABLE_API_KEY_AUTH=false
```

## Logging

The middleware logs all authentication attempts:

```
✅ API Key authentication enabled
Missing API key for POST /api/register
Invalid API key attempt for POST /api/submit
```

Monitor these logs to detect unauthorized access attempts.

## Integration with Existing Routes

All existing routes automatically require API key authentication. No code changes needed!

The middleware works with:
- ✅ `/api/register` - Participant registration
- ✅ `/api/submit` - Resume submission
- ✅ `/api/participant/{participant_id}/scores` - Get scores
- ✅ `/api/participant/{participant_id}/upload-count` - Get upload count
- ✅ `/api/leaderboard` - Leaderboard data
- ✅ `/api/stats` - Competition statistics

## Testing

```python
# Test script to verify API key middleware

import requests

BASE_URL = "http://localhost:8000"
API_KEY = "your-api-key"

# Test 1: Health check (public, no key needed)
response = requests.get(f"{BASE_URL}/health")
print("Health check:", response.status_code)  # Should be 200

# Test 2: Register without API key (should fail)
response = requests.post(f"{BASE_URL}/api/register")
print("Register without key:", response.status_code)  # Should be 401

# Test 3: Register with API key (should work)
headers = {"X-API-Key": API_KEY}
response = requests.post(
    f"{BASE_URL}/api/register",
    headers=headers,
    json={"name":"Test","email":"test@example.com","mobile":"1234567890"}
)
print("Register with key:", response.status_code)  # Should be 200
```
