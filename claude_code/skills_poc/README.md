# FastAPI Authentication System

A production-ready FastAPI application with JWT-based authentication, SQLite database, and comprehensive user management.

## Features

- User registration with email and username
- JWT token-based authentication
- User profile retrieval
- Password change functionality
- Secure password hashing with bcrypt
- Input validation with Pydantic
- Comprehensive test coverage
- Interactive API documentation (Swagger UI)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

The `.env` file is already configured with a secure secret key. You can modify it if needed:

```bash
# .env file is already created with:
# - SECRET_KEY: Secure random key
# - DATABASE_URL: SQLite database path
# - CORS settings
```

### 3. Run the Application

```bash
uvicorn app.main:app --reload
```

The server will start at `http://localhost:8000`

### 4. Access API Documentation

Open your browser and navigate to:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Authentication Endpoints

#### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "SecurePass123",
  "full_name": "John Doe"
}

Response: 201 Created
{
  "id": 1,
  "email": "user@example.com",
  "username": "johndoe",
  "full_name": "John Doe",
  "is_active": true,
  "created_at": "2026-01-01T10:00:00"
}
```

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=SecurePass123

Response: 200 OK
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### User Endpoints (Protected)

#### Get Current User Profile
```http
GET /api/v1/users/me
Authorization: Bearer <token>

Response: 200 OK
{
  "id": 1,
  "email": "user@example.com",
  "username": "johndoe",
  "full_name": "John Doe",
  "is_active": true,
  "created_at": "2026-01-01T10:00:00"
}
```

#### Change Password
```http
PUT /api/v1/users/me/password
Authorization: Bearer <token>
Content-Type: application/json

{
  "current_password": "SecurePass123",
  "new_password": "NewSecurePass456"
}

Response: 200 OK
{
  "message": "Password updated successfully"
}
```

## Testing the API

### Using Swagger UI (Recommended)

1. Go to `http://localhost:8000/docs`
2. Click on `/api/v1/auth/register` endpoint
3. Click "Try it out"
4. Fill in the request body and execute
5. Register a user, then login to get a token
6. Click "Authorize" button at the top
7. Enter your token in the format: `Bearer <your-token>`
8. Now you can test protected endpoints

### Using curl

```bash
# Register a new user
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "username": "testuser",
    "password": "TestPass123",
    "full_name": "Test User"
  }'

# Login
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=TestPass123"

# Get user profile (replace TOKEN with actual token)
curl -X GET "http://localhost:8000/api/v1/users/me" \
  -H "Authorization: Bearer TOKEN"

# Change password
curl -X PUT "http://localhost:8000/api/v1/users/me/password" \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "current_password": "TestPass123",
    "new_password": "NewSecurePass456"
  }'
```

## Running Tests

Run the test suite with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=app

# Run specific test file
pytest tests/api/test_auth.py -v
```

## Project Structure

```
app/
├── main.py                    # FastAPI application entry point
├── config.py                  # Settings and environment variables
├── database.py                # SQLAlchemy setup
├── dependencies.py            # Shared dependencies (auth, db)
├── api/v1/
│   ├── router.py              # Aggregate v1 routes
│   └── endpoints/
│       ├── auth.py            # Registration and login
│       └── users.py           # User profile and password change
├── models/
│   └── user.py                # User database model
├── schemas/
│   ├── user.py                # User Pydantic schemas
│   └── auth.py                # Token and auth schemas
├── crud/
│   └── user.py                # User CRUD operations
└── core/
    ├── security.py            # Password hashing and JWT
    └── exceptions.py          # Custom exceptions

tests/
├── conftest.py                # Test fixtures
└── api/
    ├── test_auth.py           # Auth endpoint tests
    └── test_users.py          # User endpoint tests
```

## Security Features

- Passwords hashed with bcrypt (cost factor: 12)
- JWT tokens with HS256 algorithm
- Token expiration (30 minutes by default)
- Password strength validation (min 8 chars, uppercase, lowercase, digit)
- CORS properly configured
- Security headers (X-Content-Type-Options, X-Frame-Options, X-XSS-Protection)
- SQL injection protection via SQLAlchemy ORM

## Environment Variables

Edit `.env` to configure:

```bash
# Security
SECRET_KEY=<your-secret-key>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=sqlite:///./auth.db

# Application
PROJECT_NAME=FastAPI Auth System
VERSION=1.0.0
API_V1_STR=/api/v1

# CORS (comma-separated origins)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
```

## Future Enhancements

- Email verification on registration
- Password reset flow with email tokens
- Refresh tokens for longer sessions
- Rate limiting on auth endpoints
- User roles and permissions
- OAuth2 integration (Google, GitHub)
- PostgreSQL support for production

---

# Skills

## What are Skills?

**Skills** are reusable, model-invoked capabilities that teach Claude how to do something specific. They're markdown files that extend Claude's knowledge without requiring explicit invocation. When you ask Claude something that matches a Skill's purpose, Claude automatically applies it.

Think of Skills as specialized instructions that Claude learns about at startup:
- Review pull requests using your team's specific standards
- Generate commit messages in your preferred format
- Explain code with visual diagrams

**Key difference from slash commands**: Claude chooses when to use Skills automatically based on your request, while slash commands require you to type `/command` explicitly.

## Setup

### File Structure

Skills are organized by location:

| Location   | Path                    | Applies to                        |
|:-----------|:------------------------|:----------------------------------|
| Personal   | `~/.claude/skills/`     | You, across all projects          |
| Project    | `.claude/skills/`       | Anyone working in this repository |
| Plugin     | Bundled with plugins    | Anyone with the plugin installed  |
| Enterprise | Organization-wide       | All users in your organization    |

### Basic Steps

1. **Create a directory** for your Skill (e.g., `~/.claude/skills/my-skill/`)
2. **Write `SKILL.md`** with YAML frontmatter and Markdown instructions
3. **Restart Claude Code** to load the new Skill
4. **Test it** by asking something that matches the description

## Quick Demo: Code Review Helper

**Directory structure:**
```
~/.claude/skills/code-review-helper/
└── SKILL.md
```

**`SKILL.md` content:**
```yaml
---
name: code-review-helper
description: Provides structured code review feedback with specific sections for improvements, strengths, and suggestions. Use when reviewing code, pull requests, or asking for feedback on code quality.
---

# Code Review Helper

## Instructions

When reviewing code, always follow this format:

1. **Strengths**: What's working well in this code? Highlight good patterns or decisions.
2. **Opportunities for improvement**: Point out specific issues (readability, performance, security).
3. **Suggested changes**: Provide concrete code examples or specific recommendations.
4. **Questions**: Ask clarifying questions about design decisions if needed.

## Best practices

- Be constructive and specific
- Reference line numbers or code sections
- Provide examples of better patterns
- Consider the context and constraints
```

### Create the Skill:

```bash
# Create directory
mkdir -p ~/.claude/skills/code-review-helper

# Create SKILL.md
cat > ~/.claude/skills/code-review-helper/SKILL.md << 'EOF'
---
name: code-review-helper
description: Provides structured code review feedback with specific sections for improvements, strengths, and suggestions. Use when reviewing code, pull requests, or asking for feedback on code quality.
---

# Code Review Helper

## Instructions

When reviewing code, always follow this format:

1. **Strengths**: What's working well in this code? Highlight good patterns or decisions.
2. **Opportunities for improvement**: Point out specific issues (readability, performance, security).
3. **Suggested changes**: Provide concrete code examples or specific recommendations.
4. **Questions**: Ask clarifying questions about design decisions if needed.

## Best practices

- Be constructive and specific
- Reference line numbers or code sections
- Provide examples of better patterns
- Consider the context and constraints
EOF

# Restart Claude Code to load the new Skill
```

### Test it:

Ask Claude: "Review this function for quality" or "Give me feedback on my code"

Claude will detect that your request matches the Skill's description, ask permission to use it, then follow the instructions.

## How Skills Work

1. **Discovery**: At startup, Claude loads all available Skill names and descriptions (not full content)
2. **Activation**: When your request matches a Skill's description, Claude asks permission to use it
3. **Execution**: After confirmation, Claude loads the full `SKILL.md` and follows its instructions

### Example Requests

With the code-review-helper Skill:

```
"Review this function for quality"      → Triggers the Skill
"How does this code work?"              → Won't trigger (description says "review")
"Give me feedback on my code changes"   → Triggers the Skill
```

## Best Practices

1. **Write clear descriptions**: Include specific keywords users would mention
2. **Keep SKILL.md focused**: Aim for under 500 lines
3. **Use progressive disclosure**: Link to detailed reference files instead of including everything in SKILL.md
4. **Restrict tools if needed**: Use `allowed-tools` in frontmatter to limit tool access:
   ```yaml
   allowed-tools: Read, Grep, Glob
   ```

## Advanced Features

### Multi-file Skills

```
~/.claude/skills/my-skill/
├── SKILL.md          # Main instructions
├── examples.md       # Reference examples
└── config.json       # Configuration data
```

Reference other files in SKILL.md:
```markdown
See examples.md for code samples
```

### Tool Restrictions

Limit which tools a Skill can use:

```yaml
---
name: read-only-skill
description: A skill that can only read files
allowed-tools: Read, Grep, Glob
---
```

## Skills vs Slash Commands

| Feature          | Skills                           | Slash Commands              |
|:-----------------|:---------------------------------|:----------------------------|
| Invocation       | Automatic (Claude decides)       | Manual (you type `/cmd`)    |
| Discovery        | Based on description match       | Explicit command name       |
| Use case         | Context-aware automation         | Explicit user-triggered     |
| Permission       | Asked first time per session     | Implicit (you invoked it)   |

## Cmd

View available Skills:
```
What Skills are available?
```

## Ref

- https://youtu.be/IoqpBKrNaZI?si=oWKM0tu6IeGpZDQm
- https://www.aiposthub.com/vibe-coding-claude-skills-guide/
- https://code.claude.com/docs/en/skills.md
