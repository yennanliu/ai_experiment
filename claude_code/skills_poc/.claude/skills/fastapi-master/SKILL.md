---
name: fastapi-master
description: Expert guidance for building FastAPI applications including API design, authentication, database integration, dependency injection, testing, and production best practices. Use when building or working with FastAPI projects.
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, LSP
---

# FastAPI Master

Expert guidance for building production-ready FastAPI applications with best practices.

## Core Principles

When working with FastAPI, always follow these principles:

1. **Type Safety First** - Use Pydantic models for all request/response data
2. **Dependency Injection** - Leverage FastAPI's DI system for clean architecture
3. **Async by Default** - Use async/await for I/O operations
4. **Clear API Design** - RESTful principles with proper HTTP methods and status codes
5. **Security Built-in** - Authentication, authorization, and input validation
6. **Comprehensive Documentation** - Auto-generated with proper docstrings and examples

## Project Structure

### Recommended Layout

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app instance and startup
│   ├── config.py            # Settings with pydantic-settings
│   ├── dependencies.py      # Shared dependencies
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── router.py    # API router aggregation
│   │   │   └── endpoints/
│   │   │       ├── users.py
│   │   │       ├── items.py
│   │   │       └── auth.py
│   ├── models/              # SQLAlchemy/database models
│   │   ├── __init__.py
│   │   └── user.py
│   ├── schemas/             # Pydantic schemas
│   │   ├── __init__.py
│   │   └── user.py
│   ├── crud/                # Database operations
│   │   ├── __init__.py
│   │   └── user.py
│   ├── services/            # Business logic
│   │   ├── __init__.py
│   │   └── auth.py
│   └── core/                # Core functionality
│       ├── __init__.py
│       ├── security.py
│       └── exceptions.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── api/
│       └── test_users.py
├── alembic/                 # Database migrations
├── requirements.txt
└── .env
```

## Implementation Patterns

### 1. Application Setup

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.v1.router import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    # Initialize database, cache, etc.
    pass

@app.on_event("shutdown")
async def shutdown_event():
    # Cleanup resources
    pass
```

### 2. Configuration with Pydantic Settings

```python
# app/config.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Project"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Database
    DATABASE_URL: str

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### 3. Pydantic Schemas

```python
# app/schemas/user.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None

class UserInDB(UserBase):
    id: int
    is_active: bool = True
    created_at: datetime

    class Config:
        from_attributes = True  # For SQLAlchemy models

class User(UserInDB):
    pass
```

### 4. Dependency Injection

```python
# app/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.auth import decode_access_token
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    user_id: int = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception

    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
```

### 5. API Endpoints with Best Practices

```python
# app/api/v1/endpoints/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.schemas.user import User, UserCreate, UserUpdate
from app.crud.user import user_crud
from app.dependencies import get_current_active_user

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_in: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new user.

    - **email**: Valid email address
    - **username**: 3-50 characters
    - **password**: Minimum 8 characters
    """
    # Check if user exists
    existing_user = user_crud.get_by_email(db, email=user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    user = user_crud.create(db, obj_in=user_in)
    return user

@router.get("/me", response_model=User)
async def read_users_me(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information."""
    return current_user

@router.get("/{user_id}", response_model=User)
async def read_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get user by ID."""
    user = user_crud.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

@router.patch("/{user_id}", response_model=User)
async def update_user(
    user_id: int,
    user_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update user information."""
    user = user_crud.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Authorization check
    if user.id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user"
        )

    user = user_crud.update(db, db_obj=user, obj_in=user_in)
    return user

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete user."""
    user = user_crud.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Authorization check
    if user.id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this user"
        )

    user_crud.remove(db, id=user_id)
```

### 6. CRUD Operations

```python
# app/crud/user.py
from sqlalchemy.orm import Session
from typing import Optional, List
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from app.core.security import get_password_hash

class CRUDUser:
    def get(self, db: Session, id: int) -> Optional[User]:
        return db.query(User).filter(User.id == id).first()

    def get_by_email(self, db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    def get_multi(
        self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> List[User]:
        return db.query(User).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: UserCreate) -> User:
        db_obj = User(
            email=obj_in.email,
            username=obj_in.username,
            full_name=obj_in.full_name,
            hashed_password=get_password_hash(obj_in.password)
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self, db: Session, *, db_obj: User, obj_in: UserUpdate
    ) -> User:
        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def remove(self, db: Session, *, id: int) -> User:
        obj = db.query(User).get(id)
        db.delete(obj)
        db.commit()
        return obj

user_crud = CRUDUser()
```

### 7. Authentication with JWT

```python
# app/core/security.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError:
        return None
```

### 8. Testing

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import Base, get_db

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    del app.dependency_overrides[get_db]

# tests/api/test_users.py
def test_create_user(client):
    response = client.post(
        "/api/v1/users/",
        json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "testpassword123"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["username"] == "testuser"
    assert "id" in data
```

## Best Practices Checklist

### API Design
- [ ] Use proper HTTP methods (GET, POST, PUT, PATCH, DELETE)
- [ ] Return appropriate status codes
- [ ] Include response models for all endpoints
- [ ] Version your API (e.g., /api/v1/)
- [ ] Use plural nouns for resource endpoints (/users, /items)
- [ ] Implement pagination for list endpoints

### Security
- [ ] Use OAuth2 with Password (and hashing), Bearer with JWT tokens
- [ ] Hash passwords with bcrypt
- [ ] Validate all input with Pydantic
- [ ] Implement rate limiting for public endpoints
- [ ] Use HTTPS in production
- [ ] Set appropriate CORS policies
- [ ] Implement proper authorization checks

### Database
- [ ] Use async database drivers (asyncpg, aiomysql)
- [ ] Implement connection pooling
- [ ] Use migrations (Alembic)
- [ ] Index frequently queried fields
- [ ] Use transactions for multi-step operations
- [ ] Implement soft deletes when appropriate

### Performance
- [ ] Use async/await for I/O operations
- [ ] Implement caching (Redis)
- [ ] Use background tasks for long-running operations
- [ ] Optimize database queries (N+1 prevention)
- [ ] Enable response compression (gzip)
- [ ] Use CDN for static files

### Code Quality
- [ ] Type hints everywhere
- [ ] Comprehensive docstrings
- [ ] Unit tests with >80% coverage
- [ ] Integration tests for critical paths
- [ ] Proper error handling and custom exceptions
- [ ] Logging with appropriate levels
- [ ] Environment-based configuration

## Common Pitfalls to Avoid

1. **Don't mix sync and async** - Use async throughout or none at all
2. **Don't use mutable defaults** - Use `None` and create in function body
3. **Don't skip input validation** - Always use Pydantic models
4. **Don't hardcode secrets** - Use environment variables
5. **Don't forget database session cleanup** - Use dependencies properly
6. **Don't return raw SQLAlchemy models** - Use Pydantic response models
7. **Don't ignore type hints** - They're not just documentation
8. **Don't skip migrations** - Always use Alembic for schema changes

## Production Deployment

### Essential Dependencies
```txt
fastapi[all]
uvicorn[standard]
python-jose[cryptography]
passlib[bcrypt]
sqlalchemy
alembic
pydantic-settings
python-multipart
```

### Running in Production
```bash
# Use uvicorn with workers
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Or use gunicorn with uvicorn workers
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Quick Commands

When implementing FastAPI features:

1. **New endpoint**: Create in appropriate file under `app/api/v1/endpoints/`
2. **New model**: Add SQLAlchemy model in `app/models/` and Pydantic schema in `app/schemas/`
3. **New dependency**: Add to `app/dependencies.py`
4. **New service**: Create in `app/services/`
5. **Database migration**: `alembic revision --autogenerate -m "description"`
6. **Run tests**: `pytest tests/ -v --cov=app`

## Response Format

When providing FastAPI code:
- Include type hints for all functions
- Add docstrings to endpoints
- Show both the endpoint and related schemas
- Include error handling
- Demonstrate proper dependency injection
- Provide working, production-ready code
