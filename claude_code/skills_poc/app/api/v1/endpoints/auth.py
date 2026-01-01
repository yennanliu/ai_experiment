from datetime import timedelta

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.config import settings
from app.core.exceptions import InvalidCredentialsException, UserAlreadyExistsException
from app.core.security import create_access_token
from app.crud.user import crud_user
from app.database import get_db
from app.schemas.auth import Token
from app.schemas.user import UserCreate, UserResponse

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    user = crud_user.get_by_email(db, email=user_in.email)
    if user:
        raise UserAlreadyExistsException(detail="Email already registered")

    user = crud_user.get_by_username(db, username=user_in.username)
    if user:
        raise UserAlreadyExistsException(detail="Username already taken")

    user = crud_user.create(db, user_in=user_in)
    return user


@router.post("/login", response_model=Token)
def login(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
):
    user = crud_user.authenticate(db, email=form_data.username, password=form_data.password)

    if not user:
        raise InvalidCredentialsException()

    if not user.is_active:
        raise InvalidCredentialsException(detail="Inactive user")

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }
