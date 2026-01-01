from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.exceptions import AuthenticationException, UserNotFoundException
from app.core.security import decode_access_token, oauth2_scheme
from app.crud.user import crud_user
from app.database import get_db
from app.models.user import User


def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    payload = decode_access_token(token)

    if payload is None:
        raise AuthenticationException()

    user_id: str = payload.get("sub")
    if user_id is None:
        raise AuthenticationException()

    user = crud_user.get(db, user_id=int(user_id))
    if user is None:
        raise UserNotFoundException()

    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    if not current_user.is_active:
        raise AuthenticationException(detail="Inactive user")
    return current_user
