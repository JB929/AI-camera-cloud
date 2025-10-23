# auth.py (final fixed for macOS + Python 3.13)
from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from starlette.status import HTTP_302_FOUND
from passlib.context import CryptContext
from sqlmodel import select
from db import get_session, init_db
from models import User

router = APIRouter()
# Use argon2 instead of bcrypt for macOS stability
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# Ensure database is initialized
init_db()

def ensure_default_admin():
    """Create admin user safely with argon2 (no 72-byte limit)."""
    with get_session() as sess:
        stmt = select(User).where(User.username == "admin")
        res = sess.exec(stmt).first()
        if not res:
            password = "admin123"
            hashed = pwd_context.hash(password)
            user = User(username="admin", hashed_password=hashed)
            sess.add(user)
            sess.commit()

ensure_default_admin()

@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return """
    <html><body style='font-family:sans-serif;text-align:center;margin-top:10%'>
      <h2>AI Monitor Login</h2>
      <form action='/login' method='post'>
        <input name='username' placeholder='Username'><br><br>
        <input type='password' name='password' placeholder='Password'><br><br>
        <button type='submit'>Login</button>
      </form>
    </body></html>
    """

@router.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    from db import get_session
    with get_session() as sess:
        stmt = select(User).where(User.username == username)
        user = sess.exec(stmt).first()
        if user and pwd_context.verify(password, user.hashed_password):
            response = RedirectResponse(url="/", status_code=HTTP_302_FOUND)
            response.set_cookie(key="user", value=username, httponly=True)
            return response
    return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=HTTP_302_FOUND)
    response.delete_cookie("user")
    return response

def get_current_user(request: Request):
    username = request.cookies.get("user")
    if not username:
        return None
    from db import get_session
    with get_session() as sess:
        stmt = select(User).where(User.username == username)
        user = sess.exec(stmt).first()
        if not user:
            return None
    return username

