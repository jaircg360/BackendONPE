from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models import UserSignUp, UserSignIn, AuthResponse, UserResponse
from database import get_supabase_anon, get_supabase
from supabase import Client
from typing import Optional

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()

@router.post("/signup", response_model=AuthResponse)
async def sign_up(user_data: UserSignUp, supabase: Client = Depends(get_supabase_anon)):
    """Registro de nuevo usuario"""
    try:
        # Crear usuario en Supabase Auth
        auth_response = supabase.auth.sign_up({
            "email": user_data.email,
            "password": user_data.password,
            "options": {
                "data": {
                    "full_name": user_data.full_name
                }
            }
        })
        
        if auth_response.user is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se pudo crear el usuario"
            )
        
        # Crear perfil
        profile_data = {
            "id": auth_response.user.id,
            "email": user_data.email,
            "full_name": user_data.full_name
        }
        
        supabase.table("profiles").insert(profile_data).execute()
        
        # Asignar rol de usuario por defecto
        role_data = {
            "user_id": auth_response.user.id,
            "role": "user"
        }
        supabase.table("user_roles").insert(role_data).execute()
        
        return AuthResponse(
            user=UserResponse(
                id=auth_response.user.id,
                email=auth_response.user.email or user_data.email,
                full_name=user_data.full_name,
                is_admin=False
            ),
            access_token=auth_response.session.access_token if auth_response.session else "",
            token_type="bearer"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/signin", response_model=AuthResponse)
async def sign_in(user_data: UserSignIn, supabase: Client = Depends(get_supabase_anon)):
    """Inicio de sesión"""
    try:
        # Autenticar con Supabase
        auth_response = supabase.auth.sign_in_with_password({
            "email": user_data.email,
            "password": user_data.password
        })
        
        if auth_response.user is None or auth_response.session is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credenciales inválidas"
            )
        
        # Verificar si es admin
        role_response = supabase.table("user_roles")\
            .select("role")\
            .eq("user_id", auth_response.user.id)\
            .eq("role", "admin")\
            .execute()
        
        is_admin = len(role_response.data) > 0
        
        # Obtener perfil
        profile_response = supabase.table("profiles")\
            .select("full_name")\
            .eq("id", auth_response.user.id)\
            .single()\
            .execute()
        
        full_name = profile_response.data.get("full_name") if profile_response.data else None
        
        return AuthResponse(
            user=UserResponse(
                id=auth_response.user.id,
                email=auth_response.user.email or user_data.email,
                full_name=full_name,
                is_admin=is_admin
            ),
            access_token=auth_response.session.access_token,
            token_type="bearer"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales inválidas"
        )

@router.post("/signout")
async def sign_out(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase_anon)
):
    """Cerrar sesión"""
    try:
        # Establecer el token en el cliente
        supabase.auth.set_session(credentials.credentials, "")
        supabase.auth.sign_out()
        return {"message": "Sesión cerrada correctamente"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase_anon)
):
    """Obtener información del usuario actual"""
    try:
        # Obtener usuario desde el token
        user = supabase.auth.get_user(credentials.credentials)
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido"
            )
        
        # Verificar si es admin
        role_response = supabase.table("user_roles")\
            .select("role")\
            .eq("user_id", user.user.id)\
            .eq("role", "admin")\
            .execute()
        
        is_admin = len(role_response.data) > 0
        
        # Obtener perfil
        profile_response = supabase.table("profiles")\
            .select("full_name")\
            .eq("id", user.user.id)\
            .single()\
            .execute()
        
        full_name = profile_response.data.get("full_name") if profile_response.data else None
        
        return UserResponse(
            id=user.user.id,
            email=user.user.email or "",
            full_name=full_name,
            is_admin=is_admin
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No autorizado"
        )

