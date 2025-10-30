from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models import VoteCreate, VoteResponse, VoteCheck
from database import get_supabase, get_supabase_anon
from supabase import Client
from typing import List, Optional

router = APIRouter(prefix="/votes", tags=["Votes"])
security = HTTPBearer()

async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase_anon)
) -> str:
    """Obtener el ID del usuario actual desde el token"""
    try:
        user = supabase.auth.get_user(credentials.credentials)
        if user is None or user.user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido"
            )
        return user.user.id
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No autorizado"
        )

@router.post("", response_model=VoteResponse, status_code=status.HTTP_201_CREATED)
async def create_vote(
    vote: VoteCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase)
):
    """Registrar un voto"""
    try:
        # Obtener ID del usuario
        user_id = await get_current_user_id(credentials, supabase)
        
        # Verificar si el usuario ya votó
        existing_vote = supabase.table("votes")\
            .select("id")\
            .eq("user_id", user_id)\
            .execute()
        
        if existing_vote.data and len(existing_vote.data) > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ya has votado anteriormente"
            )
        
        # Verificar que el candidato existe
        candidate = supabase.table("candidates")\
            .select("id")\
            .eq("id", vote.candidate_id)\
            .execute()
        
        if not candidate.data or len(candidate.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Candidato no encontrado"
            )
        
        # Crear el voto
        vote_data = vote.model_dump()
        vote_data["user_id"] = user_id
        
        response = supabase.table("votes")\
            .insert(vote_data)\
            .execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se pudo registrar el voto"
            )
        
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al registrar voto: {str(e)}"
        )

@router.get("/check", response_model=VoteCheck)
async def check_user_vote(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase)
):
    """Verificar si el usuario ya votó"""
    try:
        user_id = await get_current_user_id(credentials, supabase)
        
        response = supabase.table("votes")\
            .select("candidate_id")\
            .eq("user_id", user_id)\
            .execute()
        
        if response.data and len(response.data) > 0:
            return VoteCheck(
                has_voted=True,
                candidate_id=response.data[0]["candidate_id"]
            )
        
        return VoteCheck(has_voted=False)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al verificar voto: {str(e)}"
        )

@router.get("/all", response_model=List[VoteResponse])
async def get_all_votes(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase)
):
    """Obtener todos los votos (solo para admin)"""
    try:
        user_id = await get_current_user_id(credentials, supabase)
        
        # Verificar si es admin
        role_response = supabase.table("user_roles")\
            .select("role")\
            .eq("user_id", user_id)\
            .eq("role", "admin")\
            .execute()
        
        if not role_response.data or len(role_response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tienes permisos para esta acción"
            )
        
        response = supabase.table("votes")\
            .select("*")\
            .order("created_at", desc=True)\
            .execute()
        
        return response.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener votos: {str(e)}"
        )

@router.get("/statistics")
async def get_vote_statistics(supabase: Client = Depends(get_supabase)):
    """Obtener estadísticas de votación"""
    try:
        # Total de votos
        total_response = supabase.table("votes")\
            .select("id", count="exact")\
            .execute()
        
        # Votos por candidato
        votes_by_candidate = supabase.table("votes")\
            .select("candidate_id", count="exact")\
            .execute()
        
        # Agrupar votos por candidato
        candidate_votes = {}
        for vote in votes_by_candidate.data:
            candidate_id = vote["candidate_id"]
            if candidate_id not in candidate_votes:
                candidate_votes[candidate_id] = 0
            candidate_votes[candidate_id] += 1
        
        return {
            "total_votes": total_response.count or 0,
            "votes_by_candidate": candidate_votes
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener estadísticas: {str(e)}"
        )

@router.get("/counts")
async def get_vote_counts(supabase: Client = Depends(get_supabase)):
    """Obtener conteo de votos por candidato con información completa"""
    try:
        # Obtener todos los candidatos activos
        candidates_response = supabase.table("candidates")\
            .select("id, name, party")\
            .eq("is_active", True)\
            .execute()
        
        if not candidates_response.data:
            return {
                "success": True,
                "total_votes": 0,
                "candidates": []
            }
        
        # Obtener conteo de votos por cada candidato
        vote_counts = []
        total_votes = 0
        
        for candidate in candidates_response.data:
            candidate_id = candidate["id"]
            
            # Contar votos del candidato
            votes_response = supabase.table("votes")\
                .select("id", count="exact")\
                .eq("candidate_id", candidate_id)\
                .execute()
            
            vote_count = votes_response.count or 0
            total_votes += vote_count
            
            vote_counts.append({
                "candidate_id": candidate_id,
                "candidate_name": candidate["name"],
                "party": candidate["party"],
                "vote_count": vote_count
            })
        
        # Calcular porcentajes
        for candidate_data in vote_counts:
            if total_votes > 0:
                candidate_data["percentage"] = round((candidate_data["vote_count"] / total_votes) * 100, 2)
            else:
                candidate_data["percentage"] = 0.0
        
        # Ordenar por votos descendente
        vote_counts.sort(key=lambda x: x["vote_count"], reverse=True)
        
        return {
            "success": True,
            "total_votes": total_votes,
            "candidates": vote_counts,
            "last_updated": __import__('datetime').datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener conteo de votos: {str(e)}"
        )

