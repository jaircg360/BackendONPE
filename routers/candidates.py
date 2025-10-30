from fastapi import APIRouter, HTTPException, Depends, status
from models import CandidateCreate, CandidateResponse
from database import get_supabase
from supabase import Client
from typing import List

router = APIRouter(prefix="/candidates", tags=["Candidates"])

@router.get("", response_model=List[CandidateResponse])
async def get_candidates(supabase: Client = Depends(get_supabase)):
    """Obtener todos los candidatos"""
    try:
        response = supabase.table("candidates")\
            .select("*")\
            .order("name")\
            .execute()
        
        return response.data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener candidatos: {str(e)}"
        )

@router.get("/{candidate_id}", response_model=CandidateResponse)
async def get_candidate(candidate_id: str, supabase: Client = Depends(get_supabase)):
    """Obtener un candidato por ID"""
    try:
        response = supabase.table("candidates")\
            .select("*")\
            .eq("id", candidate_id)\
            .single()\
            .execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Candidato no encontrado"
            )
        
        return response.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener candidato: {str(e)}"
        )

@router.post("", response_model=CandidateResponse, status_code=status.HTTP_201_CREATED)
async def create_candidate(
    candidate: CandidateCreate,
    supabase: Client = Depends(get_supabase)
):
    """Crear un nuevo candidato"""
    try:
        candidate_dict = candidate.model_dump()
        
        response = supabase.table("candidates")\
            .insert(candidate_dict)\
            .execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se pudo crear el candidato"
            )
        
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al crear candidato: {str(e)}"
        )

@router.get("/{candidate_id}/votes/count")
async def get_candidate_vote_count(
    candidate_id: str,
    supabase: Client = Depends(get_supabase)
):
    """Obtener el conteo de votos de un candidato"""
    try:
        response = supabase.table("votes")\
            .select("id", count="exact")\
            .eq("candidate_id", candidate_id)\
            .execute()
        
        return {
            "candidate_id": candidate_id,
            "vote_count": response.count or 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener conteo de votos: {str(e)}"
        )

