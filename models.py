from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# Auth Models
class UserSignUp(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserSignIn(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    is_admin: bool = False

class AuthResponse(BaseModel):
    user: UserResponse
    access_token: str
    token_type: str = "bearer"

# Candidate Models
class CandidateBase(BaseModel):
    name: str
    party: str
    description: Optional[str] = None
    image_url: Optional[str] = None
    proposals: Optional[List[str]] = None

class CandidateCreate(CandidateBase):
    pass

class CandidateResponse(CandidateBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Vote Models
class VoteCreate(BaseModel):
    candidate_id: str
    full_name: str
    dni: str
    phone: str
    department: str
    province: str
    district: str
    address: str

class VoteResponse(BaseModel):
    id: str
    candidate_id: str
    user_id: Optional[str] = None
    full_name: Optional[str] = None
    dni: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    province: Optional[str] = None
    district: Optional[str] = None
    address: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class VoteCheck(BaseModel):
    has_voted: bool
    candidate_id: Optional[str] = None

# Data Processing Models
class CSVUploadResponse(BaseModel):
    headers: List[str]
    rows: List[List]
    row_count: int
    filename: str

class CleaningOptions(BaseModel):
    handleNulls: bool = False
    normalizeData: bool = False
    encodeCategories: bool = False
    removeDuplicates: bool = False

class ModelConfig(BaseModel):
    modelType: str
    isProcessing: bool = False
    isComplete: bool = False

class ModelProcessResponse(BaseModel):
    success: bool
    message: str
    accuracy: Optional[float] = None
    model_type: str

# Electoral Data Models
class ElectoralDataStats(BaseModel):
    total_records: int
    has_data: bool
    election_years: List[int]
    departments: List[str]
    total_votes: int
    message: str

