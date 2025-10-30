from supabase import create_client, Client
from config import settings

# Cliente de Supabase con la service key para operaciones del servidor
supabase: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_SERVICE_KEY
)

# Cliente con la anon key para operaciones regulares
supabase_anon: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_KEY
)

def get_supabase() -> Client:
    """Retorna el cliente de Supabase"""
    return supabase

def get_supabase_anon() -> Client:
    """Retorna el cliente de Supabase con anon key"""
    return supabase_anon

