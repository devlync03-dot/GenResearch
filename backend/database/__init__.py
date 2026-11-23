"""Database clients shared by backend services."""

from .supabase_client import get_supabase

__all__ = ["get_supabase"]
