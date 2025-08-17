from fastapi import APIRouter

from .endpoints import batch, songs, stats

api_router = APIRouter()

# Include routers from endpoint modules
api_router.include_router(stats.router, tags=["stats"])
api_router.include_router(batch.router, prefix="/batch", tags=["batch"])
api_router.include_router(songs.router, prefix="/songs", tags=["songs"])
