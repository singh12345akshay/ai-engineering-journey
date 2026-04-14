from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.ai import router as ai_router
from app.routes.search import router as search_router
from app.routes.langchain_routes import router as langchain_router
from app.config import settings
from app.routes.agent_route import router as agent_router 

app = FastAPI(
    title="AI Engineer Journey API",
    description="Built while learning AI engineering",
    version=settings.APP_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


app.include_router(ai_router)
app.include_router(search_router)
app.include_router(langchain_router)  
app.include_router(agent_router)