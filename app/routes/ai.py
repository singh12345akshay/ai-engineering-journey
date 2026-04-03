from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models import UserQuery, APIResponse, AdvancedQuery, AdvancedResponse
from app.services.llm import call_llm, stream_tokens, ask_advanced

router = APIRouter(prefix="/ai", tags=["AI"])


@router.get("/health")
async def health_check():
    return {"status": "running", "version": "0.1.0"}


@router.post("/ask", response_model=APIResponse)
async def ask(query: UserQuery):
    try:
        result = await call_llm(query)
        return APIResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask/advanced", response_model=AdvancedResponse)
async def ask_advanced_endpoint(query: AdvancedQuery):
    try:
        result = await ask_advanced(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/stream")
async def ask_stream(query: UserQuery):
    return StreamingResponse(
        stream_tokens(query),
        media_type="text/event-stream"
    )
