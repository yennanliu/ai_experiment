"""FastAPI application entry point"""

from fastapi import FastAPI
from fastapi.responses import FileResponse

from routers.graph_router import router as graph_router
from routers.tickets import router as tickets_router

app = FastAPI(title="Ticket Processing System")
app.include_router(tickets_router)
app.include_router(graph_router)


@app.get("/")
async def serve_ui():
    return FileResponse("ui/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
