from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from aic_bot.routes import router


app = FastAPI()

origins = [
    "https://github.com",
    "http://github.com",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)



if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=5555)
