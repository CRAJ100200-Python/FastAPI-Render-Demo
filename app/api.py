from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from Render (no Docker)!"}


@app.get("/health")
def health():
    return {"status": "OK"}



@app.head("/")
def head():
    return {"status": "OK"}