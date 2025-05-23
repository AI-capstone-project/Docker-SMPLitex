from http.client import HTTPException

from fastapi import FastAPI

import subprocess

app = FastAPI()

@app.get("/{id:int}")
async def read_root(id: int):
    process = subprocess.Popen(["sh", "/home/myuser/SMPLitex/scripts/create-texture.sh", str(id)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print stdout line by line
    for stdout_line in iter(process.stdout.readline, ""):
        print(stdout_line, end="")

    # Print stderr line by line
    for stderr_line in iter(process.stderr.readline, ""):
        print(stderr_line, end="")

    process.stdout.close()
    process.stderr.close()
    return_code = process.wait()
    
    if return_code != 0:
        raise HTTPException(status_code=500, detail="3D try-on script failed")
    
    return {"message": "3D try-on script executed"}
    
@app.get("/pose/{id:int}/{pose_id:int}")
async def create_pose(id: int, pose_id: int):
    process = subprocess.Popen(["sh", "/home/myuser/SMPLitex/scripts/3d-render.sh", str(id), str(pose_id)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print stdout line by line
    for stdout_line in iter(process.stdout.readline, ""):
        print(stdout_line, end="")

    # Print stderr line by line
    for stderr_line in iter(process.stderr.readline, ""):
        print(stderr_line, end="")

    process.stdout.close()
    process.stderr.close()
    return_code = process.wait()
    
    if return_code != 0:
        raise HTTPException(status_code=500, detail="3D try-on script failed")
    
    return {"message": "3D try-on script executed"}