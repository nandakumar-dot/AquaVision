from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import grpc
import os
import time
from pathlib import Path
from media_processing_pb2 import MediaRequest
import media_processing_pb2_grpc
import cv2
import asyncio
import grpc.aio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="../Frontend"), name="static")

Path("uploads").mkdir(exist_ok=True)
Path("processed").mkdir(exist_ok=True)

def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))


async def stream_frames(websocket: WebSocket):
    await websocket.accept()
    
    channel = grpc.aio.insecure_channel('localhost:50051')
    stub = media_processing_pb2_grpc.MediaProcessorStub(channel)

    async def generate_requests():
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            _, encoded_image = cv2.imencode('.jpg', frame)
            yield MediaRequest(file=encoded_image.tobytes(), filename="frame.jpg")
            await asyncio.sleep(1/15)  # Limit to 15 FPS

        cap.release()


    try:
        async for response in stub.ProcessMediaStream(generate_requests()):
            await websocket.send_bytes(response.processed_file)
    finally:
        await websocket.close()
        await channel.close()



@app.get("/")
async def serve_index():
    return FileResponse("../Frontend/index.html")

@app.post("/process-media")
async def process_media(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")


    request_start_time = time.time()

    filename = file.filename
    input_path = f"uploads/{filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())
    

    request_end_time = time.time()
    request_time = request_end_time - request_start_time

    inference_start_time = time.time()

    result = run_client(input_path)


    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time

    total_time = time.time() - request_start_time


    print(f"Request Time: {request_time:.2f} seconds")
    print(f"Inference Time: {inference_time:.2f} seconds")
    print(f"Total Time: {total_time:.2f} seconds")

    

    output_path = f"processed/processed_{filename}"
    if not os.path.exists(output_path):
        raise HTTPException(
            status_code=500,
            detail="Processing failed: Processed file not found"
        )

    return FileResponse(output_path, media_type="application/octet-stream", filename=f"processed_{filename}")


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await stream_frames(websocket)


def run_client(file_path):
    options = [('grpc.max_receive_message_length', 1000 * 1024 * 1024)]
    with grpc.insecure_channel('localhost:50051',options=options) as channel:
        stub = media_processing_pb2_grpc.MediaProcessorStub(channel)
        
        with open(file_path, "rb") as file:
            filename = file_path.split("/")[-1]
            request = MediaRequest(file=file.read(), filename=filename)
            response = stub.ProcessMedia(request)
            
            output_path = f"processed/processed_{filename}"
            with open(output_path, "wb") as output_file:
                output_file.write(response.processed_file)
            print("Processed file saved with MIME type:", response.mimetype)
            print("Message:", response.message)

            
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
