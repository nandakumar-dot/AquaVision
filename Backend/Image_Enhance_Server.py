import time
import grpc
from concurrent import futures
import cv2
import os
import torch
from ultralytics import YOLO
from media_processing_pb2 import MediaResponse
import media_processing_pb2_grpc
from google.protobuf import empty_pb2
from werkzeug.utils import secure_filename
import numpy as np
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


detection_model = YOLO('Model/best.pt')
detection_model.to(device)

preprocess, postprocess, enhancement_model = torch.hub.load('tnwei/waternet', 'waternet')
enhancement_model.to(device)  
enhancement_model.eval()

if not os.path.exists('uploads'):
    os.makedirs('uploads')
if not os.path.exists('processed'):
    os.makedirs('processed')

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

def enhance_image(input_image):
    rgb_im = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    rgb_im = cv2.resize(rgb_im, (720, 480))  

    rgb_ten, wb_ten, he_ten, gc_ten = preprocess(rgb_im)
    rgb_ten = rgb_ten.to(device)
    wb_ten = wb_ten.to(device)
    he_ten = he_ten.to(device)
    gc_ten = gc_ten.to(device)

    with torch.no_grad():
        enhanced_ten = enhancement_model(rgb_ten, wb_ten, he_ten, gc_ten)
        enhanced_ten = enhanced_ten.cpu()  
    
    enhanced_image = postprocess(enhanced_ten)
    return cv2.cvtColor(enhanced_image[0], cv2.COLOR_RGB2BGR)

class MediaProcessorServicer(media_processing_pb2_grpc.MediaProcessorServicer):
    def ProcessMedia(self, request, context):
        start_request_time = time.time()
        
        filename = secure_filename(request.filename)
        input_path = os.path.join('uploads', filename)
        output_path = os.path.join('processed', f"processed_{filename}")
        
        with open(input_path, 'wb') as f:
            f.write(request.file)

        processing_start_time = time.time()
        
        if is_image_file(filename):
            image = cv2.imread(input_path)
            enhanced_image = enhance_image(image)  
            
            results = detection_model(enhanced_image, device=device)
            
            for result in results:
                processed_image = result.plot()
            
            cv2.imwrite(output_path, processed_image)
            
            with open(output_path, 'rb') as f:
                processed_file = f.read()

            mimetype = 'image/jpeg'

        else:
            logging.basicConfig(level=logging.INFO)
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'xvid')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            target_fps = 15
            frame_interval = int(fps / target_fps) if fps > target_fps else 1

            out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

            logging.info("Starting GPU-accelerated video processing with reduced FPS.")

            frame_count = 0  

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logging.error("End of video or read error.")
                    break

                if frame_count % frame_interval == 0:
                    enhanced_frame = enhance_image(frame) 
                    
                   
                    results = detection_model(enhanced_frame, device=device)
                    for result in results:
                        rendered_frame = result.plot()
                        out.write(rendered_frame) 

                frame_count += 1

            logging.info("Video processing completed with reduced FPS.")
            cap.release()
            out.release()

            with open(output_path, 'rb') as f:
                processed_file = f.read()

            mimetype = 'video/mp4'
        
        processing_end_time = time.time()

        print(f"Request Time: {processing_start_time - start_request_time:.2f} seconds")
        print(f"Processing Time: {processing_end_time - processing_start_time:.2f} seconds")
        print(f"Total Time: {processing_end_time - start_request_time:.2f} seconds")

        return MediaResponse(
            processed_file=processed_file,
            mimetype=mimetype,
            message="File processed successfully"
        )
    
    def ProcessMediaStream(self, request_iterator, context):
        for request in request_iterator:
            filename = request.filename

            np_img = np.frombuffer(request.file, np.uint8)
            input_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            enhanced_image = enhance_image(input_image) 

            results = detection_model(enhanced_image, device=device)
            for result in results:
                processed_image = result.plot()

            _, encoded_image = cv2.imencode('.jpg', processed_image)
            processed_file = encoded_image.tobytes()

            yield MediaResponse(
                processed_file=processed_file,
                mimetype='image/jpeg',
                message="Frame processed successfully"
            )

def serve():
    options = [
        ('grpc.max_send_message_length', 1000 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1000 * 1024 * 1024)
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    media_processing_pb2_grpc.add_MediaProcessorServicer_to_server(MediaProcessorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
