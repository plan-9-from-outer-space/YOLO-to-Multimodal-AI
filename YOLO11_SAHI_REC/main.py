
# Import all the required libraries
import cv2
import time

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
# from sahi.utils.ultralytics import download_model_weights
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
# from ultralytics.utils.torch_utils import select_device

class SAHIInference:

    def __init__(self):
        self.detection_model = None

    # def load_model(self, model_weights, device):
    #     # self.detection_model = YOLO("yolo11n.pt")
    #     model_path = f"models/{model_weights}"
    #     # download_model_weights(weights_path)
    #     self.detection_model = AutoDetectionModel.from_pretrained(
    #         model_type="ultralytics",
    #         model_path=model_path,
    #         confidence_threshold=0.3,
    #         device=select_device(device)
    #     )
    
    def load_model(self, weights, device):
        weights_path = f"models/{weights}"
        # download_model_weights(weights_path)
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=weights_path,
            # confidence_threshold=0.35,
            device="cpu" # select_device(device)
        )

    def run(self, source, weights, device="cpu", view_img=True, save_img=True,
            slice_size=(512, 512), output_video_name="output.mp4"):

        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), f"Failed to open video: {source}"

        save_dir = increment_path("output_video", exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = cap.get(cv2.CAP_PROP_FPS)

        # Define output video writer
        video_writer = None
        if save_img:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(save_dir/output_video_name), fourcc, fps_input, (width, height))
        
        self.load_model(weights, device)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            start_time = time.time()

            results = get_sliced_prediction(
                image=frame[..., ::-1],
                detection_model = self.detection_model,
                slice_height = slice_size[1],
                slice_width = slice_size[0]
            )

            # Draw Bounding boxes
            for pred in results.object_prediction_list:
                x1, y1, x2, y2 = map(int, [pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy])
                class_name = pred.category.name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Calculate the FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps: .2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)

            # Display Output Video Frame
            if view_img:
                cv2.imshow("YOLO11 + SAHI Inference", frame)

            # Save Video Frame
            if save_img:
                video_writer.write(frame)

            # Exit on Pressing '1'
            if cv2.waitKey(1) & 0xFF == ord("1"):
                break
        
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

def main ():
    detector = SAHIInference()
    detector.run(
        source="input_video/input_video.mp4",
        weights = "yolo11n.pt",
        device = "cpu",
        view_img = True,
        save_img = True,
        slice_size=(512, 512),
        output_video_name="output.mp4"
    )

if __name__ == "__main__":
    main ()

