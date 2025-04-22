from ultralytics import YOLO
 
model = YOLO("yolo11l.pt")
results = model("test_img4.jpg")

for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb

    result.save(filename="result4.jpg")