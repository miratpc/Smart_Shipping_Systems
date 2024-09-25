from ultralytics import YOLO


paths = [r'runs\detect\2Class_GroundUp2\weights\best.pt']


if __name__ == '__main__':
    # Initialize model
    for path in paths:
        model = YOLO(path, task="detect")
        results = model.train(data='Boat_Detection/MultiClass_Dataset/dataset.yaml', epochs=300, batch=64, name="2Class_GroundUp", device="cuda", shear = 5, perspective= 0.0005, hsv_h =0.15 , lr0 = 0.0001, lrf = 0.001)