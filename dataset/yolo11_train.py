from ultralytics import YOLO

def train_model():
    # Load a model
    model = YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")  # build from YAML and transfer weights

    # Train the model with augmentation
    results = model.train(
        data="data.yaml", 
        epochs=1, 
        imgsz=640,
        batch=8,
        workers=8,
        device=0,
        # Dataset settings
        split=0.01,          # Set validation set ratio to 1%
        seed=42,            # Set random seed for reproducibility
        # Data augmentation parameters
        mosaic=0.7,    # Reduce mosaic augmentation probability to 70%
        mixup=0.5,     # Mixup augmentation probability
        copy_paste=0.3,  # Copy-paste augmentation probability
        degrees=10.0,    # Maximum rotation angle
        translate=0.2,   # Translation range
        scale=0.5,      # Scale range
        shear=0.2,      # Shear range
        perspective=0.0, # Perspective transform range
        flipud=0.5,     # Vertical flip probability
        fliplr=0.5,     # Horizontal flip probability
        hsv_h=0.015,    # HSV-Hue augmentation range
        hsv_s=0.7,      # HSV-Saturation augmentation range
        hsv_v=0.4,      # HSV-Value augmentation range
        # Early stopping strategy
        patience=15,     # Stop if no improvement for 15 epochs
        save_period=5,   # Save model every 5 epochs
        exist_ok=True,   # Allow overwriting existing experiment folder
        # Learning rate strategy
        lr0=0.01,          # Initial learning rate
        lrf=0.01,          # Final learning rate factor
        warmup_epochs=3,    # Number of warmup epochs
        warmup_momentum=0.8,# Warmup momentum
        warmup_bias_lr=0.1, # Warmup bias learning rate
        # Optimizer settings
        optimizer='AdamW',  # Use AdamW optimizer
        weight_decay=0.05,  # Weight decay
        momentum=0.937,     # Momentum
        # Loss function weights
        box=7.5,           # Box loss weight
        cls=0.5,           # Classification loss weight
        dfl=1.5,           # DFL loss weight
        # Other optimizations
        close_mosaic=10,   # Disable mosaic in last 10 epochs
        nbs=64,            # Nominal batch size
        overlap_mask=True,  # Use overlapping masks
        mask_ratio=4,      # Mask downsample ratio
        single_cls=False,  # Multi-class detection mode
    )

if __name__ == '__main__':
    train_model()
