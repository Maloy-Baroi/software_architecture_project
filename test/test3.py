import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.pytorch_superclass import ButterflyTorch

# Initialize ButterflyTorch
my_torch = ButterflyTorch()

# Process videos and save to distributed storage
video_paths = [
    'D:/Masters/Software_Architecture/Project/SA_next/src/videos/Video1.mp4',
    'D:/Masters/Software_Architecture/Project/SA_next/src/videos/video2.mp4'
]
my_torch.save_data(video_paths)


# Define a transform function (similar to PyTorch transforms)
def normalize_frames(frames):
    # Normalize frames to [0, 1] range
    return frames / 255.0


# Load data
dataset = my_torch.load_data(transform_fn=normalize_frames)

# Access frames seamlessly - even if they span across chunks
frames = dataset[1000:2000]

# Use frames with PyTorch as usual
processed = my_torch.nn.functional.normalize(frames)
