import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Now try to import ButterflyTorch
try:
    from services.pytorch_superclass import ButterflyTorch
except ModuleNotFoundError:
    print("Trying alternate import path...")
    # If that doesn't work, try to determine the correct path
    print(f"Project root: {project_root}")
    print("Available directories:")
    for item in os.listdir(project_root):
        if os.path.isdir(os.path.join(project_root, item)):
            print(f"- {item}")

    # Try another common structure
    try:
        from services.pytorch_superclass import ButterflyTorch
    except ModuleNotFoundError:
        print("Could not find ButterflyTorch class. Please provide the correct import path.")
        sys.exit(1)


def test_butterfly_torch_load():
    """
    Test function that demonstrates basic usage of ButterflyTorch for loading data.
    """
    # Define the path to the database file
    # Replace this with the actual path to your database
    db_path = os.path.join(project_root, "", "video_database.db")

    # Check if the database file exists
    if not os.path.isfile(db_path):
        print(f"Error: Database file not found at {db_path}")
        print("Please ensure you have processed videos before running this test.")
        return False

    try:
        # Initialize ButterflyTorch
        print("Initializing ButterflyTorch...")
        my_torch = ButterflyTorch()

        # Define a simple transform function that normalizes frames
        def normalize_frames(frames):
            """Normalize pixel values to range [0, 1]"""
            return frames / 255.0

        # Load data
        print(f"Loading data from {db_path}...")
        load_data = my_torch.load_data(db_path, transform_fn=normalize_frames)

        # Get the total number of frames
        total_frames = len(load_data)
        print(f"Total frames available: {total_frames}")

        if total_frames == 0:
            print("No frames found in the database.")
            return False

        # Define the range of frames to fetch
        start_frame = 0
        end_frame = min(10, total_frames)  # Get first 10 frames or all if less than 10

        print(f"Fetching frames from {start_frame} to {end_frame}...")

        # Fetch the specified range of frames
        fetched_from_data_storage = load_data[start_frame:end_frame]

        # Display information about the fetched data
        print(f"Successfully fetched {fetched_from_data_storage.shape[0]} frames")
        print(
            f"Frame shape: {fetched_from_data_storage.shape[1:] if len(fetched_from_data_storage.shape) > 1 else 'N/A'}")

        # Visualize the first frame if possible
        if fetched_from_data_storage.shape[0] > 0:
            frame = fetched_from_data_storage[0]

            # Check if the frame has valid dimensions for display
            if len(frame.shape) == 3 and frame.shape[2] >= 3:
                # If normalized, scale back to [0, 255] for display
                if frame.max() <= 1.0:
                    frame = frame * 255

                print("Visualizing the first frame...")
                plt.figure(figsize=(8, 6))
                plt.imshow(frame[:, :, :3].numpy().astype(np.uint8))
                plt.title(f"Frame {start_frame}")
                plt.axis('off')
                plt.savefig(os.path.join(project_root, "test_frame.png"))
                plt.close()
                print("Frame visualization saved as 'test_frame.png'")

        print("Test completed successfully!")
        return True

    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_butterfly_torch_load()