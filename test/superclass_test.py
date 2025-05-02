import os
import argparse
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the ButterflyTorch class
from services.pytorch_superclass import ButterflyTorch


def test_butterfly_torch(data_path, start_idx, end_idx, transform=True):
    """
    Test the ButterflyTorch functionality with user-specified start and end indices.

    Args:
        data_path: Path to the SQLite database file
        start_idx: Starting frame index (user input)
        end_idx: Ending frame index (user input)
        transform: Whether to apply transformation function

    Returns:
        bool: Success status
    """
    try:
        logger.info(f"Testing ButterflyTorch with data path: {data_path}")
        logger.info(f"Frame range: {start_idx} to {end_idx}")

        # Initialize ButterflyTorch (superclass of PyTorch)
        my_torch = ButterflyTorch()

        # Define a transform function if requested
        transform_fn = None
        if transform:
            def transform_fn(frames):
                # Normalize frames to [0,1] range
                return frames / 255.0

            logger.info("Using normalization transform function")

        # Load data with the provided path and transform function
        logger.info("Loading distributed video dataset...")
        load_data = my_torch.load_data(data_path, transform_fn=transform_fn)

        # Get total number of available frames
        total_frames = len(load_data)
        logger.info(f"Total frames in dataset: {total_frames}")

        # Validate indices
        if start_idx < 0 or end_idx >= total_frames or start_idx >= end_idx:
            logger.error(f"Invalid frame range: {start_idx}-{end_idx}. Valid range is 0-{total_frames - 1}")
            return False

        # Fetch frames using the user-specified range
        logger.info(f"Fetching frames from {start_idx} to {end_idx}...")
        fetched_from_data_storage = load_data[start_idx:end_idx]

        # Display information about the fetched data
        logger.info(f"Successfully fetched {fetched_from_data_storage.shape[0]} frames")

        # Show frame shape information if available
        if len(fetched_from_data_storage.shape) > 1:
            frame_shape = fetched_from_data_storage.shape[1:]
            logger.info(f"Frame shape: {frame_shape}")
            logger.info(f"Data type: {fetched_from_data_storage.dtype}")

            # Calculate and display some basic statistics
            if transform:
                logger.info(f"Mean pixel value: {fetched_from_data_storage.mean().item():.4f}")
                logger.info(f"Min pixel value: {fetched_from_data_storage.min().item():.4f}")
                logger.info(f"Max pixel value: {fetched_from_data_storage.max().item():.4f}")
            else:
                logger.info(f"Mean pixel value: {fetched_from_data_storage.mean().item()}")
                logger.info(f"Min pixel value: {fetched_from_data_storage.min().item()}")
                logger.info(f"Max pixel value: {fetched_from_data_storage.max().item()}")

        logger.info("ButterflyTorch test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error in test_butterfly_torch: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def main(data_path,
        start,
        end,):
    """
    Main function to parse command line arguments and run the test.
    """
    parser = argparse.ArgumentParser(description='Test ButterflyTorch functionality with user-specified frame range')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the SQLite database file')
    parser.add_argument('--start', type=int, required=True,
                        help='Starting frame index')
    parser.add_argument('--end', type=int, required=True,
                        help='Ending frame index')
    parser.add_argument('--no_transform', action='store_true',
                        help='Disable the transform function')

    args = parser.parse_args()

    # Execute the test with the provided arguments
    success = test_butterfly_torch(
        data_path,
        start,
        end,
        transform=not args.no_transform
    )

    if success:
        print("\nTest successful! Data was fetched correctly from distributed storage.")
    else:
        print("\nTest failed. Check the log for details.")


if __name__ == "__main__":
    start_index = input("Enter the starting frame index: ")
    end_index = input("Enter the ending frame index: ")
    data_path = input("Enter the path to the SQLite database file: ")
    main()