import os
import pickle
import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Tuple, List, Dict, Any, Optional
import logging
import traceback

# Configure logging to show detailed information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChunkTracker:
    """SQLite database to track video chunks and their locations in distributed storage."""

    def __init__(self, db_path: str):
        """
        Initialize the chunk tracker database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
        logger.info(f"Initialized ChunkTracker with database at {db_path}")

    def _init_db(self):
        """Create database tables if they don't exist."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Table for movies
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS movies (
                movie_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                duration INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL
            )
            ''')

            # Table for chunks
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY,
                movie_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                storage_path TEXT NOT NULL,
                FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
            )
            ''')

            # Index for efficient range queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_frames ON chunks (start_frame, end_frame)')

            conn.commit()
            conn.close()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def add_movie(self, title: str, duration: int, total_chunks: int) -> int:
        """
        Add a new movie to the database.

        Args:
            title: Movie title
            duration: Duration in seconds
            total_chunks: Number of chunks the movie is split into

        Returns:
            movie_id: ID of the added movie
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO movies (title, duration, total_chunks) VALUES (?, ?, ?)",
                (title, duration, total_chunks)
            )
            movie_id = cursor.lastrowid

            conn.commit()
            conn.close()
            logger.info(f"Added movie '{title}' with ID {movie_id}")

            return movie_id
        except Exception as e:
            logger.error(f"Error adding movie: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def add_chunk(self, movie_id: int, chunk_index: int,
                  start_frame: int, end_frame: int, storage_path: str) -> int:
        """
        Add a new chunk to the database.

        Args:
            movie_id: ID of the movie
            chunk_index: Index of this chunk within the movie
            start_frame: Starting frame number in the overall dataset
            end_frame: Ending frame number in the overall dataset
            storage_path: Path where the chunk is stored

        Returns:
            chunk_id: ID of the added chunk
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO chunks (movie_id, chunk_index, start_frame, end_frame, storage_path) "
                "VALUES (?, ?, ?, ?, ?)",
                (movie_id, chunk_index, start_frame, end_frame, storage_path)
            )
            chunk_id = cursor.lastrowid

            conn.commit()
            conn.close()
            logger.info(f"Added chunk {chunk_index} for movie {movie_id} at {storage_path}")

            return chunk_id
        except Exception as e:
            logger.error(f"Error adding chunk: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_chunks_for_frame_range(self, start_frame: int, end_frame: int) -> List[Dict[str, Any]]:
        """
        Get all chunks that overlap with the given frame range.

        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number

        Returns:
            List of dicts containing chunk information
        """
        try:
            logger.debug(f"Searching for chunks in range {start_frame}-{end_frame}")

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # This enables column access by name
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM chunks WHERE "
                "(start_frame <= ? AND end_frame >= ?) OR "  # Chunk encompasses entire range
                "(start_frame >= ? AND start_frame <= ?) OR "  # Start of chunk is in range
                "(end_frame >= ? AND end_frame <= ?)",  # End of chunk is in range
                (start_frame, end_frame, start_frame, end_frame, start_frame, end_frame)
            )

            chunks = [dict(row) for row in cursor.fetchall()]
            conn.close()

            if not chunks:
                logger.warning(f"No chunks found for frame range {start_frame}-{end_frame}")
            else:
                logger.debug(f"Found {len(chunks)} chunks for frame range {start_frame}-{end_frame}")

            return chunks
        except Exception as e:
            logger.error(f"Error getting chunks for frame range {start_frame}-{end_frame}: {str(e)}")
            logger.error(traceback.format_exc())
            raise


class VideoProcessor:
    """Process video files into pickle chunks for distributed storage."""

    def __init__(self, chunk_tracker: ChunkTracker, chunk_size: int = 1000):
        """
        Initialize the video processor.

        Args:
            chunk_tracker: ChunkTracker instance to track chunks
            chunk_size: Number of frames per chunk
        """
        self.chunk_tracker = chunk_tracker
        self.chunk_size = chunk_size
        logger.info(f"Initialized VideoProcessor with chunk size {chunk_size}")

    def process_video(self, video_path: str, storage_base_path: str, movie_title: str = None) -> int:
        """
        Process a video file into chunks and store them.

        Args:
            video_path: Path to video file
            storage_base_path: Base path for distributed storage
            movie_title: Title of the movie (defaults to filename)

        Returns:
            movie_id: ID of the processed movie
        """
        try:
            # Check if OpenCV is available
            try:
                import cv2
            except ImportError:
                logger.error(
                    "OpenCV (cv2) is required but not installed. Please install it with: pip install opencv-python")
                raise ImportError("OpenCV (cv2) is required but not installed")

            if movie_title is None:
                movie_title = os.path.basename(video_path)

            logger.info(f"Processing video: {video_path} as '{movie_title}'")

            # Check if video file exists
            if not os.path.isfile(video_path):
                logger.error(f"Video file not found: {video_path}")
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Open video file
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                raise IOError(f"Failed to open video file: {video_path}")

            # Get video properties
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0

            logger.info(f"Video properties: {frame_count} frames, {fps} fps, {duration:.2f} seconds")

            # Calculate total chunks
            total_chunks = (frame_count + self.chunk_size - 1) // self.chunk_size

            # Add movie to database
            movie_id = self.chunk_tracker.add_movie(movie_title, int(duration), total_chunks)

            # Create directory for this movie
            movie_dir = os.path.join(storage_base_path, f"movie_{movie_id}")
            # Normalize path
            movie_dir = os.path.normpath(movie_dir)
            os.makedirs(movie_dir, exist_ok=True)
            logger.info(f"Created directory for chunks: {movie_dir}")

            # Process frames in chunks
            current_chunk = []
            chunk_index = 0
            frame_index = 0
            chunk_start_frame = 0

            while True:
                ret, frame = video.read()
                if not ret:
                    break

                # Convert frame to numpy array and add to current chunk
                frame_array = np.array(frame)
                current_chunk.append(frame_array)

                # If chunk is full or this is the last frame, save it
                if len(current_chunk) >= self.chunk_size or frame_index == frame_count - 1:
                    chunk_end_frame = chunk_start_frame + len(current_chunk) - 1

                    # Save chunk as pickle
                    chunk_path = os.path.join(movie_dir, f"chunk_{chunk_index}.pkl")
                    # Normalize path
                    chunk_path = os.path.normpath(chunk_path)

                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(chunk_path), exist_ok=True)

                    # Convert to numpy array and save
                    chunk_data = np.array(current_chunk)
                    with open(chunk_path, "wb") as f:
                        pickle.dump(chunk_data, f)

                    # Check if the file was actually created
                    if not os.path.isfile(chunk_path):
                        logger.error(f"Failed to create chunk file: {chunk_path}")
                        raise IOError(f"Failed to create chunk file: {chunk_path}")

                    # Register chunk in database - use the absolute path to avoid issues
                    abs_chunk_path = os.path.abspath(chunk_path)
                    self.chunk_tracker.add_chunk(
                        movie_id, chunk_index, chunk_start_frame, chunk_end_frame, abs_chunk_path
                    )

                    logger.info(
                        f"Saved chunk {chunk_index}: frames {chunk_start_frame}-{chunk_end_frame} to {abs_chunk_path}")

                    # Reset for next chunk
                    current_chunk = []
                    chunk_index += 1
                    chunk_start_frame = chunk_end_frame + 1

                frame_index += 1

                # Print progress every 1000 frames
                if frame_index % 1000 == 0:
                    logger.info(
                        f"Processed {frame_index}/{frame_count} frames ({frame_index / frame_count * 100:.1f}%)")

            video.release()
            logger.info(f"Completed processing video {movie_title}: {chunk_index} chunks created")

            return movie_id

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

class DistributedVideoDataset(Dataset):
    """PyTorch Dataset for accessing video data from distributed storage."""

    def __init__(self, db_path: str, transform: Optional[Callable] = None):
        """
        Initialize the dataset.

        Args:
            db_path: Path to SQLite database file
            transform: Optional transform to apply to frames
        """
        self.chunk_tracker = ChunkTracker(db_path)
        self.transform = transform
        self._cache = {}  # Simple in-memory cache for chunks
        self._cache_limit = 5  # Maximum number of chunks to keep in cache
        logger.info(f"Initialized DistributedVideoDataset with database at {db_path}")

    def _load_chunk(self, chunk_path: str) -> np.ndarray:
        """
        Load a chunk from storage.

        Args:
            chunk_path: Path to pickle file

        Returns:
            Numpy array of frames
        """
        try:
            # Check if chunk is in cache
            if chunk_path in self._cache:
                logger.debug(f"Chunk found in cache: {chunk_path}")
                return self._cache[chunk_path]

            # Normalize path (convert to OS-specific format)
            normalized_path = os.path.normpath(chunk_path)

            # Check if file exists
            if not os.path.isfile(normalized_path):
                logger.error(f"Chunk file not found: {normalized_path}")
                logger.error(f"Original path was: {chunk_path}")

                # Try to check if the directory exists
                dir_path = os.path.dirname(normalized_path)
                if not os.path.exists(dir_path):
                    logger.error(f"Directory does not exist: {dir_path}")
                else:
                    # List files in directory to help debug
                    logger.info(f"Files in directory {dir_path}:")
                    for f in os.listdir(dir_path):
                        logger.info(f"  {f}")

                raise FileNotFoundError(f"Chunk file not found: {normalized_path}")

            # Load chunk from storage
            logger.debug(f"Loading chunk from disk: {normalized_path}")
            with open(normalized_path, "rb") as f:
                chunk_data = pickle.load(f)

            # Update cache
            if len(self._cache) >= self._cache_limit:
                # Remove oldest item
                oldest_key = next(iter(self._cache))
                logger.debug(f"Cache full, removing oldest item: {oldest_key}")
                self._cache.pop(oldest_key)

            self._cache[chunk_path] = chunk_data

            return chunk_data
        except Exception as e:
            logger.error(f"Error loading chunk {chunk_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _get_frames(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Get frames from the specified range.

        Args:
            start_idx: Starting frame index
            end_idx: Ending frame index

        Returns:
            Numpy array of frames
        """
        try:
            logger.debug(f"Getting frames from range {start_idx}-{end_idx}")

            # Get chunks that contain the requested frames
            chunks_info = self.chunk_tracker.get_chunks_for_frame_range(start_idx, end_idx)

            if not chunks_info:
                logger.error(f"No chunks found for frame range {start_idx}-{end_idx}")
                raise ValueError(f"No chunks found for frame range {start_idx}-{end_idx}")

            # Sort chunks by start_frame
            chunks_info.sort(key=lambda x: x["start_frame"])

            # Load all required chunks and extract relevant frames
            all_frames = []

            for chunk_info in chunks_info:
                chunk_data = self._load_chunk(chunk_info["storage_path"])

                # Calculate relative positions within this chunk
                chunk_start = chunk_info["start_frame"]
                chunk_end = chunk_info["end_frame"]

                # Calculate overlap between requested range and this chunk
                overlap_start = max(start_idx, chunk_start)
                overlap_end = min(end_idx, chunk_end)

                # Get the relevant frames from this chunk
                chunk_rel_start = overlap_start - chunk_start
                chunk_rel_end = overlap_end - chunk_start + 1  # +1 because slice is exclusive

                relevant_frames = chunk_data[chunk_rel_start:chunk_rel_end]
                all_frames.append(relevant_frames)

            # Concatenate all frames
            logger.debug(f"Returning {sum(len(f) for f in all_frames)} frames from {len(chunks_info)} chunks")
            return np.concatenate(all_frames)
        except Exception as e:
            logger.error(f"Error getting frames {start_idx}-{end_idx}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def __getitem__(self, idx):
        """
        Get item(s) at the specified index/indices.

        Args:
            idx: Index or slice

        Returns:
            Tensor or sequence of tensors
        """
        try:
            if isinstance(idx, slice):
                # Handle slice
                start = 0 if idx.start is None else idx.start
                stop = len(self) if idx.stop is None else idx.stop
                step = 1 if idx.step is None else idx.step

                logger.debug(f"Handling slice: {start}:{stop}:{step}")

                if step != 1:
                    # For non-contiguous slices, we need to handle each frame individually
                    frames = []
                    for i in range(start, stop, step):
                        frames.append(self[i])
                    return torch.stack(frames)

                # Get frames from the range
                frames = self._get_frames(start, stop - 1)  # -1 because stop is exclusive

                # Apply transform if specified
                if self.transform:
                    frames = self.transform(frames)

                # Convert to tensor
                return torch.from_numpy(frames)
            else:
                # Handle single index
                frames = self._get_frames(idx, idx)

                # Apply transform if specified
                if self.transform:
                    frames = self.transform(frames[0])
                    return frames

                # Convert to tensor
                return torch.from_numpy(frames[0])
        except Exception as e:
            logger.error(f"Error in __getitem__ with idx {idx}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def __len__(self):
        """
        Get the total number of frames.

        Returns:
            int: Total number of frames
        """
        try:
            # Query the database to get the maximum end_frame value
            conn = sqlite3.connect(self.chunk_tracker.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT MAX(end_frame) FROM chunks")
            max_frame = cursor.fetchone()[0]

            conn.close()

            return max_frame + 1 if max_frame is not None else 0
        except Exception as e:
            logger.error(f"Error in __len__: {str(e)}")
            logger.error(traceback.format_exc())
            raise


class ButterflyTorch(torch.nn.Module):
    """
    Superclass of PyTorch that extends its functionality to work with distributed data storage.
    """

    def __init__(self):
        """Initialize ButterflyTorch."""
        super().__init__()
        self.datasets = {}
        logger.info("Initialized ButterflyTorch")

    def load_data(self, data_path: str, shape=None, transform_fn: Optional[Callable] = None) -> DistributedVideoDataset:
        """
        Load data from distributed storage.

        Args:
            data_path: Path to SQLite database file
            shape: Optional shape information (not used currently)
            transform_fn: Optional transformation function

        Returns:
            DistributedVideoDataset: Dataset that can be indexed like a tensor
        """
        try:
            # Check if already loaded
            if data_path in self.datasets:
                logger.info(f"Dataset already loaded: {data_path}")
                return self.datasets[data_path]

            # Create and store dataset
            logger.info(f"Loading dataset from {data_path}")
            dataset = DistributedVideoDataset(data_path, transform_fn)
            self.datasets[data_path] = dataset

            return dataset
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def save_data(self, video_paths: List[str], storage_base_path: str, db_path: str, chunk_size: int = 1000):
        """
        Process videos and save them to distributed storage.

        Args:
            video_paths: List of paths to video files
            storage_base_path: Base path for distributed storage
            db_path: Path to SQLite database file
            chunk_size: Number of frames per chunk
        """
        try:
            # Create storage directory if it doesn't exist
            os.makedirs(storage_base_path, exist_ok=True)
            logger.info(f"Created storage directory: {storage_base_path}")

            # Check if video paths exist
            for video_path in video_paths:
                if not os.path.isfile(video_path):
                    logger.error(f"Video file not found: {video_path}")
                    raise FileNotFoundError(f"Video file not found: {video_path}")

            # Initialize chunk tracker and video processor
            logger.info(f"Initializing ChunkTracker with database at {db_path}")
            chunk_tracker = ChunkTracker(db_path)

            logger.info(f"Initializing VideoProcessor with chunk size {chunk_size}")
            processor = VideoProcessor(chunk_tracker, chunk_size)

            # Process each video
            for video_path in video_paths:
                logger.info(f"Processing video: {video_path}")
                processor.process_video(video_path, storage_base_path)

            logger.info(f"Processed {len(video_paths)} videos and saved to {storage_base_path}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    # Forward all PyTorch methods that aren't overridden
    def __getattr__(self, name):
        """Forward attribute access to PyTorch."""
        return getattr(torch, name)


def main():
    """
    Main function to run the code with your directory structure.
    """
    try:
        # Get current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Move up one level to get to the src directory
        src_dir = os.path.dirname(script_dir)

        # Define paths based on the directory structure shown in the image
        videos_dir = os.path.join(script_dir, 'D:/Masters/Software_Architecture/Project/SA_next/src/videos')
        storage_dir = os.path.join(script_dir, 'D:/Masters/Software_Architecture/Project/SA_next/src/storage')
        db_path = os.path.join(storage_dir, 'D:/Masters/Software_Architecture/Project/SA_next/video_database.db')

        # Normalize all paths
        videos_dir = os.path.normpath(videos_dir)
        storage_dir = os.path.normpath(storage_dir)
        db_path = os.path.normpath(db_path)

        logger.info(f"Script directory: {script_dir}")
        logger.info(f"Source directory: {src_dir}")
        logger.info(f"Videos directory: {videos_dir}")
        logger.info(f"Storage directory: {storage_dir}")
        logger.info(f"Database path: {db_path}")

        # Ensure directories exist
        os.makedirs(storage_dir, exist_ok=True)

        # Check if videos directory exists
        if not os.path.isdir(videos_dir):
            logger.error(f"Videos directory not found: {videos_dir}")
            return

        # Get all video files
        video_paths = [os.path.join(videos_dir, f) for f in os.listdir(videos_dir)
                       if f.lower().endswith(('.mp4', '.avi', '.mov'))]

        # Normalize all video paths
        video_paths = [os.path.normpath(p) for p in video_paths]

        if not video_paths:
            logger.error(f"No video files found in {videos_dir}")
            return

        logger.info(f"Found {len(video_paths)} video files: {[os.path.basename(p) for p in video_paths]}")

        # Initialize ButterflyTorch
        my_torch = ButterflyTorch()

        # Process videos and save to distributed storage
        logger.info("Starting video processing...")
        my_torch.save_data(video_paths, storage_dir, db_path, chunk_size=100)  # Using smaller chunk size for testing

        # Define a transform function (similar to PyTorch transforms)
        def transform_fn(frames):
            # Normalize frames
            return frames / 255.0

        # Load data
        logger.info("Loading processed data...")
        load_data = my_torch.load_data(db_path, transform_fn=transform_fn)

        # Get the total number of frames
        total_frames = len(load_data)
        logger.info(f"Total frames available: {total_frames}")

        if total_frames > 0:
            # Define a range to fetch (adjust as needed based on actual data)
            start_idx = min(10, total_frames // 2)
            end_idx = min(start_idx + 20, total_frames - 1)

            # Access data by range (seamlessly across chunks)
            logger.info(f"Fetching frames {start_idx}:{end_idx}...")
            fetched_data = load_data[start_idx:end_idx]

            logger.info(f"Successfully fetched {fetched_data.shape[0]} frames")
            logger.info(f"Frame shape: {fetched_data.shape[1:] if len(fetched_data.shape) > 1 else 'N/A'}")

            # Check storage directory for files
            storage_files = []
            for root, dirs, files in os.walk(storage_dir):
                for file in files:
                    if file.endswith('.pkl'):
                        storage_files.append(os.path.join(root, file))

            logger.info(f"Storage directory contains {len(storage_files)} pickle files")

        logger.info("Operation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("Successfully processed videos and stored chunks in the storage directory.")
    else:
        print("Error occurred during processing. Check logs for details.")