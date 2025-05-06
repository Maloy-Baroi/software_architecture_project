import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Tuple, List, Dict, Any, Optional, Union
import logging
import traceback
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson.objectid import ObjectId
from dotenv import load_dotenv
import paramiko
import io
import concurrent.futures
import re
import time
import hashlib
import json

# Configure logging to show detailed information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RemoteFileHandler:
    """Handler for accessing files on remote servers via SSH."""

    def __init__(self, host=None, port=None, username=None, password=None, key_path=None):
        """
        Initialize the remote file handler.

        Args:
            host: Remote host (defaults to env variable)
            port: SSH port (defaults to env variable)
            username: SSH username (defaults to env variable)
            password: SSH password (defaults to env variable)
            key_path: Path to SSH key file (defaults to env variable)
        """
        # Load environment variables
        load_dotenv()

        # Use provided values or fall back to environment variables
        self.host = host or os.getenv("REMOTE_HOST")
        self.port = int(port or os.getenv("REMOTE_PORT", "22"))
        self.username = username or os.getenv("REMOTE_USER")
        self.password = password or os.getenv("REMOTE_PASSWORD")
        self.key_path = key_path or os.getenv("REMOTE_KEY_PATH")

        # Connection cache
        self.ssh_client = None
        self.sftp_client = None

        # Cache for file existence checks
        self._file_exists_cache = {}

        # Cache for directory listings
        self._dir_listing_cache = {}

        # Connection status
        self.is_remote = bool(self.host and self.username and (self.password or self.key_path))

        logger.info(f"Initialized RemoteFileHandler, remote mode: {self.is_remote}")

    def connect(self):
        """Establish connection to the remote server."""
        if not self.is_remote:
            logger.warning("Remote connection details not provided, operating in local mode")
            return

        if self.ssh_client is not None and self.ssh_client.get_transport() and self.ssh_client.get_transport().is_active():
            logger.debug("Using existing SSH connection")
            return

        try:
            logger.info(f"Connecting to {self.host}:{self.port} as {self.username}")
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            if self.key_path and os.path.exists(self.key_path):
                logger.info(f"Using SSH key: {self.key_path}")
                key = paramiko.RSAKey.from_private_key_file(self.key_path)
                self.ssh_client.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    pkey=key
                )
            else:
                logger.info("Using password authentication")
                self.ssh_client.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password
                )

            self.sftp_client = self.ssh_client.open_sftp()
            logger.info("Successfully connected to remote server")
        except Exception as e:
            logger.error(f"Error connecting to remote server: {str(e)}")
            logger.error(traceback.format_exc())

            # Clean up
            if self.ssh_client:
                self.ssh_client.close()
                self.ssh_client = None
            if self.sftp_client:
                self.sftp_client.close()
                self.sftp_client = None

            raise

    def disconnect(self):
        """Close connection to the remote server."""
        if self.sftp_client:
            logger.debug("Closing SFTP connection")
            self.sftp_client.close()
            self.sftp_client = None

        if self.ssh_client:
            logger.debug("Closing SSH connection")
            self.ssh_client.close()
            self.ssh_client = None

        logger.info("Disconnected from remote server")

    def ensure_connection(self):
        """Ensure that a connection exists before performing operations."""
        if not self.is_remote:
            return

        try:
            if not self.ssh_client or not self.ssh_client.get_transport() or not self.ssh_client.get_transport().is_active():
                logger.debug("Connection not active, reconnecting")
                self.connect()
        except Exception as e:
            logger.error(f"Error ensuring connection: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def file_exists(self, path):
        """
        Check if a file exists on the remote server or locally.

        Args:
            path: Path to the file

        Returns:
            bool: True if the file exists, False otherwise
        """
        # Check cache first
        if path in self._file_exists_cache:
            return self._file_exists_cache[path]

        try:
            if not self.is_remote:
                # Local mode
                result = os.path.isfile(path)
                self._file_exists_cache[path] = result
                return result

            # Remote mode
            self.ensure_connection()

            try:
                # Using SFTP stat to check if file exists
                self.sftp_client.stat(path)
                self._file_exists_cache[path] = True
                return True
            except FileNotFoundError:
                self._file_exists_cache[path] = False
                return False

        except Exception as e:
            logger.error(f"Error checking if file exists ({path}): {str(e)}")
            # Don't cache errors
            return False

    def read_file(self, path):
        """
        Read a file from the remote server or locally.

        Args:
            path: Path to the file

        Returns:
            bytes: File content
        """
        try:
            if not self.is_remote:
                # Local mode
                with open(path, 'rb') as f:
                    return f.read()

            # Remote mode
            self.ensure_connection()

            # Check if file exists
            if not self.file_exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            # Read file using SFTP
            with self.sftp_client.open(path, 'rb') as f:
                data = f.read()
                return data

        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def write_file(self, path, data):
        """
        Write data to a file on the remote server or locally.

        Args:
            path: Path to the file
            data: Data to write

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_remote:
                # Local mode
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

                with open(path, 'wb') as f:
                    f.write(data)
                return True

            # Remote mode
            self.ensure_connection()

            # Ensure remote directory exists
            remote_dir = os.path.dirname(path)
            try:
                self.sftp_client.stat(remote_dir)
            except FileNotFoundError:
                # Create directory structure
                self._mkdir_p(remote_dir)

            # Write file using SFTP
            with self.sftp_client.open(path, 'wb') as f:
                f.write(data)

            # Update cache
            self._file_exists_cache[path] = True
            return True

        except Exception as e:
            logger.error(f"Error writing file {path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _mkdir_p(self, remote_dir):
        """
        Recursively create directories on the remote server (mkdir -p equivalent).

        Args:
            remote_dir: Remote directory path
        """
        if remote_dir == '/':
            return

        try:
            self.sftp_client.stat(remote_dir)
        except FileNotFoundError:
            # Create parent directory first
            self._mkdir_p(os.path.dirname(remote_dir))

            # Then create this directory
            try:
                logger.debug(f"Creating remote directory: {remote_dir}")
                self.sftp_client.mkdir(remote_dir)
            except Exception as e:
                logger.error(f"Error creating directory {remote_dir}: {str(e)}")
                raise

    def list_directory(self, path):
        """
        List files in a directory on the remote server or locally.

        Args:
            path: Directory path

        Returns:
            list: List of filenames
        """
        # Check cache first
        if path in self._dir_listing_cache:
            # Cache directory listings for 1 minute
            cache_time, listing = self._dir_listing_cache[path]
            if time.time() - cache_time < 60:
                return listing

        try:
            if not self.is_remote:
                # Local mode
                if not os.path.isdir(path):
                    raise NotADirectoryError(f"Not a directory: {path}")

                result = os.listdir(path)
                self._dir_listing_cache[path] = (time.time(), result)
                return result

            # Remote mode
            self.ensure_connection()

            # List files using SFTP
            result = self.sftp_client.listdir(path)
            self._dir_listing_cache[path] = (time.time(), result)
            return result

        except Exception as e:
            logger.error(f"Error listing directory {path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_file_info(self, path):
        """
        Get file information (size, modification time, etc.).

        Args:
            path: Path to the file

        Returns:
            dict: File information
        """
        try:
            if not self.is_remote:
                # Local mode
                stat_info = os.stat(path)
                return {
                    'size': stat_info.st_size,
                    'mtime': stat_info.st_mtime,
                    'atime': stat_info.st_atime,
                    'ctime': stat_info.st_ctime,
                    'mode': stat_info.st_mode
                }

            # Remote mode
            self.ensure_connection()

            # Get file information using SFTP
            stat_info = self.sftp_client.stat(path)
            return {
                'size': stat_info.st_size,
                'mtime': stat_info.st_mtime,
                'atime': stat_info.st_atime,
                'mode': stat_info.st_mode
            }

        except Exception as e:
            logger.error(f"Error getting file info for {path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def execute_command(self, command):
        """
        Execute a command on the remote server.

        Args:
            command: Command to execute

        Returns:
            tuple: (stdout, stderr)
        """
        if not self.is_remote:
            raise RuntimeError("Cannot execute commands in local mode")

        try:
            self.ensure_connection()

            logger.debug(f"Executing command: {command}")
            stdin, stdout, stderr = self.ssh_client.exec_command(command)

            # Read output
            stdout_str = stdout.read().decode('utf-8')
            stderr_str = stderr.read().decode('utf-8')

            if stderr_str:
                logger.warning(f"Command produced errors: {stderr_str}")

            return stdout_str, stderr_str

        except Exception as e:
            logger.error(f"Error executing command '{command}': {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def __del__(self):
        """Clean up resources when this object is garbage collected."""
        self.disconnect()


class ChunkTracker:
    """MongoDB database to track video chunks and their locations in distributed storage."""

    def __init__(self, connection_string: str = None, db_name: str = None):
        """
        Initialize the MongoDB chunk tracker.

        Args:
            connection_string: MongoDB connection string (optional, defaults to env variable)
            db_name: Database name (optional, defaults to env variable)
        """
        # Load environment variables from .env file
        load_dotenv()

        # Use provided values or fall back to environment variables
        self.connection_string = connection_string or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = db_name or os.getenv("MONGODB_DB_NAME", "video_chunks_db")

        self._init_db()
        logger.info(f"Initialized ChunkTracker with MongoDB at {self.connection_string}, database: {self.db_name}")

    def _init_db(self):
        """Create database collections and indexes if they don't exist."""
        try:
            # Connect to MongoDB
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.db_name]

            # Create collections if they don't exist (MongoDB creates them automatically)
            self.movies_collection = self.db["movies"]
            self.chunks_collection = self.db["chunks"]

            # Create indexes for efficient querying
            self.chunks_collection.create_index([("movie_id", ASCENDING)])
            self.chunks_collection.create_index([("start_frame", ASCENDING), ("end_frame", ASCENDING)])

            # Test connection
            self.client.admin.command('ping')
            logger.info("MongoDB connection successful, database initialized")
        except Exception as e:
            logger.error(f"Error initializing MongoDB: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def add_movie(self, title: str, duration: int, total_chunks: int) -> str:
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
            movie_data = {
                "title": title,
                "duration": duration,
                "total_chunks": total_chunks
            }

            result = self.movies_collection.insert_one(movie_data)
            movie_id = str(result.inserted_id)

            logger.info(f"Added movie '{title}' with ID {movie_id}")
            return movie_id
        except Exception as e:
            logger.error(f"Error adding movie: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def add_chunk(self, movie_id: str, chunk_index: int,
                  start_frame: int, end_frame: int, storage_path: str) -> str:
        """
        Add a new chunk to the database.

        Args:
            movie_id: ID of the movie
            chunk_index: Index of the chunk in the movie
            start_frame: Starting frame number of the chunk
            end_frame: Ending frame number of the chunk
            storage_path: Path to the chunk in storage

        Returns:
            chunk_id: ID of the added chunk
        """
        try:
            chunk_data = {
                "movie_id": movie_id,
                "chunk_index": chunk_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "storage_path": storage_path
            }

            result = self.chunks_collection.insert_one(chunk_data)
            chunk_id = str(result.inserted_id)

            logger.info(f"Added chunk {chunk_index} for movie {movie_id} at {storage_path}")
            return chunk_id
        except Exception as e:
            logger.error(f"Error adding chunk: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_chunks_for_frame_range(self, start_frame: int, end_frame: int) -> List[Dict[str, Any]]:
        """
        Get all chunks that contain frames in the specified range.

        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number

        Returns:
            List of chunk documents that contain frames in the range
        """
        try:
            logger.debug(f"Searching for chunks in range {start_frame}-{end_frame}")

            # Query for chunks that overlap with the requested range
            query = {
                "$or": [
                    # Chunk encompasses entire range
                    {"$and": [{"start_frame": {"$lte": start_frame}}, {"end_frame": {"$gte": end_frame}}]},
                    # Start of chunk is in range
                    {"$and": [{"start_frame": {"$gte": start_frame}}, {"start_frame": {"$lte": end_frame}}]},
                    # End of chunk is in range
                    {"$and": [{"end_frame": {"$gte": start_frame}}, {"end_frame": {"$lte": end_frame}}]}
                ]
            }

            chunks = list(self.chunks_collection.find(query))

            # Convert ObjectId to string for easier handling
            for chunk in chunks:
                chunk["_id"] = str(chunk["_id"])

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

    def __init__(self, chunk_tracker: ChunkTracker, remote_handler: RemoteFileHandler = None, chunk_size: int = 1000):
        """
        Initialize the video processor.

        Args:
            chunk_tracker: ChunkTracker instance to track chunks
            remote_handler: RemoteFileHandler instance for remote file operations
            chunk_size: Number of frames per chunk
        """
        self.chunk_tracker = chunk_tracker
        self.chunk_size = chunk_size
        self.remote_handler = remote_handler or RemoteFileHandler()
        logger.info(
            f"Initialized VideoProcessor with chunk size {chunk_size}, remote mode: {self.remote_handler.is_remote}")

    def process_video(self, video_path: str, storage_base_path: str, movie_title: str = None) -> str:
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

            # Check if video file exists (locally or remotely)
            if not os.path.isfile(video_path) and not (
                    self.remote_handler.is_remote and self.remote_handler.file_exists(video_path)):
                logger.error(f"Video file not found: {video_path}")
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Handle remote or local video appropriately
            if self.remote_handler.is_remote and self.remote_handler.file_exists(video_path):
                # For remote videos, we need to download to process them
                logger.info(f"Downloading remote video for processing: {video_path}")
                temp_dir = os.getenv("TEMP_DIR", "/tmp")
                temp_filename = f"{int(time.time())}_{os.path.basename(video_path)}"
                temp_path = os.path.join(temp_dir, temp_filename)

                # Ensure temp directory exists
                os.makedirs(temp_dir, exist_ok=True)

                # Download video data
                video_data = self.remote_handler.read_file(video_path)

                # Write to temp file
                with open(temp_path, 'wb') as f:
                    f.write(video_data)

                # Use the temp path for processing
                process_path = temp_path
                using_temp = True
            else:
                # Local video, use directly
                process_path = video_path
                using_temp = False

            # Open video file
            video = cv2.VideoCapture(process_path)
            if not video.isOpened():
                logger.error(f"Failed to open video file: {process_path}")
                if using_temp:
                    # Clean up temp file
                    os.remove(temp_path)
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

                    # Create chunk path
                    chunk_path = os.path.join(movie_dir, f"chunk_{chunk_index}.pkl")
                    # Normalize path
                    chunk_path = os.path.normpath(chunk_path)

                    # Convert to numpy array
                    chunk_data = np.array(current_chunk)

                    # Serialize chunk data to bytes
                    pickled_data = pickle.dumps(chunk_data)

                    # Store the chunk data (locally or remotely)
                    self.remote_handler.write_file(chunk_path, pickled_data)

                    # Register chunk in database - use the absolute path to avoid issues
                    if not self.remote_handler.is_remote:
                        # For local paths, use absolute path
                        abs_chunk_path = os.path.abspath(chunk_path)
                    else:
                        # For remote paths, use the relative path as provided
                        abs_chunk_path = chunk_path

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

            # Clean up temp file if used
            if using_temp:
                try:
                    os.remove(temp_path)
                    logger.debug(f"Removed temporary file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {temp_path}: {str(e)}")

            logger.info(f"Completed processing video {movie_title}: {chunk_index} chunks created")

            return movie_id

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise


class DistributedVideoDataset(Dataset):
    """PyTorch Dataset for accessing video data from distributed storage."""

    def __init__(
            self,
            connection_string: str = None,
            db_name: str = None,
            transform: Optional[Callable] = None,
            remote_handler: RemoteFileHandler = None
    ):
        """
        Initialize the distributed video dataset.

        Args:
            connection_string: MongoDB connection string (optional, defaults to env variable)
            db_name: Database name (optional, defaults to env variable)
            transform: Optional transform function to apply to frames
            remote_handler: RemoteFileHandler instance for remote file operations
        """
        # Load environment variables from .env file if not already loaded
        load_dotenv()

        # Connection details from parameters or environment variables
        connection_str = connection_string or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db = db_name or os.getenv("MONGODB_DB_NAME", "video_chunks_db")

        self.chunk_tracker = ChunkTracker(connection_str, db)
        self.transform = transform
        self.remote_handler = remote_handler or RemoteFileHandler()

        # Cache settings
        self._cache = {}  # Simple in-memory cache for chunks
        self._cache_limit = int(os.getenv("CACHE_LIMIT", "5"))  # Maximum number of chunks to keep in cache
        self._cache_size_bytes = 0
        self._max_cache_bytes = int(os.getenv("MAX_CACHE_BYTES", str(1024 * 1024 * 1024)))  # 1GB default

        # LRU tracking for cache
        self._cache_access_times = {}

        logger.info(f"Initialized DistributedVideoDataset with MongoDB at {connection_str}")
        logger.info(f"Cache limit: {self._cache_limit} chunks, {self._max_cache_bytes / 1024 / 1024:.1f} MB")
        logger.info(f"Remote mode: {self.remote_handler.is_remote}")

    def _load_chunk(self, chunk_path: str) -> np.ndarray:
        """
        Load a chunk from storage or cache.

        Args:
            chunk_path: Path to the chunk file

        Returns:
            Numpy array containing the chunk data
        """
        try:
            # Check if chunk is in cache
            if chunk_path in self._cache:
                logger.debug(f"Chunk found in cache: {chunk_path}")
                # Update access time for LRU tracking
                self._cache_access_times[chunk_path] = time.time()
                return self._cache[chunk_path]

            # Normalize path (convert to OS-specific format if not remote)
            if not self.remote_handler.is_remote:
                normalized_path = os.path.normpath(chunk_path)
            else:
                normalized_path = chunk_path

            # Check if file exists
            if not self.remote_handler.file_exists(normalized_path):
                logger.error(f"Chunk file not found: {normalized_path}")
                logger.error(f"Original path was: {chunk_path}")
                raise FileNotFoundError(f"Chunk file not found: {normalized_path}")

            # Load chunk from storage
            logger.debug(
                f"Loading chunk from {'remote' if self.remote_handler.is_remote else 'local'} storage: {normalized_path}")

            # Read file using appropriate handler
            file_data = self.remote_handler.read_file(normalized_path)

            # Deserialize the data
            chunk_data = pickle.loads(file_data)

            # Calculate size of the chunk
            chunk_size_bytes = len(file_data)

            # Update cache if there's room
            if len(self._cache) >= self._cache_limit or self._cache_size_bytes + chunk_size_bytes > self._max_cache_bytes:
                # Need to evict from cache - use LRU strategy
                self._evict_from_cache()

            # Add to cache
            self._cache[chunk_path] = chunk_data
            self._cache_access_times[chunk_path] = time.time()
            self._cache_size_bytes += chunk_size_bytes

            logger.debug(
                f"Added to cache: {chunk_path}, size: {chunk_size_bytes / 1024 / 1024:.2f} MB, total: {self._cache_size_bytes / 1024 / 1024:.2f} MB")

            return chunk_data
        except Exception as e:
            logger.error(f"Error loading chunk {chunk_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _evict_from_cache(self):
        """Remove least recently used items from cache until there's enough room."""
        if not self._cache:
            return

        # Sort by access time (oldest first)
        sorted_items = sorted(self._cache_access_times.items(), key=lambda x: x[1])

        # Remove items until we're under the limits
        while (len(self._cache) >= self._cache_limit or
               self._cache_size_bytes > self._max_cache_bytes * 0.8) and sorted_items:  # Keep 20% buffer

            # Get oldest item
            oldest_path, _ = sorted_items.pop(0)

            if oldest_path in self._cache:
                # Estimate size (approximate since we don't store exact size)
                item_size = len(pickle.dumps(self._cache[oldest_path]))

                # Remove from cache
                logger.debug(f"Evicting from cache: {oldest_path}, approx size: {item_size / 1024 / 1024:.2f} MB")
                del self._cache[oldest_path]
                del self._cache_access_times[oldest_path]

                # Update cache size tracking
                self._cache_size_bytes = max(0, self._cache_size_bytes - item_size)

    def _get_frames(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Get frames from distributed storage.

        Args:
            start_idx: Starting frame index
            end_idx: Ending frame index

        Returns:
            Numpy array containing the requested frames
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
            result = self.chunk_tracker.chunks_collection.find_one(
                {},
                sort=[("end_frame", DESCENDING)]
            )

            max_frame = result["end_frame"] if result else None

            return max_frame + 1 if max_frame is not None else 0
        except Exception as e:
            logger.error(f"Error in __len__: {str(e)}")
            logger.error(traceback.format_exc())
            raise


class ButterflyTorch:
    """
    Superclass of PyTorch that extends its functionality to work with distributed data storage.
    """

    def __init__(self):
        """Initialize ButterflyTorch."""
        self.datasets = {}
        logger.info("Initialized ButterflyTorch")

        # Store the original torch module
        self.torch = torch

        # Initialize remote handler (will use env vars if available)
        self.remote_handler = RemoteFileHandler()

        # Cache for processed datasets
        self._dataset_cache = {}

    def load_data(self, connection_string: str = None, shape=None, transform_fn: Optional[Callable] = None,
                  db_name: str = None) -> DistributedVideoDataset:
        """
        Load data from the distributed storage.

        Args:
            connection_string: MongoDB connection string (optional, defaults to env variable)
            shape: Optional shape constraint for the data
            transform_fn: Optional transform function to apply to frames
            db_name: MongoDB database name (optional, defaults to env variable)

        Returns:
            DistributedVideoDataset instance
        """
        try:
            # Load environment variables
            load_dotenv()

            # Use provided values or environment variables
            conn_str = connection_string or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            db = db_name or os.getenv("MONGODB_DB_NAME", "video_chunks_db")

            # Create a unique key for this dataset configuration
            dataset_key = f"{conn_str}:{db}"

            # Check if already loaded
            if dataset_key in self.datasets:
                logger.info(f"Dataset already loaded: {dataset_key}")
                return self.datasets[dataset_key]

            # Create and store dataset
            logger.info(f"Loading dataset from MongoDB at {conn_str}, database: {db}")
            dataset = DistributedVideoDataset(conn_str, db, transform_fn, self.remote_handler)
            self.datasets[dataset_key] = dataset

            return dataset
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def save_data(self, video_paths: List[str], storage_base_path: str = None, connection_string: str = None,
                  db_name: str = None, chunk_size: int = None):
        """
        Process videos and save them to distributed storage.

        Args:
            video_paths: List of paths to video files
            storage_base_path: Base path for distributed storage (optional, defaults to env variable)
            connection_string: MongoDB connection string (optional, defaults to env variable)
            db_name: MongoDB database name (optional, defaults to env variable)
            chunk_size: Number of frames per chunk (optional, defaults to env variable)
        """
        try:
            # Load environment variables
            load_dotenv()

            # Use provided values or environment variables
            storage_path = storage_base_path or os.getenv("STORAGE_PATH", "./storage")
            conn_str = connection_string or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            db = db_name or os.getenv("MONGODB_DB_NAME", "video_chunks_db")
            chunk_size_val = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))

            # Create storage directory if it doesn't exist (local mode only)
            if not self.remote_handler.is_remote:
                os.makedirs(storage_path, exist_ok=True)
                logger.info(f"Created storage directory: {storage_path}")

            # Initialize chunk tracker and video processor
            logger.info(f"Initializing ChunkTracker with MongoDB at {conn_str}, database: {db}")
            chunk_tracker = ChunkTracker(conn_str, db)

            logger.info(f"Initializing VideoProcessor with chunk size {chunk_size_val}")
            processor = VideoProcessor(chunk_tracker, self.remote_handler, chunk_size_val)

            # Process each video - check for each video and process it
            for video_path in video_paths:
                if self.remote_handler.is_remote:
                    # For remote mode, check if the file exists on the remote server
                    file_exists = self.remote_handler.file_exists(video_path)
                else:
                    # For local mode, check if the file exists locally
                    file_exists = os.path.isfile(video_path)

                if not file_exists:
                    logger.error(f"Video file not found: {video_path}")
                    continue  # Skip this file and continue with others

                logger.info(f"Processing video: {video_path}")
                processor.process_video(video_path, storage_path)

            logger.info(f"Processed {len(video_paths)} videos and saved to {storage_path}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    # Helper functions to work with remote files
    def list_remote_files(self, remote_path: str = None) -> List[str]:
        """
        List files on the remote server.

        Args:
            remote_path: Path on remote server (defaults to env variable)

        Returns:
            List of file paths
        """
        try:
            # Load environment variables
            load_dotenv()

            # Use provided value or environment variable
            path = remote_path or os.getenv("REMOTE_VIDEOS_PATH")

            if not path:
                raise ValueError("Remote path not provided and REMOTE_VIDEOS_PATH environment variable not set")

            if not self.remote_handler.is_remote:
                logger.warning("Not in remote mode, using local directory listing")
                if os.path.isdir(path):
                    return [os.path.join(path, f) for f in os.listdir(path)]
                else:
                    return []

            files = self.remote_handler.list_directory(path)
            return [os.path.join(path, f) for f in files]
        except Exception as e:
            logger.error(f"Error listing remote files: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_remote_videos(self, remote_path: str = None) -> List[str]:
        """
        Get a list of video files on the remote server.

        Args:
            remote_path: Path on remote server (defaults to env variable)

        Returns:
            List of video file paths
        """
        try:
            files = self.list_remote_files(remote_path)
            # Filter for video files
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
            return [f for f in files if f.lower().endswith(video_extensions)]
        except Exception as e:
            logger.error(f"Error getting remote videos: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def setup_remote_storage(self, remote_storage_path: str = None):
        """
        Set up remote storage directory structure.

        Args:
            remote_storage_path: Path on remote server for storage (defaults to env variable)
        """
        try:
            # Load environment variables
            load_dotenv()

            # Use provided value or environment variable
            storage_path = remote_storage_path or os.getenv("REMOTE_STORAGE_PATH")

            if not storage_path:
                raise ValueError(
                    "Remote storage path not provided and REMOTE_STORAGE_PATH environment variable not set")

            if not self.remote_handler.is_remote:
                logger.warning("Not in remote mode, using local directory creation")
                os.makedirs(storage_path, exist_ok=True)
                return

            # Create necessary directories on remote
            # The _mkdir_p method will create the directory if it doesn't exist
            self.remote_handler._mkdir_p(storage_path)
            logger.info(f"Set up remote storage at {storage_path}")
        except Exception as e:
            logger.error(f"Error setting up remote storage: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    # Forward all PyTorch attributes that aren't overridden
    def __getattr__(self, name):
        """Forward attribute access to PyTorch."""
        return getattr(torch, name)


def main():
    """
    Main function to demonstrate the usage.
    """
    try:
        # Load environment variables from .env file
        load_dotenv()

        # Get current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Get configuration from environment variables
        videos_dir = os.getenv("VIDEOS_PATH", os.path.join(script_dir, 'videos'))
        storage_dir = os.getenv("STORAGE_PATH", os.path.join(script_dir, 'storage'))
        mongo_connection = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB_NAME", "video_chunks_db")
        chunk_size = int(os.getenv("CHUNK_SIZE", "100"))

        # Remote configuration
        remote_mode = os.getenv("REMOTE_MODE", "False").lower() == "true"
        remote_host = os.getenv("REMOTE_HOST")
        remote_user = os.getenv("REMOTE_USER")
        remote_pass = os.getenv("REMOTE_PASSWORD")
        remote_videos = os.getenv("REMOTE_VIDEOS_PATH")
        remote_storage = os.getenv("REMOTE_STORAGE_PATH")

        # Get testing mode setting
        create_test_video = os.getenv("CREATE_TEST_VIDEO", "False").lower() == "true"

        # Normalize paths for local mode
        if not remote_mode:
            videos_dir = os.path.normpath(videos_dir)
            storage_dir = os.path.normpath(storage_dir)

            logger.info(f"Local mode:")
            logger.info(f"  Script directory: {script_dir}")
            logger.info(f"  Videos directory: {videos_dir}")
            logger.info(f"  Storage directory: {storage_dir}")
        else:
            logger.info(f"Remote mode:")
            logger.info(f"  Remote host: {remote_host}")
            logger.info(f"  Remote videos: {remote_videos}")
            logger.info(f"  Remote storage: {remote_storage}")

        logger.info(f"MongoDB connection: {mongo_connection}")
        logger.info(f"Database name: {db_name}")
        logger.info(f"Chunk size: {chunk_size}")

        # Initialize ButterflyTorch
        my_torch = ButterflyTorch()

        # Set up storage directories
        if remote_mode:
            my_torch.setup_remote_storage(remote_storage)
            # Get video paths from remote server
            video_paths = my_torch.get_remote_videos(remote_videos)
        else:
            # Ensure local directories exist
            os.makedirs(storage_dir, exist_ok=True)
            os.makedirs(videos_dir, exist_ok=True)

            # Get local video files
            video_paths = [os.path.join(videos_dir, f) for f in os.listdir(videos_dir)
                           if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            video_paths = [os.path.normpath(p) for p in video_paths]

        if not video_paths and create_test_video and not remote_mode:
            logger.warning(f"No video files found, creating test video")
            # Create a small test video if no videos found
            logger.info("Creating a test video for demonstration...")
            try:
                import numpy as np
                import cv2

                test_video_path = os.path.join(videos_dir, "test_video.mp4")

                # Create a small test video (100 frames of colored noise)
                height, width = 320, 240
                fps = 30
                seconds = 3

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(test_video_path, fourcc, fps, (width, height))

                for i in range(seconds * fps):
                    # Create a random colored frame
                    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                    out.write(frame)

                out.release()
                logger.info(f"Created test video at {test_video_path}")

                video_paths = [test_video_path]
            except ImportError:
                logger.error("OpenCV (cv2) is required but not installed. Cannot create test video.")
                logger.info("Please install it with: pip install opencv-python")
        elif not video_paths:
            if remote_mode:
                logger.warning(f"No video files found in remote path {remote_videos}")
            else:
                logger.warning(f"No video files found in {videos_dir} and CREATE_TEST_VIDEO is not enabled")
            logger.info("Please add video files to the videos directory or set CREATE_TEST_VIDEO=True in .env")
            return False

        logger.info(f"Found {len(video_paths)} video files: {[os.path.basename(p) for p in video_paths]}")

        # Process videos and save to distributed storage
        logger.info("Starting video processing...")

        if remote_mode:
            # Use remote paths for storage
            my_torch.save_data(video_paths, remote_storage)
        else:
            # Use local paths
            my_torch.save_data(video_paths, storage_dir)

        # Define a transform function (similar to PyTorch transforms)
        def transform_fn(frames):
            # Normalize frames
            return frames / 255.0

        # Load data
        logger.info("Loading processed data...")
        load_data = my_torch.load_data(transform_fn=transform_fn)

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
            if not remote_mode:
                storage_files = []
                for root, dirs, files in os.walk(storage_dir):
                    for file in files:
                        if file.endswith('.pkl'):
                            storage_files.append(os.path.join(root, file))
                logger.info(f"Storage directory contains {len(storage_files)} pickle files")

            # Demonstrate using some PyTorch functionality through ButterflyTorch
            if len(fetched_data) > 0:
                # Convert to PyTorch tensor
                tensor_data = my_torch.tensor(fetched_data)
                logger.info(f"Successfully converted to PyTorch tensor of shape {tensor_data.shape}")

                # Do some simple tensor operations
                mean_val = tensor_data.mean().item()
                logger.info(f"Mean value in fetched frames: {mean_val}")

                # Example of using other PyTorch functionality
                if tensor_data.shape[0] > 1:
                    # Calculate standard deviation
                    std_val = tensor_data.std().item()
                    logger.info(f"Standard deviation in fetched frames: {std_val}")

                    # Apply a PyTorch function
                    normalized = my_torch.nn.functional.normalize(tensor_data, dim=0)
                    logger.info(f"Applied normalization, new mean: {normalized.mean().item()}")

        logger.info("Operation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("Successfully processed videos and stored chunks in the distributed storage.")
    else:
        print("Error occurred during processing. Check logs for details.")