# ButterflyTorch

ButterflyTorch is a PyTorch extension that enables seamless access to large-scale video data (100TB+) stored in distributed storage systems.

## Features

- Extends PyTorch functionality while maintaining its familiar interface
- Manages distributed video data and provides access as if it were local
- Supports both local and remote (SSH) data storage
- MongoDB-based tracking of video chunks
- Transparent frame-level access across chunk boundaries
- Efficient caching system for improved performance
- Handles data transforms similar to standard PyTorch
- Works with large datasets across distributed storage

## Installation

### Requirements

```bash
pip install torch pymongo python-dotenv paramiko opencv-python
```

### Configuration

Create a `.env` file in your project directory with the following settings:

```
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/?directConnection=true
MONGODB_DB_NAME=video_chunks_db

# Local Storage Configuration
STORAGE_PATH=./storage
VIDEOS_PATH=./src/videos

# Processing Configuration
CHUNK_SIZE=1000
CACHE_LIMIT=10
MAX_CACHE_BYTES=1073741824  # 1GB in bytes

# Remote Server Configuration (if needed)
REMOTE_MODE=False  # Set to True to enable remote mode
REMOTE_HOST=your_server_hostname
REMOTE_PORT=22
REMOTE_USER=your_username
REMOTE_PASSWORD=your_password
# REMOTE_KEY_PATH=/path/to/your/ssh/key  # Uncomment if using key-based auth

# Remote Paths (if remote mode is enabled)
REMOTE_VIDEOS_PATH=/remote/path/to/videos
REMOTE_STORAGE_PATH=/remote/path/to/storage
```

## Usage

### Basic Usage (Local Mode)

```python
# Initialize ButterflyTorch
my_torch = ButterflyTorch()

# Process videos and save to distributed storage
video_paths = ['path/to/video1.mp4', 'path/to/video2.mp4']
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
```

### Remote Mode Usage

When working with remote servers holding large datasets:

1. Update your `.env` file with remote server details and set `REMOTE_MODE=True`
2. Run the same code as above, and ButterflyTorch will automatically handle the remote connection

```python
# Initialize ButterflyTorch (will use remote connection from .env)
my_torch = ButterflyTorch()

# Find videos on remote server
remote_videos = my_torch.get_remote_videos()

# Process videos on remote server
my_torch.save_data(remote_videos)

# Load data - seamlessly fetches from remote storage
dataset = my_torch.load_data(transform_fn=normalize_frames)

# Access frames just like they were local
frames = dataset[1000:2000]
```

## Architecture

ButterflyTorch consists of several components:

1. **ButterflyTorch**: Main class that extends PyTorch functionality
2. **ChunkTracker**: MongoDB-based tracking system for video chunks
3. **VideoProcessor**: Processes videos into manageable chunks
4. **DistributedVideoDataset**: PyTorch-compatible dataset for accessing distributed data
5. **RemoteFileHandler**: Handles file operations on remote servers

## Handling Large Datasets (100TB+)

ButterflyTorch is designed to handle extremely large datasets by:

1. Splitting videos into manageable chunks
2. Using MongoDB for efficient chunk location tracking
3. Loading only required frames into memory
4. Implementing intelligent caching with LRU eviction
5. Supporting distributed storage across servers
6. Optimizing memory usage with on-demand loading

## Best Practices

- Adjust `CHUNK_SIZE` based on your memory constraints and access patterns
- Increase `CACHE_LIMIT` and `MAX_CACHE_BYTES` on systems with more RAM
- Use SSH keys instead of passwords for more secure remote connections
- For extremely large datasets spanning multiple servers, use multiple RemoteFileHandler instances
- Monitor MongoDB performance and index usage as your dataset grows

## License

MIT