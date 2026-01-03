# Lainbow

Lainbow is a music analysis engine designed to process local music libraries. It offers batch processing capabilities to extract deep learning embeddings (MERT, CLAP, MuQ, MuQ-MuLan) and various acoustic features from audio files. These features are then stored in a vector database to power an API for tasks like similarity-based music recommendations and natural language search.

The project is primarily intended for integration with the MPD client, [Sola MPD](https://github.com/prokosna/sola_mpd).

## Architecture Overview

Lainbow is built on a microservices architecture, with each component containerized using Docker. The system consists of four main services that work together:

- **Web API Server**: The main entry point for user requests. It handles API calls, interacts with the databases, and delegates heavy tasks to the batch server.
- **Inference Server**: A dedicated service that hosts the deep learning models and performs inference tasks (e.g., generating embeddings from audio or text).
- **Batch Server**: A Celery-based worker that processes long-running, asynchronous tasks, such as scanning the music library and analyzing songs.
- **Databases**: A set of databases for storing metadata, vector embeddings, and managing task queues:
    - **PostgreSQL**: Stores song metadata, features, and task information.
    - **Qdrant**: A vector database for storing and searching high-dimensional embeddings.
    - **RabbitMQ**: A message broker that facilitates communication between the Web API and the Batch Server.

```mermaid
graph TD
    subgraph User
        U[User]
    end

    subgraph "api.yaml"
        Web_API[Web API Server]
    end

    subgraph "batch.yaml"
        Batch_Server[Batch Server]
    end

    subgraph "inference.yaml"
        Inference_Server["Inference Server (GPU Required)"]
    end

    subgraph "database.yaml"
        PostgreSQL[PostgreSQL]
        Qdrant["Vector DB (Qdrant)"]
        RabbitMQ["Message Queue (RabbitMQ)"]
    end

    %% Connections
    U --> Web_API
    Web_API -- Enqueue Task --> RabbitMQ
    Web_API -- CRUD --> PostgreSQL
    Web_API -- Search --> Qdrant

    RabbitMQ -- Consume Task --> Batch_Server
    Batch_Server -- HTTP Request --> Inference_Server
    Batch_Server -- CRUD --> PostgreSQL
    Batch_Server -- Insert --> Qdrant

    style U fill:#f9f,stroke:#333,stroke-width:2px
    style Web_API fill:#bbf,stroke:#333,stroke-width:2px
    style Inference_Server fill:#ffc,stroke:#333,stroke-width:2px
    style Batch_Server fill:#cdf,stroke:#333,stroke-width:2px
```

## Model Preparation

**Before running the application, you need to download the required deep learning models.**

1.  **Install Dependencies**:
    Ensure you have `uv` installed.

2.  **Download Models**:
    Run the provided script to download and place the models in the `./models` directory. The script will skip any models that are already downloaded.
    ```bash
    uv run --with httpx --with huggingface-hub --with muq python download_models.py
    ```

This script will download:
- The MERT model (`m-a-p/MERT-v1-330M`)
- The CLAP model (`laion/clap-htsat-unfused`)
- The MuQ model (`OpenMuQ/MuQ-large-msd-iter`)
- The MuQ-MuLan model (`OpenMuQ/MuQ-MuLan-large`)

Once the script completes, the application will be ready to run.

## Setup and Configuration

### Environment Variables
1.  **Create `.env` file**: Copy the template to create your own environment file.
    ```bash
    cp .env.template .env
    ```
2.  **Edit `.env` file**: Open the `.env` file and customize the variables. **You must at least set `MUSIC_NAS_ROOT_DIR` to the absolute path of your music library.** The default settings should work out-of-the-box if you are running all components on a single machine with no port conflicts.

### GPU Configuration for Inference Server
The `docker/inference.Dockerfile` uses a specific PyTorch base image optimized for author's GPU (RTX 5070 Ti). **You may need to modify the `FROM` instruction in this Dockerfile** to use a base image compatible with your GPU hardware.

Additionally, **ensure that the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is properly installed and configured**. This is required for Docker to access and utilize the GPU.

There is also `docker/inference.rocm.Dockerfile` for ROCm based inference server. To use this Docker image, you need to setup [ROCm on Docker](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html) similar to NVIDIA.

### Starting the Application

Start the services on your server(s). You can run each service on a separate server or on the same machine. **Ensure that the `.env` file on each server is configured correctly** to allow communication between the services.

**Database Server**
```bash
docker compose -f docker-compose.database.yaml up -d
```

### Migrating from Milvus to Qdrant (Optional)
If you already have embeddings stored in Milvus and want to migrate them to Qdrant:

1. Start both Milvus and Qdrant on the database server (uncomment the Milvus services in `docker-compose.database.yaml`).
2. Run the migration script:
   ```bash
   uv run --group batch python migrate_milvus_to_qdrant.py
   ```

After the migration finishes, you can disable Milvus again and keep using Qdrant.

**Web API Server**
```bash
docker compose -f docker-compose.api.yaml up -d
```

**Inference API Server**
```bash
docker compose -f docker-compose.inference.yaml --profile [cuda|rocm] up -d
```

**Batch Server**
```bash
docker compose -f docker-compose.batch.yaml up -d --scale batch-cpu=N
```
*(Note: `N` is the number of parallel processes. A value around 4-6 is recommended.)*

## API Endpoints

### General

#### `GET /health`
- **Summary**: Health check endpoint.
- **Response (200 OK)**: `{"status":"ok"}`

### Statistics

#### `GET /api/v1/stats`
- **Summary**: Get database statistics.
- **Description**: Retrieve statistics about the current state of the database, including song counts and task statuses.
- **Response (200 OK)**:
  ```json
  {
    "total_songs": 1000,
    "songs_with_acoustic_features": 50,
    "songs_with_clap": 100,
    "songs_with_mert": 100,
    "songs_with_muq": 200,
    "songs_with_muq_mulan": 200,
    "pending_tasks": 5,
    "running_tasks": 2
  }
  ```

### Batch Processing

#### `POST /api/v1/batch/scan/run`
- **Summary**: Start a music library scan.
- **Description**: Triggers a background task to scan the music library path specified in `.env`.
- **Response (200 OK)**: Returns a `TaskResult` object.
  ```json
  {
    "id": "...",
    "name": "scan",
    "status": "PENDING",
    "result": null
  }
  ```

#### `POST /api/v1/batch/vacuum/run`
- **Summary**: Run library vacuum.
- **Description**: Triggers a background task to remove database entries for songs that no longer exist on disk.
- **Response (200 OK)**: Returns a `TaskResult` object.

#### `POST /api/v1/batch/analyze/run`
- **Summary**: Run song analysis.
- **Description**: Enqueues a task to analyze all songs and generate embeddings for the specified models.
- **Request Body** (optional):
  ```json
  {
    "models": ["muq", "muq_mulan"]
  }
  ```
  *If the body is omitted, it defaults to `["muq", "muq_mulan"]`.*
- **Response (200 OK)**: Returns a `TaskResult` object.

#### `GET /api/v1/batch/tasks/{task_id}`
- **Summary**: Get task status.
- **Description**: Retrieves the current status and result of a specific background task.
- **Path Parameters**:
  - `task_id` (UUID, required): The ID of the task.
- **Response (200 OK)**: Returns a `TaskResult` object with the current status.

### Songs

#### `GET /api/v1/songs/search`
- **Summary**: Search songs by natural language.
- **Description**: Searches for songs based on a natural language query, e.g., "a song for a summer evening".
- **Query Parameters**:
  - `q` (string, required): Natural language query.
  - `model_name` (string, optional, default: `muq_mulan`): The text embedding model to use. Can be `clap` or `muq_mulan`.
  - `limit` (integer, optional, default: 10): Maximum number of results.
- **Response (200 OK)**: Returns a list of `SearchSong` objects.

#### `GET /api/v1/songs/{file_path}/analysis`
- **Summary**: Get detailed song analysis.
- **Description**: Retrieves detailed analysis for a given song, including metadata, features, and embedding status.
- **Path Parameters**:
  - `file_path` (string, required): The URL-encoded file path of the song.
- **Response (200 OK)**: Returns a `SongAnalysis` object.

#### `GET /api/v1/songs/{file_path}/similar`
- **Summary**: Find similar songs.
- **Description**: Finds songs acoustically similar to the given song using a specified vector embedding model.
- **Path Parameters**:
  - `file_path` (string, required): The URL-encoded file path of the song.
- **Query Parameters**:
  - `model_name` (string, optional, default: `muq`): The embedding model to use. Can be `acoustic_features`, `clap`, `mert`, `muq`, `muq_mulan`.
  - `limit` (integer, optional, default: 10): Maximum number of results.
- **Response (200 OK)**: Returns a list of `SearchSong` objects.
