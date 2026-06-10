# Embedding API

Unified RESTful API for remote sensing embeddings and downstream task results from Harbin New Area and Haidian District.

## Features

- **Multi-region support**: Harbin (哈尔滨新区) & Haidian (海淀区)
- **Embedding queries**: PNG visualization, NPY arrays, JSON statistics
- **Downstream tasks**: Construction, building change, farmland, land conversion, demolition monitoring
- **Tile service**: Map tile serving for web GIS integration
- **Hot-reload config**: Add new regions/tasks without restarting

## Quick Start

### Local Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker-compose up -d
```

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Base
- `GET /health` - Health check

### Regions
- `GET /regions` - List all regions
- `GET /regions/{region_id}` - Get region details

### Patches
- `GET /regions/{region_id}/patches` - List patches (supports bbox filtering)
- `GET /regions/{region_id}/patches/{patch_id}` - Get patch details

### Embeddings
- `GET /regions/{region_id}/patches/{patch_id}/embedding?format=png|npy|json`

### Downstream Tasks
- `GET /regions/{region_id}/tasks` - List tasks
- `GET /regions/{region_id}/tasks/{task_type}/summary` - Task summary
- `GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result` - Result image
- `GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/prediction` - Raw prediction
- `GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/label` - Label data

### Tiles
- `GET /regions/{region_id}/tasks/{task_type}/tiles` - List tiles
- `GET /regions/{region_id}/tasks/{task_type}/tiles/{z}/{x}/{y}.png` - Map tile

## Configuration

Edit `config.yaml` to add new regions or tasks. Changes are detected automatically without restart.

## Data Structure

See `docs/API.md` for detailed API documentation.
