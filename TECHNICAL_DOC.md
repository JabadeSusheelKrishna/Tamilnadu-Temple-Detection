# Technical Documentation: Tamil Nadu Temple Recognition System

## Overview
This system identifies famous Tamil Nadu temples from user-uploaded images using a Contrastive Language-Image Pre-training (CLIP) approach. The architecture is modular, allowing for different "Strategies" (Versions) to be used for inference.

## Architecture

### 1. Frontend: Streamlit
The UI is built with Streamlit, leveraging custom CSS for a premium aesthetic.
- **State Management**: Uses `st.cache_resource` to load the CLIP model once and share it across sessions.
- **Interactivity**: Dynamic tabs for displaying metadata (History, Visit Info, Maps).

### 2. Backend: CLIP + Strategies
The `backend.py` module defines the `TempleClassifier` class.

#### Inference Strategies (Versions):

| Version | Name | Description |
| :--- | :--- | :--- |
| **V1** | Baseline | Direct zero-shot using temple names. |
| **V2** | Prompt Engineering | Uses prompts like *"A majestic photo of [Name]..."* to provide better context to the CLIP text encoder. |
| **V3** | ROI Focus | Center-crops the image (70% area) to minimize background noise and focus on temple architecture. |
| **V4** | Image Enhancement | Applies a sharpening kernel via OpenCV to highlight intricate stone carvings before inference. |
| **V5** | Hybrid Ensemble | Simulates a learned head by weighted ensembling of V2 and V4 results. |

### 3. Data Schema
The system relies on `temples_metadata.json` which stores:
- `history`: Contextual background of the temple.
- `hours`: Visitation timings.
- `tickets`: Pricing details.
- `coordinates`: Lat/Long for map visualization.
- `maps_url`: Direct search link for Google Maps.

## Handling False Positives
To ensure the model doesn't hallucinate temple names for random images, a **Negative Class** is included in the label set:
*"A photo of a person, animal, city street, or object that is not a temple"*
If this class has the highest probability, the system reports no temple found.

## Performance Considerations
- **Model**: `openai/clip-vit-base-patch32` (~600MB).
- **Device**: Automatically detects CUDA; falls back to CPU if unavailable.
- **Latency**: Inference typically takes <500ms on CPU.
