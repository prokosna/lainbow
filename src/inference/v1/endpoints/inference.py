import io
import logging
import time
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from inference.core.model_manager import MODEL_REGISTRY, ModelManager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/inference/{model_name}")
async def run_inference(
    request: Request,
    model_name: str,
    file: Annotated[UploadFile, File(...)],
) -> StreamingResponse:
    """Runs inference on a given model with a serialized numpy array."""
    start = time.time()
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found.",
        )

    try:
        npy_bytes = await file.read()

        with io.BytesIO(npy_bytes) as buffer:
            input_data = np.load(buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process uploaded file: {e}",
        ) from e

    try:
        # Run inference with the batch of segments
        model_manager: ModelManager = request.app.state.model_manager
        embedding = model_manager.run_inference(
            model_name=model_name,
            input_data=input_data,
        )

        # Serialize the single output embedding array back to bytes in .npy format.
        with io.BytesIO() as buffer:
            np.save(buffer, embedding)
            buffer.seek(0)
            output_bytes = buffer.read()

        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={model_name}_embedding.npy"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during inference: {e}",
        ) from e
    finally:
        logger.info(f"{model_name} inference completed in {time.time() - start:.2f} seconds.")
