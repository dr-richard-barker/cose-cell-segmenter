import tempfile
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from cellpose import io as cp_io

from cose_cell_segmenter.core.segmentation import SegmentationEngine, SegmentParams

app = FastAPI(title="CoSE Cell Segmenter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SegmentationEngine()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    params: str = Form(...)  # JSON string of params
):
    try:
        parsed_params = json.loads(params)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid params JSON: {e}"})
    
    seg_params = SegmentParams(**parsed_params)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load image
        image = cp_io.imread(tmp_path)
        
        # Run segmentation
        masks, flows = engine.run(image, seg_params)
        
        from skimage.measure import find_contours
        
        polygons = []
        for cell_id in np.unique(masks):
            if cell_id == 0:
                continue
            
            # find_contours returns a list of (N, 2) arrays
            contours = find_contours(masks == cell_id, 0.5)
            for contour in contours:
                # To reduce payload size, we might want to downsample the contour points
                points = [{"x": round(float(p[1]), 1), "y": round(float(p[0]), 1)} for p in contour[::2]] # Downsample by 2
                polygons.append({
                    "id": int(cell_id),
                    "points": points
                })
                
        return {"polygons": polygons, "shape": image.shape}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
