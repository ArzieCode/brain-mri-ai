"""
DICOM Service
=============
Convert DICOM (.dcm) files to JPEG/PNG untuk diproses pipeline.
Juga ekstrak metadata teknis dari header DICOM.
"""

import io
import numpy as np
from typing import Optional
from loguru import logger
from PIL import Image


def is_dicom(filename: str, file_bytes: bytes) -> bool:
    """Cek apakah file adalah DICOM berdasarkan ekstensi atau magic bytes."""
    if filename.lower().endswith(".dcm"):
        return True
    # DICOM magic: bytes 128-132 = "DICM"
    if len(file_bytes) > 132 and file_bytes[128:132] == b"DICM":
        return True
    return False


def dicom_to_png(file_bytes: bytes) -> tuple[bytes, dict]:
    """
    Convert DICOM bytes ke PNG bytes + ekstrak metadata.
    Returns (png_bytes, metadata_dict)
    """
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    except ImportError:
        raise RuntimeError("pydicom not installed. Run: pip install pydicom")

    ds = pydicom.dcmread(io.BytesIO(file_bytes))

    # Ekstrak pixel array
    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply VOI LUT jika ada (windowing)
    try:
        pixel_array = apply_voi_lut(pixel_array, ds)
    except Exception:
        pass

    # Normalize ke 0-255
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax > pmin:
        pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
    pixel_array = pixel_array.astype(np.uint8)

    # Handle multi-frame (ambil frame tengah)
    if pixel_array.ndim == 3 and pixel_array.shape[0] > 1:
        mid = pixel_array.shape[0] // 2
        pixel_array = pixel_array[mid]

    # Convert ke RGB PIL
    if pixel_array.ndim == 2:
        pil_img = Image.fromarray(pixel_array, mode="L").convert("RGB")
    else:
        pil_img = Image.fromarray(pixel_array).convert("RGB")

    # Save ke PNG bytes
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    # Ekstrak metadata
    metadata = {}
    fields = [
        ("PatientID",          "patient_id"),
        ("StudyDate",          "study_date"),
        ("Modality",           "modality"),
        ("Manufacturer",       "manufacturer"),
        ("ManufacturerModelName", "scanner_model"),
        ("MagneticFieldStrength", "field_strength"),
        ("SliceThickness",     "slice_thickness"),
        ("RepetitionTime",     "repetition_time"),
        ("EchoTime",           "echo_time"),
        ("StudyDescription",   "study_description"),
        ("SeriesDescription",  "series_description"),
        ("PixelSpacing",       "pixel_spacing"),
        ("Rows",               "rows"),
        ("Columns",            "columns"),
    ]
    for dicom_tag, key in fields:
        try:
            val = getattr(ds, dicom_tag, None)
            if val is not None:
                metadata[key] = str(val)
        except Exception:
            pass

    logger.info(f"[DICOM] Converted successfully. Modality={metadata.get('modality', 'unknown')}")
    return png_bytes, metadata
