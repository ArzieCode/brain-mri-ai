"""
Report Service — Enhanced
==========================
Support: DicomMetadata, SecondOpinion, Comparison, disk persistence.
"""

import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from backend.core.config import settings
from backend.api.schemas.schemas import (
    FullAnalysisReport, ValidationResult, PredictionResult,
    GradCAMResult, DicomMetadata, SecondOpinionResult, ComparisonResult,
)

REPORTS_DIR = Path(settings.OUTPUTS_DIR) / "reports"


def _sanitize(text: str) -> str:
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"', "\u2022": "*", "\u00b1": "+/-",
        "\u2265": ">=", "\u2264": "<=", "\u00d7": "x", "\u2026": "...",
        "\u2192": "->",
    }
    for char, rep in replacements.items():
        text = text.replace(char, rep)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class ReportService:

    def __init__(self):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self._reports: Dict[str, FullAnalysisReport] = {}
        self._load_all_from_disk()

    def _report_path(self, report_id: str) -> Path:
        return REPORTS_DIR / f"{report_id}.json"

    def _save_to_disk(self, report: FullAnalysisReport) -> None:
        try:
            self._report_path(report.report_id).write_text(
                report.model_dump_json(), encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Failed to save report {report.report_id}: {e}")

    def _load_all_from_disk(self) -> None:
        loaded = 0
        for path in REPORTS_DIR.glob("*.json"):
            try:
                data   = json.loads(path.read_text(encoding="utf-8"))
                report = FullAnalysisReport.model_validate(data)
                self._reports[report.report_id] = report
                loaded += 1
            except Exception as e:
                logger.warning(f"Could not load report {path.name}: {e}")
        if loaded:
            logger.info(f"Loaded {loaded} reports from disk")

    def create_report(
        self,
        image_filename: str,
        validation: ValidationResult,
        prediction: Optional[PredictionResult]       = None,
        gradcam: Optional[GradCAMResult]             = None,
        dicom_metadata: Optional[DicomMetadata]      = None,
        second_opinion: Optional[PredictionResult]   = None,
    ) -> FullAnalysisReport:
        report_id = str(uuid.uuid4())[:8].upper()

        # Build second opinion result jika ada
        second_opinion_result = None
        if second_opinion and prediction:
            agrees = second_opinion.prediction == prediction.prediction
            second_opinion_result = SecondOpinionResult(
                prediction=second_opinion.prediction,
                confidence=second_opinion.confidence,
                agrees_with_primary=agrees,
                disagreement_note=(
                    f"Second opinion suggests {second_opinion.prediction.value} "
                    f"({second_opinion.confidence:.1%}) vs primary {prediction.prediction.value} "
                    f"({prediction.confidence:.1%}). Consider additional review."
                ) if not agrees else None,
                inference_time_ms=second_opinion.inference_time_ms,
            )

        # Build DicomMetadata jika ada dict
        dicom_meta_obj = None
        if dicom_metadata:
            if isinstance(dicom_metadata, dict):
                dicom_meta_obj = DicomMetadata(**dicom_metadata)
            else:
                dicom_meta_obj = dicom_metadata

        report = FullAnalysisReport(
            report_id=report_id,
            image_filename=image_filename,
            timestamp=datetime.now(timezone.utc),
            validation=validation,
            prediction=prediction,
            gradcam=gradcam,
            dicom_metadata=dicom_meta_obj,
            second_opinion=second_opinion_result,
        )

        self._reports[report_id] = report
        self._save_to_disk(report)
        logger.info(f"Report created: {report_id}")
        return report

    def get_report(self, report_id: str) -> Optional[FullAnalysisReport]:
        return self._reports.get(report_id)

    def list_reports(self) -> list[FullAnalysisReport]:
        return sorted(self._reports.values(), key=lambda r: r.timestamp, reverse=True)

    def delete_report(self, report_id: str) -> bool:
        if report_id in self._reports:
            del self._reports[report_id]
            p = self._report_path(report_id)
            if p.exists():
                p.unlink()
            return True
        return False

    def compare_reports(self, report_id_a: str, report_id_b: str) -> Optional[ComparisonResult]:
        a = self.get_report(report_id_a)
        b = self.get_report(report_id_b)
        if not a or not b:
            return None
        if not a.prediction or not b.prediction:
            return None

        changed = a.prediction.prediction != b.prediction.prediction
        return ComparisonResult(
            scan_a_report_id=report_id_a,
            scan_b_report_id=report_id_b,
            scan_a_prediction=a.prediction.prediction.value,
            scan_b_prediction=b.prediction.prediction.value,
            scan_a_confidence=a.prediction.confidence,
            scan_b_confidence=b.prediction.confidence,
            changed=changed,
            change_note=(
                f"Prediction changed from {a.prediction.prediction.value} to "
                f"{b.prediction.prediction.value}. This may indicate disease progression "
                "or response to treatment. Clinical correlation is essential."
            ) if changed else None,
        )

    def generate_pdf(self, report_id: str) -> Optional[bytes]:
        report = self.get_report(report_id)
        if not report:
            logger.warning(f"Report not found: {report_id}")
            return None

        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Header
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_fill_color(20, 30, 50)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 15, "Brain MRI AI Analysis Report", ln=True, fill=True, align="C")
            pdf.ln(5)

            # Disclaimer
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(255, 200, 50)
            pdf.set_text_color(50, 30, 0)
            pdf.multi_cell(0, 8,
                "WARNING: AI-ASSISTED ANALYSIS ONLY - NOT FOR CLINICAL USE - "
                "CONSULT A QUALIFIED MEDICAL PROFESSIONAL",
                fill=True, align="C"
            )
            pdf.ln(5)

            # Metadata
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 7, f"Report ID : {report.report_id}", ln=True)
            pdf.cell(0, 7, f"Generated : {report.timestamp.strftime('%Y-%m-%d %H:%M UTC')}", ln=True)
            pdf.cell(0, 7, _sanitize(f"File      : {report.image_filename}"), ln=True)
            pdf.cell(0, 7, f"Model Ver : {report.model_version}", ln=True)
            pdf.ln(5)

            # DICOM Metadata
            if report.dicom_metadata:
                d = report.dicom_metadata
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(20, 30, 50)
                pdf.cell(0, 10, "DICOM Scan Information", ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(0, 0, 0)
                fields = [
                    ("Modality",        d.modality),
                    ("Study Date",      d.study_date),
                    ("Scanner",         d.scanner_model),
                    ("Field Strength",  d.field_strength),
                    ("Slice Thickness", d.slice_thickness),
                    ("Description",     d.study_description),
                ]
                for label, val in fields:
                    if val:
                        pdf.cell(0, 6, _sanitize(f"  {label:<18}: {val}"), ln=True)
                pdf.ln(5)

            # Validation
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(20, 30, 50)
            pdf.cell(0, 10, "1. Image Validation", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(0, 0, 0)
            v = report.validation
            status = "VALID Brain MRI" if v.is_valid_brain_mri else f"INVALID - {v.status.value}"
            pdf.cell(0, 7, f"  Status        : {status}", ln=True)
            pdf.cell(0, 7, f"  Detected Type : {v.detected_type} ({v.type_confidence * 100:.1f}%)", ln=True)
            if v.quality_report:
                q = v.quality_report
                pdf.cell(0, 7, f"  Image Size    : {q.width} x {q.height} px", ln=True)
                pdf.cell(0, 7, f"  Blur Score    : {q.blur_score:.1f}", ln=True)
                pdf.cell(0, 7, f"  Brightness    : {q.mean_brightness:.1f} / 255", ln=True)
            pdf.ln(5)

            # Prediction
            if report.prediction:
                p = report.prediction
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(20, 30, 50)
                pdf.cell(0, 10, "2. Tumor Classification", ln=True)
                pdf.set_font("Helvetica", "B", 12)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 9, f"  Prediction : {p.prediction.value.upper()}", ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 7, f"  Confidence : {p.confidence * 100:.1f}%", ln=True)
                pdf.cell(0, 7, f"  Risk Level : {p.risk_level.value.upper()}", ln=True)
                pdf.cell(0, 7, f"  Inference  : {p.inference_time_ms:.1f} ms", ln=True)
                pdf.ln(3)

                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, "  Class Probabilities:", ln=True)
                pdf.set_font("Helvetica", "", 10)
                for cp in p.class_probabilities:
                    bar = "|" * int(cp.percentage / 5)
                    pdf.cell(0, 6, f"    {cp.class_name:<14} {cp.percentage:5.1f}%  {bar}", ln=True)
                pdf.ln(3)

                u = p.uncertainty
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, "  Uncertainty (Monte Carlo Dropout):", ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 6, f"    Level     : {u.uncertainty_level.value.upper()}", ln=True)
                pdf.cell(0, 6, f"    Std Dev   : +/- {u.std_confidence * 100:.1f}%", ln=True)
                pdf.cell(0, 6, f"    95% CI    : [{u.confidence_interval_low * 100:.1f}% to {u.confidence_interval_high * 100:.1f}%]", ln=True)
                pdf.cell(0, 6, f"    MC Passes : {u.mc_passes}", ln=True)
                pdf.ln(5)

                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, "  Clinical Context:", ln=True)
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(80, 80, 80)
                pdf.multi_cell(0, 5, _sanitize(f"  {p.clinical_notes}"))
                pdf.set_text_color(0, 0, 0)
                pdf.ln(5)

            # Second Opinion
            if report.second_opinion:
                s = report.second_opinion
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(20, 30, 50)
                pdf.cell(0, 10, "3. Second Opinion (30-pass MC Dropout)", ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(0, 0, 0)
                agree_str = "AGREES with primary" if s.agrees_with_primary else "DISAGREES with primary"
                pdf.cell(0, 7, f"  Result    : {s.prediction.value.upper()} ({s.confidence * 100:.1f}%)", ln=True)
                pdf.cell(0, 7, f"  Status    : {agree_str}", ln=True)
                if s.disagreement_note:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.set_text_color(180, 50, 50)
                    pdf.multi_cell(0, 5, _sanitize(f"  {s.disagreement_note}"))
                    pdf.set_text_color(0, 0, 0)
                pdf.ln(5)

            # GradCAM
            if report.gradcam:
                g = report.gradcam
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(20, 30, 50)
                pdf.cell(0, 10, "4. Explainability (GradCAM)", ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 7, _sanitize(f"  Attention Region : {g.attention_region}"), ln=True)
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(80, 80, 80)
                pdf.multi_cell(0, 5, _sanitize(f"  {g.explanation}"))
                pdf.ln(5)

            # Footer
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "B", 9)
            pdf.ln(5)
            pdf.set_fill_color(240, 240, 240)
            pdf.multi_cell(0, 5,
                "AI-Assisted Analysis Only. This report is generated by an AI system "
                "and is NOT a substitute for professional medical diagnosis. "
                "Always consult a qualified radiologist or neurologist.",
                fill=True
            )

            return pdf.output()

        except ImportError:
            logger.warning("fpdf2 not installed")
            return None
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return None


# Singleton
report_service = ReportService()