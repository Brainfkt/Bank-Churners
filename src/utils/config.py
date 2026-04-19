from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


UNKNOWN_COLUMNS = ("Education_Level", "Marital_Status", "Income_Category")
NAIVE_BAYES_PREFIX = "Naive_Bayes_Classifier_"
TARGET_COLUMN = "Attrition_Flag"
TARGET_NAME = "churn_flag"
IDENTIFIER_COLUMN = "CLIENTNUM"
POSITIVE_CLASS_LABEL = "Attrited Customer"
NEGATIVE_CLASS_LABEL = "Existing Customer"
RANDOM_STATE = 42


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized repository paths."""

    root: Path = Path(__file__).resolve().parents[2]

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def figures(self) -> Path:
        return self.root / "reports" / "figures"

    @property
    def report_markdown(self) -> Path:
        return self.root / "reports" / "markdown"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def output_metrics(self) -> Path:
        return self.root / "outputs" / "metrics"

    @property
    def output_predictions(self) -> Path:
        return self.root / "outputs" / "predictions"

    @property
    def output_segmentation(self) -> Path:
        return self.root / "outputs" / "segmentation"

    @property
    def output_dashboard(self) -> Path:
        return self.root / "outputs" / "dashboard"

    @property
    def raw_dataset(self) -> Path:
        return self.data_raw / "BankChurners.csv"


PATHS = ProjectPaths()

