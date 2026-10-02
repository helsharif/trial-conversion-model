import importlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST


@pytest.fixture
def workflow(monkeypatch):
    # Importing the existing module selects an experiment; prevent network I/O.
    with patch("mlflow.set_experiment"), patch("dotenv.load_dotenv"):
        training = importlib.import_module("trial_conversion_model.train")
    tracking = MagicMock()
    tracking.start_run.return_value.__enter__.return_value.info.run_id = "candidate"
    tracking.xgboost.log_model.return_value.model_uri = "models:/candidate"
    tracking.register_model.return_value.version = "2"
    monkeypatch.setattr(training, "mlflow", tracking)
    client = MagicMock()
    client.get_registered_model.side_effect = MlflowException(
        "No registered model", error_code=RESOURCE_DOES_NOT_EXIST
    )
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: client)
    s3 = MagicMock()
    monkeypatch.setattr(training.boto3, "client", lambda service: s3)
    table = pd.DataFrame({"feature": range(40), training.TARGET: [0, 1] * 20})
    monkeypatch.setattr(training, "load_processed", lambda: table)
    monkeypatch.setattr(training, "data_version", lambda: "test-data")
    model = MagicMock()
    model.predict_proba.side_effect = lambda X: np.tile([0.2, 0.8], (len(X), 1))
    model.save_model.side_effect = lambda path: path.write_text("candidate model")
    monkeypatch.setattr(training, "XGBClassifier", lambda **kwargs: model)
    monkeypatch.setattr(training, "roc_auc_score", lambda *args: 0.83)
    return SimpleNamespace(training=training, tracking=tracking, client=client, s3=s3)


def save_incumbent(path, auc):
    (path / "model.json").write_text("incumbent model")
    (path / "metrics.json").write_text(json.dumps({"test_auc": auc}))


def registered_incumbent(workflow, auc, alias=True):
    workflow.client.get_registered_model.side_effect = None
    version = SimpleNamespace(version="1", run_id="incumbent")
    workflow.client.get_registered_model.return_value = SimpleNamespace(
        aliases={"champion": "1"} if alias else {}, latest_versions=[version]
    )
    workflow.client.get_model_version_by_alias.return_value = version
    workflow.client.get_run.return_value.data.metrics = {"test_auc": auc}


@pytest.mark.parametrize("previous_auc,promoted", [
    (None, True), (0.80, True), (0.79, True), (0.800001, False),
    (0.82, False), (0.83, False), (0.90, False),
])
def test_local_promotion_gate(workflow, tmp_path, previous_auc, promoted):
    if previous_auc is not None:
        save_incumbent(tmp_path, previous_auc)
    result = workflow.training.train_and_register(tmp_path)
    assert result["promoted"] is promoted
    assert result["previous_auc"] == previous_auc
    workflow.tracking.log_metric.assert_any_call("test_auc", 0.83)
    if promoted:
        assert (tmp_path / "model.json").read_text() == "candidate model"
        assert json.loads((tmp_path / "metrics.json").read_text())["test_auc"] == 0.83
        workflow.tracking.register_model.assert_called_once_with(
            "models:/candidate", "trial_conversion_model"
        )
        workflow.client.set_registered_model_alias.assert_called_once_with(
            "trial_conversion_model", "champion", "2"
        )
        assert [call.kwargs["Key"] for call in workflow.s3.upload_file.call_args_list] == [
            "models/model.json", "models/metrics.json"
        ]
    else:
        assert (tmp_path / "model.json").read_text() == "incumbent model"
        assert json.loads((tmp_path / "metrics.json").read_text())["test_auc"] == previous_auc
        workflow.tracking.xgboost.log_model.assert_not_called()
        workflow.tracking.register_model.assert_not_called()
        workflow.s3.upload_file.assert_not_called()
        workflow.client.set_registered_model_alias.assert_not_called()


@pytest.mark.parametrize("alias", [True, False])
@pytest.mark.parametrize("auc,promoted", [(0.80, True), (0.81, False)])
def test_registered_baseline(workflow, tmp_path, alias, auc, promoted):
    registered_incumbent(workflow, auc, alias)
    result = workflow.training.train_and_register(tmp_path)
    assert result["previous_auc"] == auc
    assert result["promoted"] is promoted


@pytest.mark.parametrize("local,remote", [(0.82, 0.78), (0.78, 0.82)])
def test_stronger_baseline_wins(workflow, tmp_path, local, remote):
    save_incumbent(tmp_path, local)
    registered_incumbent(workflow, remote)
    assert not workflow.training.train_and_register(tmp_path)["promoted"]
    workflow.s3.upload_file.assert_not_called()


@pytest.mark.parametrize("auc", [None, float("nan"), 1.1, "0.8", True])
def test_invalid_incumbent_fails_closed(workflow, tmp_path, auc):
    registered_incumbent(workflow, auc)
    with pytest.raises(ValueError, match="valid test_auc"):
        workflow.training.train_and_register(tmp_path)
    workflow.tracking.register_model.assert_not_called()


def test_registry_failure_does_not_bootstrap(workflow, tmp_path):
    workflow.client.get_registered_model.side_effect = MlflowException(
        "Unavailable", error_code=INTERNAL_ERROR
    )
    with pytest.raises(MlflowException, match="Unavailable"):
        workflow.training.train_and_register(tmp_path)
    workflow.s3.upload_file.assert_not_called()


def test_missing_local_metrics_fails_closed(workflow, tmp_path):
    (tmp_path / "model.json").write_text("incumbent model")
    with pytest.raises(ValueError, match="metrics.json is missing"):
        workflow.training.train_and_register(tmp_path)


def test_upload_failure_preserves_local_champion_and_alias(workflow, tmp_path):
    save_incumbent(tmp_path, 0.79)
    workflow.s3.upload_file.side_effect = RuntimeError("Upload failed")
    with pytest.raises(RuntimeError, match="Upload failed"):
        workflow.training.train_and_register(tmp_path)
    assert (tmp_path / "model.json").read_text() == "incumbent model"
    assert json.loads((tmp_path / "metrics.json").read_text())["test_auc"] == 0.79
    workflow.client.set_registered_model_alias.assert_not_called()
    workflow.tracking.set_tag.assert_called_with("promotion_status", "failed")
