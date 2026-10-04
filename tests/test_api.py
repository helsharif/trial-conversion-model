from fastapi.testclient import TestClient

from trial_conversion_model.api.main import app

client = TestClient(app)

VALID_TRIAL = {
    "sessions_day1": 3,
    "sessions_day2": 2,
    "sessions_day3": 2,
    "listen_sessions_3d": 5,
    "total_minutes_3d": 180.0,
    "country": "US",
    "device_type": "iOS",
}

TRIAL_MISSING_FIELD = {
    "sessions_day1": 3,
    "sessions_day2": 2,
    "sessions_day3": 2,
    "listen_sessions_3d": 5,
    "total_minutes_3d": 180.0,
    "country": "US",
} # device_type is missing, so this should be rejected with a 422.


# Written for you, as the worked example. The two below follow the same
# shape: build a request, send it, state what should be true. Note that
# this one checks the status key rather than the whole response body, so
# that adding a field later does not fail a test that never meant to
# promise anything about it.
def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_a_probability_and_a_band():
    # post VALID_TRIAL to /predict and assert three things: 
    # 1. that the request succeeded, 
    # 2. that the conversion probability (conv_prob) sits between 0 and
    # 3. that the band (conv_band) is one of low, medium or high.
    response = client.post("/predict", json=VALID_TRIAL)
    body = response.json()
    assert response.status_code == 200, f"Expected 200 OK, but instead got {response.status_code}"
    assert 0.0 <= body["conv_prob"] <= 1.0, f"Expected conv_prob to be between 0 and 1, but got {body['conv_prob']}"
    assert body["conv_band"] in ["low", "medium", "high"], f"Expected conv_band to be one of 'low', 'medium', or 'high', but got {body['conv_band']}"


def test_predict_rejects_a_request_missing_a_field():
    # post TRIAL_MISSING_FIELD to /predict and assert the
    # service turns it down with a 422. It does that only because your
    # schema declares the field required, so this test is what notices if
    # that ever quietly changes.
    response = client.post("/predict", json=TRIAL_MISSING_FIELD)
    body = response.json()
    assert response.status_code == 422, f"Expected 422 Unprocessable Entity, but instead got {response.status_code}"

def test_valid_trial_retuns_expected_response_structure():
    # post VALID_TRIAL to /predict and assert that the response body has
    # (at least) the fields conv_prob and conv_band. This is
    # a test of your schema, not of the model, so it should not care what
    # the values are, only that they exist and are the right type.
    # Send a valid trial to /predict.
    response = client.post("/predict", json=VALID_TRIAL)
    body = response.json()

    # Verify the required response fields are present.
    required_keys = {"conv_prob", "conv_band"}
    assert required_keys.issubset(body.keys()), (
        f"Expected response to include {required_keys}, but got {body.keys()}"
    )

    # Verify each required field has the expected type.
    assert isinstance(body["conv_prob"], float), (
        f"Expected conv_prob to be a float, but got {type(body['conv_prob'])}"
    )
    assert isinstance(body["conv_band"], str), (
        f"Expected conv_band to be a string, but got {type(body['conv_band'])}"
    )