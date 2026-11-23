from fastapi.testclient import TestClient

import app.main as main

GOOD = {
    "worker_exp_yr": 20, "rain_mm_week": 150, "slope_deg": 10, "temp_c": 22.0,
    "bush_age_yr": 40, "hours_worked": 7.5, "flush": "monsoon", "section": "Lower Garden",
}


class FakeModel:
    """Stands in for the real model so tests need no model file."""
    def predict(self, X):
        return [33.3] * len(X)


client = TestClient(main.app)


def setup_function():
    main.MODEL = FakeModel()


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_a_number():
    response = client.post("/predict", json=GOOD)
    assert response.status_code == 200
    assert response.json()["prediction"] == 33.3


def test_worker_experience_above_55_is_rejected():
    assert client.post("/predict", json={**GOOD, "worker_exp_yr": 56}).status_code == 422
    assert client.post("/predict", json={**GOOD, "worker_exp_yr": 55}).status_code == 200


def test_bush_age_below_1_5_is_rejected():
    assert client.post("/predict", json={**GOOD, "bush_age_yr": 1.4}).status_code == 422
    assert client.post("/predict", json={**GOOD, "bush_age_yr": 1.5}).status_code == 200


def test_missing_field_is_rejected():
    bad = {k: v for k, v in GOOD.items() if k != "temp_c"}
    assert client.post("/predict", json=bad).status_code == 422
