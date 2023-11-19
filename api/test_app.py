from app import app
import pytest

def test_get_root():
    response = app.get("/")
    assert response.status_code == 200
    assert response.get_data() == b"<p>Hello, World!</p>"

def test_post_root():
    suffix = [[0, 0, 5, 13, 9, 1, 0, 0], [0, 0, 13, 15, 10, 15, 5, 0], [0, 3, 15, 2, 0, 11, 8, 0], [0, 4, 12, 0, 0, 8, 8, 0], [0, 5, 8, 0, 0, 9, 8, 0], [0, 4, 11, 0, 1, 12, 7, 0], [0, 2, 14, 5, 10, 12, 0, 0], [0, 0, 6, 13, 10, 0, 0, 0]]
    response = app.post("/predict", json={"X":suffix})
    assert response.status_code == 200    
    assert response.get_data() == b'Predicted value: 0\n'

    suffix = [[0, 0, 0, 12, 13, 5, 0, 0], [0, 0, 0, 11, 16, 9, 0, 0], [0, 0, 3, 15, 16, 6, 0, 0], [0, 7, 15, 16, 16, 2, 0, 0], [0, 0, 1, 16, 16, 3, 0, 0], [0, 0, 1, 16, 16, 6, 0, 0], [0, 0, 1, 16, 16, 6, 0, 0], [0, 0, 0, 11, 16, 10, 0, 0]]
    response = app.post("/predict", json={"X":suffix})
    assert response.status_code == 200    
    assert response.get_data() == b'Predicted value: 1\n'

    suffix = [[0, 0, 0, 4, 15, 12, 0, 0], [0, 0, 3, 16, 15, 14, 0, 0], [0, 0, 8, 13, 8, 16, 0, 0], [0, 0, 1, 6, 15, 11, 0, 0], [0, 1, 8, 13, 15, 1, 0, 0], [0, 9, 16, 16, 5, 0, 0, 0], [0, 3, 13, 16, 16, 11, 5, 0], [0, 0, 0, 3, 11, 16, 9, 0]]
    response = app.post("/predict", json={"X":suffix})
    assert response.status_code == 200    
    assert response.get_data() == b'Predicted value: 2\n'

    suffix = [[0, 0, 7, 15, 13, 1, 0, 0], [0, 8, 13, 6, 15, 4, 0, 0], [0, 2, 1, 13, 13, 0, 0, 0], [0, 0, 2, 15, 11, 1, 0, 0], [0, 0, 0, 1, 12, 12, 1, 0], [0, 0, 0, 0, 1, 10, 8, 0], [0, 0, 8, 4, 5, 14, 9, 0], [0, 0, 7, 13, 13, 9, 0, 0]]
    response = app.post("/predict", json={"X":suffix})
    assert response.status_code == 200    
    assert response.get_data() == b'Predicted value: 3\n'

    suffix = [[0, 0, 0, 1, 11, 0, 0, 0], [0, 0, 0, 7, 8, 0, 0, 0], [0, 0, 1, 13, 6, 2, 2, 0], [0, 0, 7, 15, 0, 9, 8, 0], [0, 5, 16, 10, 0, 16, 6, 0], [0, 4, 15, 16, 13, 16, 1, 0], [0, 0, 0, 3, 15, 10, 0, 0], [0, 0, 0, 2, 16, 4, 0, 0]]
    response = app.post("/predict", json={"X":suffix})
    assert response.status_code == 200    
    assert response.get_data() == b'Predicted value: 4\n'

    suffix = [[0, 0, 12, 10, 0, 0, 0, 0], [0, 0, 14, 16, 16, 14, 0, 0], [0, 0, 13, 16, 15, 10, 1, 0], [0, 0, 11, 16, 16, 7, 0, 0], [0, 0, 0, 4, 7, 16, 7, 0], [0, 0, 0, 0, 4, 16, 9, 0], [0, 0, 5, 4, 12, 16, 4, 0], [0, 0, 9, 16, 16, 10, 0, 0]]
    response = app.post("/predict", json={"X":suffix})
    assert response.status_code == 200    
    assert response.get_data() == b'Predicted value: 5\n'

    suffix = [[0, 0, 0, 12, 13, 0, 0, 0], [0, 0, 5, 16, 8, 0, 0, 0], [0, 0, 13, 16, 3, 0, 0, 0], [0, 0, 14, 13, 0, 0, 0, 0], [0, 0, 15, 12, 7, 2, 0, 0], [0, 0, 13, 16, 13, 16, 3, 0], [0, 0, 7, 16, 11, 15, 8, 0], [0, 0, 1, 9, 15, 11, 3, 0]]
    response = app.post("/predict", json={"X":suffix})
    assert response.status_code == 200    
    assert response.get_data() == b'Predicted value: 6\n'

    suffix = [[0, 0, 7, 8, 13, 16, 15, 1], [0, 0, 7, 7, 4, 11, 12, 0], [0, 0, 0, 0, 8, 13, 1, 0], [0, 4, 8, 8, 15, 15, 6, 0], [0, 2, 11, 15, 15, 4, 0, 0], [0, 0, 0, 16, 5, 0, 0, 0], [0, 0, 9, 15, 1, 0, 0, 0], [0, 0, 13, 5, 0, 0, 0, 0]]
    response = app.post("/predict", json={"X":suffix})
    assert response.status_code == 200    
    assert response.get_data() == b'Predicted value: 7\n'

    suffix = [[0, 0, 9, 14, 8, 1, 0, 0], [0, 0, 12, 14, 14, 12, 0, 0], [0, 0, 9, 10, 0, 15, 4, 0], [0, 0, 3, 16, 12, 14, 2, 0], [0, 0, 4, 16, 16, 2, 0, 0], [0, 3, 16, 8, 10, 13, 2, 0], [0, 1, 15, 1, 3, 16, 8, 0], [0, 0, 11, 16, 15, 11, 1, 0]]
    response = app.post("/predict", json={"X":suffix})
    assert response.status_code == 200    
    assert response.get_data() == b'Predicted value: 8\n'

    suffix = [[0, 0, 11, 12, 0, 0, 0, 0], [0, 2, 16, 16, 16, 13, 0, 0], [0, 3, 16, 12, 10, 14, 0, 0], [0, 1, 16, 1, 12, 15, 0, 0], [0, 0, 13, 16, 9, 15, 2, 0], [0, 0, 0, 3, 0, 9, 11, 0], [0, 0, 0, 0, 9, 15, 4, 0], [0, 0, 9, 12, 13, 3, 0, 0]]
    response = app.post("/predict", json={"X":suffix})
    assert response.status_code == 200    
    assert response.get_data() == b'Predicted value: 9\n'

    