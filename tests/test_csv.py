import pytest

from reconstruction.io.csv import read_circular_landmarks


def test_read_circular_landmarks(tmp_path):
    path = tmp_path / "landmarks.csv"
    path.write_text(
        "name,x,y,z,circular\n"
        "pivot,0,0,0,false\n"
        "bearing,1,2,3,true\n"
        "mount,4,5,6,\n",
        encoding="utf-8",
    )

    assert read_circular_landmarks(path) == {"bearing"}


def test_read_circular_landmarks_without_column(tmp_path):
    path = tmp_path / "landmarks.csv"
    path.write_text("name,x,y,z\npivot,0,0,0\n", encoding="utf-8")

    assert read_circular_landmarks(path) == set()


def test_read_circular_landmarks_rejects_invalid_value(tmp_path):
    path = tmp_path / "landmarks.csv"
    path.write_text(
        "name,x,y,z,circular\npivot,0,0,0,sometimes\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid circular value for pivot"):
        read_circular_landmarks(path)