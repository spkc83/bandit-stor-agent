import json
from pathlib import Path

from bandit_stor.data.open_bandit import load_open_bandit, load_tiny_fixture
from bandit_stor.training.full_pipeline import run_full_pipeline
from bandit_stor.utils import load_yaml


def test_tiny_fixture_loads_and_splits_are_deterministic():
    cfg = load_yaml("configs/data/tiny_fixture.yaml")
    dataset1, splits1 = load_tiny_fixture(cfg)
    dataset2, splits2 = load_tiny_fixture(cfg)
    assert len(dataset1) == 10
    assert dataset1.context_dim == 2
    assert dataset1.n_actions == 5
    row = dataset1[0]
    assert row["context"].shape == (2,)
    assert row["candidate_actions"].shape == (5,)
    assert row["action_context"].shape[0] == 5
    assert row["action_context"].shape[1] > 1
    assert row["pscore"] > 0
    assert [s.indices for s in (splits1.train, splits1.valid, splits1.test)] == [
        s.indices for s in (splits2.train, splits2.valid, splits2.test)
    ]


def test_full_pipeline_tiny_fixture_writes_required_artifacts(tmp_path: Path):
    run_dir = run_full_pipeline(data="tiny_fixture", output_dir=tmp_path)
    assert (run_dir / "actor.pt").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "ope_report.json").exists()
    assert (run_dir / "policy_report.md").exists()
    ope = json.loads((run_dir / "ope_report.json").read_text())
    assert {"ips", "snips", "doubly_robust", "effective_sample_size", "unsupported_action_mass"} <= set(
        ope["metrics"]
    )



def test_open_bandit_full_download_extracts_official_layout(tmp_path: Path):
    import zipfile

    archive = tmp_path / "source_obd.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("random/all/all.csv", "id,timestamp,item_id,position,click,propensity_score,user_feature_0\n0,1,0,1,1,0.5,a\n")
        zf.writestr("random/all/item_context.csv", "item_id,item_feature_0,item_feature_1\n0,0.1,x\n")

    from bandit_stor.data.open_bandit import download_open_bandit_full_dataset

    data_path = tmp_path / "open_bandit"
    download_open_bandit_full_dataset(
        {
            "data_path": str(data_path),
            "download_url": archive.as_uri(),
            "archive_path": str(tmp_path / "downloaded.zip"),
        }
    )
    assert (data_path / "random" / "all" / "all.csv").exists()
    assert (data_path / "random" / "all" / "item_context.csv").exists()


def test_open_bandit_obp_layout_loads_with_polars(tmp_path: Path):
    root = tmp_path / "open_bandit" / "random" / "all"
    root.mkdir(parents=True)
    (root / "all.csv").write_text(
        "\n".join(
            [
                ",timestamp,item_id,position,click,propensity_score,user_feature_0,user_feature_1",
                "0,2020-01-01T00:00:03Z,0,1,1,0.5,a,x",
                "1,2020-01-01T00:00:01Z,1,2,0,0.25,b,x",
                "2,2020-01-01T00:00:02Z,2,1,1,0.25,a,y",
                "3,2020-01-01T00:00:04Z,1,2,0,0.25,b,y",
            ]
        )
        + "\n"
    )
    (root / "item_context.csv").write_text(
        "\n".join(
            [
                ",item_id,item_feature_0,item_feature_1",
                "0,0,0.1,red",
                "1,1,0.2,blue",
                "2,2,0.3,red",
            ]
        )
        + "\n"
    )

    dataset, splits = load_open_bandit(
        {
            "data_path": str(tmp_path / "open_bandit"),
            "download": False,
            "behavior_policy": "random",
            "campaign": "all",
            "context_encoding": "categorical_codes",
            "split": {"strategy": "chronological", "valid_size": 0.25, "test_size": 0.25},
        }
    )

    assert len(dataset) == 4
    assert dataset.context_dim == 2
    assert dataset.n_actions == 3
    assert len(splits.train) == 2
    assert dataset[0]["logged_action_index"] == 1
    assert dataset[0]["position"] == 1
    assert dataset[0]["pscore"] == 0.25
    assert dataset[0]["action_context"].shape[0] == 3
