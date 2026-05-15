import json
from pathlib import Path

from bandit_stor.data.open_bandit import load_tiny_fixture
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
