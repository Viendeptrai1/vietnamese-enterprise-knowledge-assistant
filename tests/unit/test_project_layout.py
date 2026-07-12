from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_new_package_and_legacy_tree_exist():
    assert (ROOT / "src/knowledge_assistant/__init__.py").exists()
    assert (ROOT / "legacy/llm-engineers-handbook").is_dir()
    assert not (ROOT / "llm_engineering").exists()
