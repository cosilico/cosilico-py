from pathlib import Path

from typer.testing import CliRunner

from skeleton_tool.main import app, app_dir

runner = CliRunner()


def test_login():
    result = runner.invoke(app, ["login", "--username", "test", "--password", "slkdfj"])
    assert result.exit_code == 0
    assert "Logged" in result.stdout
    assert "test" in result.stdout

def test_config():
    result = runner.invoke(app, ["config", "--username", "test", "--password", "slkdfj", "--organization", "myorg", "--make-default"])
    assert result.exit_code == 0

    config_path: Path = Path(app_dir) / "cosilico.config"
    assert config_path.is_file()

    text = config_path.read_text()
    assert "[myorg]" in text
    assert "[default]" in text
    assert "username = " in text