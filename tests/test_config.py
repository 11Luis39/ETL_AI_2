from etl.config import DatabaseSettings


def test_url_de_sqlalchemy_escapa_credenciales(monkeypatch):
    monkeypatch.setenv("CRM_HOST", "db.example")
    monkeypatch.setenv("CRM_PORT", "3306")
    monkeypatch.setenv("CRM_DATABASE", "crm")
    monkeypatch.setenv("CRM_USERNAME", "usuario@empresa")
    monkeypatch.setenv("CRM_PASSWORD", "clave/con:signos")

    settings = DatabaseSettings.from_env("CRM", "mysql+pymysql", 3306)
    url = settings.sqlalchemy_url().render_as_string(hide_password=False)

    assert "usuario%40empresa" in url
    assert "clave%2Fcon%3Asignos" in url
