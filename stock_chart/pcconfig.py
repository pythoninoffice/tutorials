import pynecone as pc

config = pc.Config(
    app_name="stock_chart",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)
