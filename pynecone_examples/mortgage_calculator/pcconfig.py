import pynecone as pc

config = pc.Config(
    app_name="mortgage_calculator",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)
