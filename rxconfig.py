import reflex as rx
import os

config = rx.Config(
    app_name="DOCU_AI",
    backend_port=int(os.environ.get("PORT", 8000)),
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)
