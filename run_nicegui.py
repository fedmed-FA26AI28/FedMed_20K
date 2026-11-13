"""Compatibility launcher for the dashboard on localhost:8088."""
from app_nicegui import launch


if __name__ == '__main__':
    launch(host='127.0.0.1', port=8088)
