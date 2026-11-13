"""python app_nicegui.py — FedMedAI LAN orchestration console."""
import argparse
import asyncio
import sys


def launch(host='0.0.0.0', port=8080):
    parser = argparse.ArgumentParser(description='FedMedAI NiceGUI dashboard')
    parser.add_argument('--host', default=host)
    parser.add_argument('--port', type=int, default=port)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    from dashboard.app import main
    main(host=args.host, port=args.port, show=not args.no_browser)


if __name__ == '__main__':
    launch()
