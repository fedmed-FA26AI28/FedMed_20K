import argparse
import getpass
import json
import os
import sys
import uuid
from pathlib import Path
from portal.common import ROOT, atomic_json, password_hash


def main():
    parser = argparse.ArgumentParser(description='FedMedAI LAN portal')
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init', help='Create a local instance and operator account')
    init.add_argument('--state-dir', default='.portal')
    init.add_argument('--mode', choices=['coordinator','institution'], required=True)
    init.add_argument('--name', default='FedMedAI Research Network')
    init.add_argument('--username', default='admin')
    init.add_argument('--trainer-python', default=sys.executable)
    init.add_argument('--data-root', default=str(Path.home()/'.medmnist'))
    init.add_argument('--advertise-host', default='127.0.0.1')
    init.add_argument('--port', type=int)
    init.add_argument('--flower-port', type=int, default=8080)
    init.add_argument('--device-type', choices=['pc','jetson_nano','jetson_orin'], default='pc')
    serve = sub.add_parser('serve', help='Serve the built website and supervise local workers')
    serve.add_argument('--state-dir', default='.portal')
    serve.add_argument('--host', default='0.0.0.0')
    restart = sub.add_parser('restart', help='Restart an idle local portal; refuses to interrupt active work')
    restart.add_argument('--state-dir', default='.portal')
    restart.add_argument('--host', default='0.0.0.0')
    export = sub.add_parser('export-openapi')
    export.add_argument('--output', default='web/openapi.json')
    args = parser.parse_args()
    if args.command == 'init':
        state = Path(args.state_dir).resolve()
        if (state/'settings.json').exists():
            parser.error('This instance already exists. Use a different state directory.')
        password = os.environ.get('FEDMED_INITIAL_PASSWORD') or getpass.getpass('Create operator password (at least 10 characters): ')
        if len(password) < 10:
            parser.error('Password must contain at least 10 characters.')
        settings = vars(args).copy()
        for key in ('command','state_dir'):
            settings.pop(key)
        settings.update(id=uuid.uuid4().hex, password_hash=password_hash(password),
                        port=args.port or (8000 if args.mode == 'coordinator' else 8001),
                        trainer_python=str(Path(args.trainer_python).resolve()), data_root=str(Path(args.data_root).expanduser().resolve()),
                        artifact_roots=[str(ROOT/'results'), str(ROOT.parent/'results')])
        atomic_json(state/'settings.json', settings)
        if os.name != 'nt':
            os.chmod(state/'settings.json', 0o600)
        print(f'Created {args.mode} instance: {state}\nStart with: python -m portal serve --state-dir "{state}"')
    elif args.command in ('serve', 'restart'):
        import uvicorn
        from portal.app import create_app
        state = Path(args.state_dir).resolve()
        if not (state/'settings.json').exists():
            parser.error('Run python -m portal init first.')
        if args.command == 'restart':
            from portal.restart import stop_idle_portal
            try:
                stop_idle_portal(state)
            except ValueError as exc:
                parser.error(str(exc))
        config = json.loads((state/'settings.json').read_text())
        uvicorn.run(create_app(state), host=args.host, port=config['port'], workers=1, access_log=False)
    else:
        import tempfile
        from portal.app import create_app
        # Schema generation never starts the supervisor or training workers.
        with tempfile.TemporaryDirectory() as folder:
            atomic_json(Path(folder)/'settings.json', {'id':'schema', 'mode':'coordinator','name':'FedMedAI'})
            app = create_app(folder, start_service=False)
            atomic_json(args.output, app.openapi())
        print(f'OpenAPI written to {args.output}')


if __name__ == '__main__':
    main()
