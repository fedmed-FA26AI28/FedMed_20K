"""Reactive NiceGUI dashboard; all process state is shared, view state is per browser."""
import asyncio
import csv
import io
import time

from nicegui import app, ui
import psutil
import yaml

from dashboard.controller import Controller, ROOT, ACTIVE, DATASETS, STRATEGIES, PARTITIONS, prepare_config
from dashboard.system import lan_ips, port_owners, partition_data
from dashboard.client_setup import remote_client_setup

controller = Controller()
CSS = '''
:root {--navy:#0f172a;--panel:#1e293b;--accent:#14b8a6;--border:#334155;--muted:#94a3b8;--ink:#e2e8f0;--back:#0b1220;}
body {font-family:Segoe UI,Arial,sans-serif;background:var(--back);color:var(--ink);}
body.body--light {--panel:#fff;--back:#f1f5f9;--ink:#17263c;--muted:#52647b;--border:#d6e0eb;}
.q-page,.q-tab-panels {background:transparent!important;}
.q-header {background:var(--navy)!important;border-bottom:1px solid var(--border);}
.q-card {background:var(--panel);border:1px solid var(--border);box-shadow:none;border-radius:14px;}
.shell {width:100%;max-width:1600px;margin:auto;padding:24px;gap:22px;}
.eyebrow {font-size:11px;letter-spacing:2px;font-weight:700;color:var(--accent);}
.muted {font-size:13px;color:var(--muted);}
.kpis {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;width:100%;}
.summary-kpis{grid-template-columns:repeat(3,minmax(0,1fr));}
.network-mini{gap:0;padding:6px 12px;border:1px solid var(--border);border-radius:8px;}
.network-mini .eyebrow{font-size:9px;letter-spacing:1px;}
.kpi-number {font-size:30px;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-1px;}
.two-col {display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:20px;width:100%;align-items:start;}
.fields {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;width:100%;}
.q-field {min-width:0;}.q-table__container {max-width:100%;background:var(--panel);}
.terminal {background:#080f1d;color:#b9e6e0;font:12px/1.65 Consolas,monospace;border:1px solid var(--border);border-radius:10px;}
.q-tab {text-transform:none;min-height:56px;}.q-tab-panels {width:100%;}.q-tab-panel {padding:20px 0;}
.q-btn:focus-visible {outline:2px solid var(--accent);outline-offset:3px;}
.teal {color:var(--accent);}
@media(max-width:1050px){.two-col{grid-template-columns:1fr}.shell{padding:16px}}
@media(max-width:650px){.kpis{grid-template-columns:1fr 1fr}.fields{grid-template-columns:1fr}.shell{padding:12px}.kpi-number{font-size:24px}}
'''


def options(names, percent=False):
    axis = {'type': 'value', 'splitLine': {'lineStyle': {'color': '#334155', 'opacity': 0.4}}}
    if percent:
        axis.update(min=0, max=100, axisLabel={'formatter': '{value}%'})
    return {'animation': False, 'color': ['#14b8a6', '#818cf8', '#fbbf24', '#38bdf8', '#f472b6'],
            'textStyle': {'color': '#94a3b8'}, 'tooltip': {'trigger': 'axis'},
            'legend': {'bottom': 0, 'textStyle': {'color': '#94a3b8'}},
            'grid': {'left': 52, 'right': 25, 'top': 25, 'bottom': 65},
            'xAxis': {'type': 'category', 'name': 'Round', 'data': []}, 'yAxis': axis,
            'series': [{'name': n, 'type': 'line', 'showSymbol': False, 'connectNulls': False, 'data': []} for n in names]}


def table(columns, key='id'):
    return ui.table(columns=[{'name': k, 'label': label, 'field': k, 'sortable': True, 'align': 'left'}
                             for k, label in columns], rows=[], row_key=key, pagination=10).classes('w-full')


def display(value, suffix='', digits=1):
    return '—' if value is None else f'{value:.{digits}f}{suffix}'


@ui.page('/')
def index():
    ui.add_css(CSS)
    ui.colors(primary='#0d9488', secondary='#6366f1', positive='#10b981', negative='#ef4444')
    dark = ui.dark_mode(True)
    server_fields = {}
    state = {'cursor': 0, 'cleared': 0, 'signature': None, 'revision': None, 'visible': []}

    def select(fields, key, label, choices, value):
        fields[key] = ui.select(choices, value=value, label=label).props('outlined dense').classes('w-full')
        return fields[key]

    def number(fields, key, label, value, minimum=0, maximum=None, step=1):
        fields[key] = ui.number(label, value=value, min=minimum, max=maximum, step=step).props('outlined dense').classes('w-full')
        return fields[key]

    def values(fields):
        return {key: field.value for key, field in fields.items()}

    def selected_job():
        return controller.jobs.get(job_select.value)

    with ui.header().classes('items-center px-5 py-3 gap-3'):
        ui.icon('health_and_safety', size='32px').classes('teal')
        with ui.column().classes('gap-0'):
            ui.label('FedMedAI Orchestrator').classes('text-lg font-bold')
            ui.label('DISTRIBUTED MEDICAL INTELLIGENCE').classes('text-xs tracking-widest text-slate-400')
        server_badge = ui.badge('SERVER · IDLE', color='secondary').classes('ml-4')
        ui.space()
        with ui.column().classes('network-mini'):
            ui.label('PORT & NETWORK').classes('eyebrow')
            port_value = ui.label('gRPC :8081').classes('text-sm font-semibold')
            port_status = ui.label('Checking…').classes('text-xs text-slate-300')
            web_address = ui.label(f'WEB · :{controller.web_port}').classes('text-xs text-slate-400')
        ui.button(icon='contrast', on_click=dark.toggle).props('flat round color=white aria-label="Dark / Light"').tooltip('Dark / Light')
    with ui.column().classes('shell'):
        with ui.row().classes('w-full items-end justify-between gap-4'):
            with ui.column().classes('gap-1'):
                ui.label('FEDERATED LEARNING / OPERATIONS').classes('eyebrow')
            job_select = ui.select({}, label='Monitored run', on_change=lambda: reset_view()).props('outlined dense').classes('w-80 max-w-full')
        with ui.element('div').classes('kpis summary-kpis'):
            kpi_labels, kpi_hints = [], []
            for title, icon in [('FL PROGRESS', 'sync'), ('BEST TEST ACCURACY', 'verified'), ('GLOBAL LOSS', 'show_chart')]:
                with ui.card().classes('p-5 w-full gap-2'):
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.label(title).classes('eyebrow')
                        ui.icon(icon, size='20px').classes('muted')
                    kpi_labels.append(ui.label('—').classes('kpi-number'))
                    kpi_hints.append(ui.label('Waiting for data').classes('muted'))
                    if title == 'FL PROGRESS':
                        round_progress = ui.linear_progress(0, show_value=False)
        with ui.tabs().classes('w-full border-b border-slate-700').props('active-color=primary indicator-color=primary align=left') as tabs:
            server_tab = ui.tab('FL Server', icon='dns')
            client_tab = ui.tab('Client Nodes', icon='devices')
            analytics_tab = ui.tab('Live Analytics', icon='insights')
            logs_tab = ui.tab('Live Terminal', icon='terminal')
            telemetry_tab = ui.tab('Cluster & Data', icon='monitor_heart')
        with ui.tab_panels(tabs, value=server_tab).classes('w-full').props('transition-duration=0'):
            with ui.tab_panel(server_tab):
                with ui.element('div').classes('two-col'):
                    with ui.card().classes('w-full p-6 gap-5'):
                        ui.label('01 / Server configuration').classes('eyebrow')
                        ui.label('Federation configuration').classes('text-xl font-semibold')
                        mode = ui.toggle({'server': 'LAN Server', 'simulation': 'Local Simulation'}, value='server')
                        with ui.element('div').classes('fields'):
                            host = select(server_fields, 'server_host', 'Bind IPv4', ['0.0.0.0', '127.0.0.1'], '0.0.0.0')
                            number(server_fields, 'server_port', 'gRPC Port', 8081, 1, 65535)
                            select(server_fields, 'dataset', 'Dataset', DATASETS, 'pathmnist')
                            select(server_fields, 'model', 'Model architecture', {'baseline': 'Baseline', 'baseline_bn': 'BaselineBN'}, 'baseline')
                            strategy = select(server_fields, 'strategy', 'Aggregation strategy', {s: s.upper() for s in STRATEGIES if s != 'scr'}, 'fedavg')
                            mu = number(server_fields, 'proximal_mu', 'FedProx μ', 0.01, 0, step=0.001)
                            mu.bind_visibility_from(strategy, 'value', backward=lambda v: v == 'fedprox')
                            number(server_fields, 'rounds', 'Communication rounds', 40, 1, 100)
                            number(server_fields, 'num_clients', 'Total clients / partition', 2, 1, 1000)
                            number(server_fields, 'min_fit_clients', 'Min Fit Clients', 2, 1)
                            number(server_fields, 'min_available_clients', 'Min Available Clients', 2, 1)
                            number(server_fields, 'fraction_fit', 'Fraction Fit', 1.0, 0.01, 1, 0.05)
                            select(server_fields, 'partition_mode', 'Non-IID partition', PARTITIONS, 'practical')
                            number(server_fields, 'local_epochs', 'Local epochs / round', 5, 1)
                            number(server_fields, 'batch_size', 'Batch size', 16, 1)
                            number(server_fields, 'learning_rate', 'Learning rate', 0.01, 0.000001, step=0.001)
                            number(server_fields, 'seed', 'Partition seed', 42, 0)
                            select(server_fields, 'optimizer', 'Optimizer', ['sgd', 'adam', 'adamw'], 'sgd')
                            server_device = select(server_fields, 'device', 'Compute device', ['auto', 'cpu'], 'auto')
                        with ui.row().classes('gap-4'):
                            server_fields['local_test'] = ui.switch('Client local test', value=True)
                            server_fields['server_eval'] = ui.switch('Central test', value=False)
                        for key in ('min_fit_clients', 'min_available_clients', 'fraction_fit', 'server_port', 'server_host'):
                            server_fields[key].bind_enabled_from(mode, 'value', backward=lambda v: v == 'server')
                        ui.label('Local Simulation uses every client each round and selects an available port automatically.').classes('muted')
                        with ui.row().classes('w-full gap-3'):
                            start_server = ui.button('Start FL Server', icon='play_arrow', on_click=lambda: launch_server())
                            stop_server = ui.button('Stop Server', icon='stop', on_click=lambda: stop_servers()).props('outline color=negative')
                        ui.button('Export server YAML', icon='download', on_click=lambda: export_yaml()).props('flat dense')
                    with ui.column().classes('w-full gap-5'):
                        with ui.card().classes('p-6 w-full gap-4'):
                            ui.label('02 / Execution monitor').classes('eyebrow')
                            session_title = ui.label('Ready to start a run').classes('text-xl font-semibold')
                            session_text = ui.label('Start the FL server to accept PC and Jetson clients over your LAN.').classes('muted break-all')
                        with ui.card().classes('p-5 w-full'):
                            ui.label('Managed processes').classes('font-semibold')
                            jobs_table = table([('mode', 'Role'), ('id', 'Run'), ('pid', 'PID'), ('state', 'Status'), ('exit', 'Exit')])
            with ui.tab_panel(client_tab):
                with ui.card().classes('p-4 w-full gap-3'):
                    with ui.row().classes('w-full items-center gap-3'):
                        ui.label('Client configuration').classes('font-semibold')
                        remote_address = ui.input('Server IP:Port', value='').props('outlined dense').classes('w-64 max-w-full')
                        remote_id = ui.number('Client ID', value=0, min=0, step=1).props('outlined dense').classes('w-32')
                        ui.button('Download YAML', icon='download', on_click=lambda: export_client_yaml()).props('dense')
                        ui.button('Copy command', icon='content_copy', on_click=lambda: copy_client_command()).props('outline dense')
                    config_source = ui.label('Configuration source: server form').classes('muted')
                with ui.element('div').classes('kpis'):
                    node_kpis = []
                    for title in ['CONFIGURED CLIENTS', 'CLIENTS OBSERVED', 'FIT REPORTS THIS ROUND', 'AWAITING FIT REPORT']:
                        with ui.card().classes('p-5 gap-2'):
                            ui.label(title).classes('eyebrow')
                            node_kpis.append(ui.label('—').classes('kpi-number'))
                with ui.card().classes('p-6 w-full gap-4'):
                    ui.label('Federation nodes').classes('text-lg font-semibold')
                    node_search = ui.input('Search nodes', placeholder='Client ID, IP, device, status…').props('outlined dense').classes('w-80 max-w-full')
                    ui.label('Clients report telemetry at the end of each round. A recent report is not a live heartbeat. A dash indicates unavailable data.').classes('muted')
                    nodes_table = table([('id', 'Client'), ('ip', 'IPv4'), ('device', 'Device'), ('state', 'Status'), ('round', 'Round'),
                                         ('samples', 'Train samples'), ('training_time', 'Fit time (s)'),
                                         ('ping', 'Ping ms'), ('vram', 'VRAM MB¹'), ('cpu', 'CPU %'), ('ram', 'RAM %'),
                                         ('train_accuracy', 'Train %'), ('test_accuracy', 'Test %'), ('train_loss', 'Train loss'), ('age', 'Last report')])
                    node_search.bind_value(nodes_table, 'filter')
                    ui.label('¹ GPU memory allocated by PyTorch in the client process.').classes('muted')
                with ui.card().classes('p-6 w-full gap-4'):
                    with ui.row().classes('w-full items-center justify-between'):
                        client_select = ui.select({}, label='Inspect client', on_change=lambda: render_client()).props('outlined dense').classes('w-72 max-w-full')
                        ui.button('Export client history', icon='download', on_click=lambda: export_client_history()).props('flat')
                    client_identity = ui.label('Select a reporting client.').classes('text-lg font-semibold')
                    client_details = {}
                    with ui.element('div').classes('fields'):
                        for key, label in [('round', 'Latest reported round'), ('last_seen', 'Last received'),
                                           ('num_samples', 'Train samples'), ('test_samples', 'Test samples'),
                                           ('epochs_run', 'Epochs completed'), ('learning_rate', 'Learning rate'),
                                           ('training_time', 'Fit time (seconds)'), ('epoch_time_avg', 'Mean epoch time (seconds)'),
                                           ('weight_size_kb', 'Model payload (KB)'), ('ram_used_mb', 'Host RAM used (MB)')]:
                            with ui.column().classes('gap-0'):
                                ui.label(label).classes('muted')
                                client_details[key] = ui.label('—').classes('text-lg font-semibold')
                    with ui.element('div').classes('two-col'):
                        with ui.column().classes('w-full'):
                            ui.label('Client accuracy · Train / Test').classes('font-semibold')
                            client_accuracy = ui.echart(options(['Train', 'Test'], True)).classes('w-full h-72')
                        with ui.column().classes('w-full'):
                            ui.label('Client loss · Train / Test').classes('font-semibold')
                            client_loss = ui.echart(options(['Train', 'Test'])).classes('w-full h-72')
                    client_history_table = table([('round', 'Round'), ('train_accuracy', 'Train acc (0–1)'),
                        ('test_accuracy', 'Test acc (0–1)'), ('train_loss', 'Train loss'), ('test_loss', 'Test loss'),
                        ('training_time', 'Fit time (s)'), ('ping_ms', 'Ping ms'), ('cpu_percent', 'CPU %'),
                        ('ram_percent', 'RAM %')], key='round')
            with ui.tab_panel(analytics_tab):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('Real-time model convergence').classes('text-xl font-semibold')
                analytics_note = ui.label('Select a run to view live metrics.').classes('muted')
                with ui.element('div').classes('two-col mt-4'):
                    with ui.card().classes('w-full p-4'):
                        ui.label('Accuracy · Train / Test / Best').classes('font-semibold')
                        accuracy = ui.echart(options(['Train', 'Test', 'BTA', 'MCTA', 'BMCTA'], True)).classes('w-full h-80')
                    with ui.card().classes('w-full p-4'):
                        ui.label('Loss · Train / Test').classes('font-semibold')
                        loss = ui.echart(options(['Train', 'Test'])).classes('w-full h-80')
                round_table = table([('round', 'Round'), ('train_accuracy', 'Train acc'), ('test_accuracy', 'Test acc'),
                                     ('bta', 'BTA'), ('bmcta', 'BMCTA'), ('train_loss', 'Train loss'), ('test_loss', 'Test loss'),
                                     ('num_clients_reporting', 'Clients')], key='round')
            with ui.tab_panel(logs_tab):
                with ui.row().classes('w-full items-center gap-3'):
                    log_source = ui.select({'selected': 'Selected run', 'all': 'All processes'}, value='selected', label='Source').props('outlined dense').classes('w-52')
                    level = ui.select(['All', 'INFO', 'WARNING', 'ERROR', 'METRIC'], value='All', label='Log level').props('outlined dense').classes('w-36')
                    search = ui.input('Search logs', placeholder='keyword').props('outlined dense').classes('grow')
                    ui.button('Clear Console', icon='clear_all', on_click=lambda: clear_console()).props('flat')
                    ui.button('Copy Logs', icon='content_copy', on_click=lambda: ui.clipboard.write('\n'.join(state['visible']))).props('flat')
                console = ui.log(max_lines=2000).classes('terminal w-full h-[560px] p-4')
                ui.label('Clear removes visible logs in this browser tab only. Log files and other tabs are preserved. Copy copies the currently displayed lines.').classes('muted')
            with ui.tab_panel(telemetry_tab):
                ui.label('Host telemetry').classes('text-xl font-semibold')
                ui.label('Sampled on the dashboard host every 3 seconds. System GPU metrics come from NVIDIA SMI or Jetson sensors.').classes('muted')
                with ui.element('div').classes('kpis my-4'):
                    hardware_labels = []
                    for name in ['CPU', 'RAM', 'GPU UTILIZATION', 'GPU TEMPERATURE']:
                        with ui.card().classes('p-5 w-full'):
                            ui.label(name).classes('eyebrow')
                            hardware_labels.append(ui.label('—').classes('kpi-number'))
                hardware_info = ui.label('Detecting hardware…').classes('muted')
                with ui.card().classes('w-full p-5 mt-4'):
                    ui.label('Non-IID data distribution').classes('text-lg font-semibold')
                    partition_select = ui.select({}, label='Partition JSON', on_change=lambda: load_partition()).props('outlined dense').classes('w-full')
                    ui.button('Scan partitions', icon='refresh', on_click=lambda: scan_partitions()).props('flat')
                    partition_note = ui.label('Choose a partition to inspect per-class sample counts for each client.').classes('muted')
                    distribution = ui.echart(options([])).classes('w-full h-96')

    def reset_view():
        state.update(signature=None, revision=None)

    def clear_console():
        state.update(cleared=controller.sequence, cursor=controller.sequence, visible=[], signature=None)
        console.clear()

    async def launch_server():
        try:
            cfg = values(server_fields)
            if mode.value == 'simulation':
                cfg['server_host'] = '127.0.0.1'
            job = await controller.start(mode.value, cfg)
            job_select.set_options({j.id: j.id for j in controller.jobs.values()}, value=job.id)
            reset_view()
            ui.notify('Run started. Open Live Terminal to follow its progress.', type='positive')
        except (ValueError, TypeError, OSError, psutil.Error) as exc:
            ui.notify(str(exc), type='negative')

    async def stop_servers():
        try:
            await asyncio.gather(*(controller.stop(j.id) for j in controller.jobs.values() if j.mode != 'client' and j.state in ACTIVE))
        except (OSError, psutil.Error) as exc:
            ui.notify(str(exc), type='negative')

    def export_yaml():
        try:
            cfg = prepare_config(mode.value, values(server_fields), controller.web_port)
            ui.download(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False).encode(), 'fedmedai-config.yaml')
        except (ValueError, TypeError) as exc:
            ui.notify(str(exc), type='negative')

    def client_setup():
        job = selected_job()
        if job and job.mode != 'server':
            raise ValueError('Select a LAN server run to configure remote clients.')
        cfg = job.config if job else prepare_config('server', values(server_fields), controller.web_port)
        return remote_client_setup(cfg, remote_address.value, remote_id.value)

    def export_client_yaml():
        try:
            cfg, _ = client_setup()
            ui.download(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False).encode(), 'client_dashboard.yaml')
        except (ValueError, TypeError) as exc:
            ui.notify(str(exc), type='negative')

    def copy_client_command():
        try:
            _, command = client_setup()
            ui.clipboard.write(command)
            ui.notify('Client command copied.', type='positive')
        except (ValueError, TypeError) as exc:
            ui.notify(str(exc), type='negative')

    async def scan_partitions():
        paths = await asyncio.to_thread(lambda: sorted((ROOT / 'data' / 'partitions').glob('*.json')))
        partition_select.set_options([str(p.relative_to(ROOT)) for p in paths])

    async def load_partition():
        if partition_select.value not in partition_select.options:
            return
        try:
            data, labels, series = await asyncio.to_thread(partition_data, ROOT / partition_select.value)
            distribution.options['xAxis'].update(data=labels, name='Client')
            distribution.options['series'] = series
            distribution.update()
            partition_note.text = f"{data.get('dataset')} · {data.get('partition_mode')} · seed {data.get('seed')} · {data.get('total_samples')} train samples"
        except (ValueError, OSError, TypeError) as exc:
            ui.notify(f'Partition: {exc}', type='negative')

    async def network_refresh():
        try:
            ips = await asyncio.to_thread(lan_ips)
            host.set_options(['0.0.0.0', *ips])
            web_address.text = f"WEB · {next((ip for ip in ips if not ip.startswith('127.')), '127.0.0.1')}:{controller.web_port}"
            active_server = next((j for j in reversed(list(controller.jobs.values())) if j.mode == 'server' and j.state in ACTIVE), None)
            cfg = active_server.config if active_server else values(server_fields)
            if not remote_address.value:
                bind = cfg['server_host']
                lan_ip = next((ip for ip in ips if not ip.startswith('127.')), '127.0.0.1')
                remote_address.value = f"{lan_ip if bind == '0.0.0.0' else bind}:{int(cfg['server_port'])}"
            port = int(cfg['server_port'])
            owners = await asyncio.to_thread(port_owners, port)
            port_value.text = f'gRPC :{port}'
            port_status.text = f"{cfg['server_host']} · {'LISTENING' if owners else 'FREE'}"
        except (ValueError, TypeError, OSError, psutil.Error) as exc:
            port_status.text = f'Unable to inspect port: {exc}'

    def refresh():
        jobs = list(controller.jobs.values())
        if list(job_select.options) != [j.id for j in jobs]:
            job_select.set_options({j.id: j.id for j in jobs}, value=job_select.value or (jobs[-1].id if jobs else None))
        job = selected_job()
        config_source.text = f"Configuration source: {job.id if job else 'server form'}"
        servers = [j for j in jobs if j.mode != 'client']
        server = servers[-1] if servers else None
        server_badge.text = 'SERVER · ' + ({'failed': 'ERROR'}.get(server.state, server.state.upper()) if server else 'IDLE')
        server_badge.props('color=' + ('negative' if server and server.state == 'failed' else 'positive' if server and server.state == 'running' else 'secondary'))
        busy = any(j.state in ACTIVE and j.mode != 'client' for j in jobs)
        start_server.set_enabled(not busy)
        stop_server.set_enabled(busy)
        job_rows = [{'id': j.id, 'mode': j.mode, 'state': j.state, 'pid': j.process.pid if j.process else '—', 'exit': j.exit_code} for j in reversed(jobs)]
        if jobs_table.rows != job_rows:
            jobs_table.rows = job_rows
            jobs_table.update()
        if job:
            session_title.text = f'{job.mode.upper()} · {job.state.upper()}'
            duration = int((job.ended or time.time()) - job.started)
            session_text.text = f"{job.folder.relative_to(ROOT)} · {duration // 60:02d}:{duration % 60:02d} · Exit {job.exit_code if job.exit_code is not None else '—'}"
        rows = job.metrics.records() if job else []
        total = job.config['rounds'] if job else 0
        current = job.metrics.round if job else 0
        kpi_labels[0].text = f'{current} / {total}' if job else '— / —'
        kpi_hints[0].text = 'Current round · selected run'
        round_progress.value = min(current / total, 1) if total else 0
        best = max((r['bta'] for r in rows if r.get('bta') is not None), default=None)
        kpi_labels[1].text = display(best * 100 if best is not None else None, '%', 2)
        kpi_hints[1].text = 'BTA · best test accuracy'
        losses = [r['test_loss'] for r in rows if r.get('test_loss') is not None]
        kpi_labels[2].text = display(losses[-1] if losses else None, digits=4)
        kpi_hints[2].text = (f"{'↓' if losses[-1] < losses[-2] else '↑' if losses[-1] > losses[-2] else '→'} Δ {losses[-1] - losses[-2]:+.4f} vs previous round" if len(losses) > 1 else 'Aggregated test loss · waiting for data')
        revision = (job.id if job else None, job.metrics.revision if job else 0)
        if state['revision'] != revision:
            for chart, keys, scale in [(accuracy, ['train_accuracy', 'test_accuracy', 'bta', 'mcta', 'bmcta'], 100),
                                       (loss, ['train_loss', 'test_loss'], 1)]:
                chart.options['xAxis']['data'] = [r['round'] for r in rows]
                for series, key in zip(chart.options['series'], keys):
                    series['data'] = [r[key] * scale if r.get(key) is not None else None for r in rows]
                chart.update()
            round_table.rows = rows
            round_table.update()
            analytics_note.text = f"{job.id if job else 'No run selected'} · {len(rows)} rounds · BMCTA = best mean client test accuracy · missing values remain blank"
            state['revision'] = revision
        node_rows = []
        if job:
            for cid, node in job.metrics.nodes.items():
                age = int(time.time() - node['last_seen'])
                node_rows.append({'id': cid, 'ip': node.get('device_ipv4', '—'),
                    'device': node.get('device_type', '—'), 'ram': display(node.get('ram_percent')),
                    'samples': node.get('num_samples', '—'), 'training_time': display(node.get('training_time')),
                    'test_accuracy': display(node.get('test_accuracy') * 100 if node.get('test_accuracy') is not None else None),
                    'state': 'Run ended' if job.state not in ACTIVE else ('Reported' if age < 60 else 'Awaiting new report'),
                    'round': node.get('round'), 'ping': display(node.get('ping_ms') if node.get('ping_ms', -1) >= 0 else None),
                    'train_accuracy': display(node.get('train_accuracy') * 100 if node.get('train_accuracy') is not None else None),
                    'train_loss': display(node.get('train_loss'), digits=4),
                    'vram': display(node.get('gpu_memory_mb')), 'cpu': display(node.get('cpu_percent')), 'age': f'{age}s'})
        nodes_table.rows = node_rows
        nodes_table.update()
        configured = job.config['num_clients'] if job else 0
        fit_reports = sum(1 for records in job.metrics.node_history.values()
                          if records.get(current, {}).get('train_loss') is not None) if job else 0
        for label, value in zip(node_kpis, [configured, len(node_rows), fit_reports, max(0, configured-fit_reports)]):
            label.text = str(value) if job else '—'
        choices = {cid: f"Client {cid} · {node.get('device_ipv4', '—')}" for cid, node in job.metrics.nodes.items()} if job else {}
        if client_select.options != choices:
            client_select.set_options(choices, value=client_select.value if client_select.value in choices else next(iter(choices), None))
        render_client()
        hw = controller.telemetry
        for label, key, suffix in zip(hardware_labels, ['cpu', 'ram', 'gpu', 'temperature'], ['%', '%', '%', '°C']):
            label.text = display(hw.get(key), suffix)
        hardware_info.text = f"{hw.get('gpu_name', 'Detecting…')} · System VRAM {display(hw.get('vram_mb'), ' MB')} · RAM {display(hw.get('ram_gb'), ' GB')}"
        devices = hw.get('devices', ['auto', 'cpu'])
        for device_field in (server_device,):
            if device_field.options != devices:
                device_field.set_options(devices)
        render_logs(job)

    def render_client():
        job = selected_job()
        node = job.metrics.nodes.get(client_select.value) if job else None
        history = job.metrics.client_records(client_select.value) if job and node else []
        client_identity.text = (f"Client {client_select.value} · {node.get('device_ipv4', '—')} · {node.get('device_type', '—')}"
                                if node else 'Select a reporting client.')
        for key, label in client_details.items():
            value = node.get(key) if node else None
            label.text = (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value)) if key == 'last_seen' and value else
                          display(value, digits=6 if key == 'learning_rate' else 2) if value is not None else '—')
        signature = (job.id if job else None, job.metrics.revision if job else 0, client_select.value)
        if state.get('client_revision') == signature:
            return
        for chart, keys, scale in [(client_accuracy, ['train_accuracy', 'test_accuracy'], 100),
                                   (client_loss, ['train_loss', 'test_loss'], 1)]:
            chart.options['xAxis']['data'] = [r['round'] for r in history]
            for series, key in zip(chart.options['series'], keys):
                series['data'] = [r[key]*scale if r.get(key) is not None else None for r in history]
            chart.update()
        client_history_table.rows = history
        client_history_table.update()
        state['client_revision'] = signature

    def export_client_history():
        job = selected_job()
        history = job.metrics.client_records(client_select.value) if job else []
        if not history:
            ui.notify('No client history is available.', type='warning')
            return
        output = io.StringIO()
        columns = sorted({key for row in history for key in row})
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
        ui.download(output.getvalue().encode('utf-8'), f'{job.id}-client-{client_select.value}.csv')

    def render_logs(job):
        signature = (job.id if job else None, log_source.value, level.value, search.value, state['cleared'])
        if signature != state['signature']:
            console.clear()
            state.update(cursor=state['cleared'], visible=[], signature=signature)
        for line in list(controller.logs):
            if line['seq'] <= state['cursor']:
                continue
            state['cursor'] = line['seq']
            if log_source.value == 'selected' and (job is None or line['job'] != job.id):
                continue
            if level.value != 'All' and line['level'] != level.value:
                continue
            if search.value.casefold() not in line['text'].casefold():
                continue
            rendered = f"{line['time']} [{line['job']}] {line['text'][:16000]}"
            console.push(rendered)
            state['visible'].append(rendered)
        state['visible'] = state['visible'][-2000:]

    ui.timer(0.7, refresh)
    ui.timer(5, network_refresh)
    ui.timer(0.1, network_refresh, once=True)
    ui.timer(0.1, scan_partitions, once=True)
    refresh()


def main(host='0.0.0.0', port=8080, show=True):
    controller.web_port = port
    app.on_startup(controller.startup)
    app.on_shutdown(controller.shutdown)
    ui.run(host=host, port=port, title='FedMedAI Orchestrator', favicon='⚕', reload=False, show=show)
