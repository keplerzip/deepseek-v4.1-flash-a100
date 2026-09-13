"""IPv4 TCP forwarding on loopback and inspected Docker bridges only."""
import argparse
import asyncio
import ipaddress
import json
from pathlib import Path
import socket
import urllib.error
import urllib.request


PRIVATE = [ipaddress.ip_network(value) for value in
           ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16')]


def private_network(value):
    network = ipaddress.ip_network(value, strict=True)
    if network.version != 4 or not any(network.subnet_of(item) for item in PRIVATE):
        raise ValueError('Expected an RFC1918 Docker IPv4 subnet: ' + value)
    return network


def configure(networks, upstream, port):
    upstream = ipaddress.IPv4Address(upstream)
    subnets, gateways = set(), set()
    for row in networks:
        if row.get('Driver') != 'bridge' or row.get('Scope', 'local') != 'local':
            continue
        for config in row.get('IPAM', {}).get('Config') or []:
            subnet, gateway = config.get('Subnet'), config.get('Gateway')
            if not subnet or ':' in subnet:
                continue
            network = private_network(subnet)
            if not gateway:
                raise ValueError('Docker IPv4 bridge has no gateway: ' + row.get('Name', '?'))
            address = ipaddress.IPv4Address(gateway)
            if address not in network or address in (network.network_address, network.broadcast_address):
                raise ValueError('Invalid Docker bridge gateway')
            subnets.add(str(network))
            gateways.add(str(address))
    if not subnets or not any(upstream in ipaddress.ip_network(value) for value in subnets):
        raise ValueError('Upstream is not in the inspected local Docker bridges')
    if not 1024 <= port <= 65535:
        raise ValueError('Invalid frontend port')
    return {'policy': 'loopback-and-docker-only', 'client_api_key_required': False,
            'listen': ['127.0.0.1'] + sorted(gateways, key=ipaddress.IPv4Address),
            'allowed_docker_subnets': sorted(subnets), 'upstream': str(upstream),
            'upstream_port': 8000, 'port': port}


def validate(config):
    if config['policy'] != 'loopback-and-docker-only':
        raise ValueError('Unsupported access policy')
    networks = [private_network(value) for value in config['allowed_docker_subnets']]
    listeners = [ipaddress.IPv4Address(value) for value in config['listen']]
    if not networks or ipaddress.IPv4Address('127.0.0.1') not in listeners:
        raise ValueError('Loopback and inspected Docker subnets are required')
    for address in listeners:
        if str(address) != '127.0.0.1' and not any(address in network for network in networks):
            raise ValueError('Listener outside the Docker bridges')
    upstream = ipaddress.IPv4Address(config['upstream'])
    if not any(upstream in network for network in networks):
        raise ValueError('Upstream outside the Docker bridges')
    if not 1024 <= config['port'] <= 65535 or config['upstream_port'] != 8000:
        raise ValueError('Invalid port')
    return networks, [str(value) for value in listeners]


async def serve(config):
    networks, listeners = validate(config)

    async def handle(reader, writer):
        upstream_writer = None
        try:
            peer = ipaddress.IPv4Address(writer.get_extra_info('peername')[0])
            if not peer.is_loopback and not any(peer in network for network in networks):
                writer.write(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
                await writer.drain()
                return
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(config['upstream'], config['upstream_port']), timeout=10)

            async def copy(source, destination):
                while chunk := await asyncio.wait_for(source.read(65536), timeout=3600):
                    destination.write(chunk)
                    await destination.drain()
                if destination.can_write_eof():
                    destination.write_eof()

            await asyncio.gather(copy(reader, upstream_writer), copy(upstream_reader, writer))
        except (ConnectionError, OSError, asyncio.TimeoutError):
            pass
        finally:
            writer.close()
            if upstream_writer:
                upstream_writer.close()

    servers = []
    try:
        for address in listeners:
            servers.append(await asyncio.start_server(handle, address, config['port'], family=socket.AF_INET))
        print(json.dumps({'status': 'READY', **config}), flush=True)
        await asyncio.gather(*(server.serve_forever() for server in servers))
    finally:
        for server in servers:
            server.close()
        await asyncio.gather(*(server.wait_closed() for server in servers))


def check(base_url):
    base = base_url.rstrip('/')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + '/health', timeout=10) as response:
        assert response.status == 200
    with opener.open(base + '/v1/models', timeout=10) as response:
        ids = [row['id'] for row in json.load(response)['data']]
    assert ids == ['DeepSeek-V4.1-Flash'], ids
    payload = {'model': ids[0], 'input': 'Reply exactly LOCAL_ACCESS_READY.',
               'stream': True, 'store': False, 'max_output_tokens': 64,
               'chat_template_kwargs': {'thinking': False}}
    request = urllib.request.Request(base + '/v1/responses', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    completed, events = None, 0
    with opener.open(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]':
                continue
            row = json.loads(line[5:])
            events += 1
            assert row.get('type') != 'error' and not row.get('error'), row
            if row.get('type') == 'response.completed':
                completed = row['response']
    assert completed and completed['status'] == 'completed' and completed['model'] == ids[0]
    assert any(item.get('type') == 'message' and item.get('content') for item in completed['output'])
    for invalid in [
        {'model': 'invalid-alias', 'messages': [{'role': 'user', 'content': 'test'}]},
        {'model': ids[0], 'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'https://example.invalid/image.png'}}]}]},
    ]:
        request = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(invalid).encode(),
                                         headers={'Content-Type': 'application/json'})
        try:
            with opener.open(request, timeout=10) as response:
                raise AssertionError('Invalid model or remote media was not rejected')
        except urllib.error.HTTPError as exc:
            assert exc.code in (400, 404), exc.code
    return {'status': 'PASS', 'base_url': base, 'authorization_header_sent': False,
            'models': ids, 'responses_sse_events': events,
            'anonymous_responses': 'PASS', 'model_and_remote_media_rejection': 'PASS'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    make = sub.add_parser('configure')
    make.add_argument('--networks', type=Path, required=True)
    make.add_argument('--upstream', required=True)
    make.add_argument('--port', type=int, required=True)
    make.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('serve')
    run.add_argument('--config', type=Path, required=True)
    probe = sub.add_parser('check')
    probe.add_argument('--base-url', required=True)
    probe.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'configure':
        config = configure(json.loads(args.networks.read_text()), args.upstream, args.port)
        validate(config)
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(config, indent=2) + '\n')
        temp.chmod(0o644)
        temp.replace(args.output)
        print(json.dumps(config))
    elif args.action == 'serve':
        asyncio.run(serve(json.loads(args.config.read_text())))
    else:
        result = check(args.base_url)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
