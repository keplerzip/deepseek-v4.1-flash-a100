"""Forward the LAN port to one fixed internal Docker IPv4 address; no DNS or downloads."""
import argparse
import asyncio
import ipaddress


async def serve(args):
    upstream = str(ipaddress.IPv4Address(args.upstream))
    listen = str(ipaddress.IPv4Address(args.listen))
    if not ipaddress.IPv4Address(upstream).is_private:
        raise ValueError('Upstream must be a private Docker address')

    async def handle(reader, writer):
        peer_writer = None
        try:
            peer_reader, peer_writer = await asyncio.wait_for(asyncio.open_connection(upstream, args.upstream_port), timeout=10)

            async def copy(source, destination):
                while chunk := await asyncio.wait_for(source.read(65536), timeout=3600):
                    destination.write(chunk)
                    await destination.drain()
                if destination.can_write_eof():
                    destination.write_eof()

            await asyncio.gather(copy(reader, peer_writer), copy(peer_reader, writer))
        except (ConnectionError, OSError, asyncio.TimeoutError):
            pass
        finally:
            writer.close()
            if peer_writer:
                peer_writer.close()
    server = await asyncio.start_server(handle, listen, args.port)
    print('Local API frontend ready', flush=True)
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', required=True)
    p.add_argument('--upstream-port', type=int, default=8000)
    p.add_argument('--listen', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8005)
    asyncio.run(serve(p.parse_args()))
