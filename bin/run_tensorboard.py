import subprocess
import argparse

from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-to', '--', dest='dur', type=float, default=60.,
                        help='timeout duration')
    parser.add_argument('-d', '--logdir', dest='log_dir', type=str, default='.', 
                        help='tensorboard log directory')
    parser.add_argument('-p', '--port', dest='port', type=int, default=8787,
                        help="port to run on")
    parser.add_argument('-h', '--host', dest='host', default='0.0.0.0', type=str,
                        help="host IP address for tensorboard server")
    parser.parse_args()
    
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise ValueError(f"log directory {log_dir} doesn't exist!")
    
    cmd = f"""
    while true; do
    timeout -sHUP {dur}s tensorboard --logdir={args.log_dir} --port={args.port} --host={args.host};
    done
    """

    subprocess.Popen(cmd)