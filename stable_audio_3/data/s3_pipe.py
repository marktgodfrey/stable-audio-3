import argparse
import os
import select
import signal
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream an S3 object to stdout with bounded AWS and idle timeouts."
    )
    parser.add_argument("--s3-path", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--cli-connect-timeout-sec", type=int, required=True)
    parser.add_argument("--cli-read-timeout-sec", type=int, required=True)
    parser.add_argument("--stream-timeout-sec", type=int, required=True)
    parser.add_argument("--stream-idle-timeout-sec", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, required=True)
    parser.add_argument("--retry-mode", required=True)
    return parser.parse_args()


def write_all(fd, data):
    view = memoryview(data)
    while len(view) > 0:
        written = os.write(fd, view)
        view = view[written:]


def terminate_process_group(process):
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError, PermissionError):
        process.terminate()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):
        process.kill()


def build_aws_cp_command(args):
    cmd = [
        "aws",
        "--cli-connect-timeout",
        str(args.cli_connect_timeout_sec),
        "--cli-read-timeout",
        str(args.cli_read_timeout_sec),
    ]

    if args.profile:
        cmd.extend(["--profile", args.profile])

    cmd.extend(["s3", "cp", args.s3_path, "-", "--only-show-errors"])
    return cmd


def stream_s3_object(args):
    env = os.environ.copy()
    env["AWS_MAX_ATTEMPTS"] = str(args.max_attempts)
    env["AWS_RETRY_MODE"] = args.retry_mode

    process = subprocess.Popen(
        build_aws_cp_command(args),
        stdout=subprocess.PIPE,
        stderr=None,
        env=env,
        start_new_session=True,
    )

    stdout_fd = process.stdout.fileno()
    output_fd = sys.stdout.fileno()
    last_progress_time = time.monotonic()

    try:
        while True:
            now = time.monotonic()
            idle_elapsed = now - last_progress_time

            if idle_elapsed > args.stream_idle_timeout_sec:
                terminate_process_group(process)
                raise TimeoutError(
                    f"S3 shard stream exceeded idle timeout of {args.stream_idle_timeout_sec}s "
                    f"for {args.s3_path}"
                )

            wait_timeout = min(1.0, args.stream_idle_timeout_sec - idle_elapsed)
            ready, _, _ = select.select([stdout_fd], [], [], max(wait_timeout, 0.0))
            if not ready:
                continue

            chunk = os.read(stdout_fd, 1024 * 1024)
            if chunk:
                write_all(output_fd, chunk)
                last_progress_time = time.monotonic()
                continue

            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"aws s3 cp exited with status {return_code} while streaming {args.s3_path}"
                )
            return
    finally:
        if process.poll() is None:
            terminate_process_group(process)


def main():
    args = parse_args()

    for key in (
        "cli_connect_timeout_sec",
        "cli_read_timeout_sec",
        "stream_timeout_sec",
        "stream_idle_timeout_sec",
        "max_attempts",
    ):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be greater than 0")

    if not args.retry_mode.strip():
        raise ValueError("retry_mode must be a non-empty string")

    stream_s3_object(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
