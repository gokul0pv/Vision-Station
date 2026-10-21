"""Talk to QuickBuild over TCP/IP, the way a PLC or robot would -- to check the
connection from the line PC before wiring the controller.

    python example_line_client.py                       localhost, port 3000
    python example_line_client.py 192.168.0.20 3000 "TRIGGER" "GET frt_top.hole_spacing.value_px"

Turn the interface on first: QuickBuild > Run > Communication.
"""
import socket
import sys

host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
commands = sys.argv[3:] or ["PING", "STATUS", "TRIGGER"]

with socket.create_connection((host, port), timeout=120) as s:
    f = s.makefile("rwb")
    for cmd in commands:
        f.write((cmd + "\r\n").encode("ascii"))
        f.flush()
        print(">>", cmd)
        trigger = cmd.upper().split()[0] in ("TRIGGER", "T")
        while True:
            line = f.readline().decode("ascii").strip()
            print("<<", line)
            if not trigger or line.startswith(("DONE", "BUSY", "ERROR")):
                break
