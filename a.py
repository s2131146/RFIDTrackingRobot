import socket

server_ip = '192.168.10.33'
server_port = 8001

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

try:
    while True:
        data = client_socket.recv(1024)
        if not data:
            continue
        print(data.decode())
except Exception as e:
    print(f"Error: {e}")
finally:
    client_socket.close()
