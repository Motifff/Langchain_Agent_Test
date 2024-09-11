import socket

# 设置监听的地址和端口
UDP_IP = "127.0.0.1"
UDP_PORT = 3000

# 创建一个UDP套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}")

while True:
    # 接收UDP数据包，缓冲区大小为1024字节
    data, addr = sock.recvfrom(1024)
    print(f"Received message: {data.decode('utf-8')} from {addr}")
