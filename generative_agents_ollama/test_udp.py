import socket
import json

UDP_IP = "localhost"
UDP_PORT = 6666

def send_udp_message():
    # Create a UDP socket for sending messages
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        command = input("Enter command to send (or 'quit' to exit): ")
        if command.lower() == 'quit':
            break
        
        message = json.dumps({"command": command})
        sock.sendto(message.encode('utf-8'), (UDP_IP, UDP_PORT))
        print(f"Message '{message}' sent to {UDP_IP}:{UDP_PORT}")

if __name__ == "__main__":
    send_udp_message()
