import socket

def send_udp_message(message):
    udp_ip = "localhost"
    udp_port = 6666

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Encode the message to bytes and send it to the server
    sock.sendto(message.encode('utf-8'), (udp_ip, udp_port))
    print(f"Message '{message}' sent to {udp_ip}:{udp_port}")

# Test the listener by sending the "start" message
if __name__ == "__main__":
    send_udp_message("start")
