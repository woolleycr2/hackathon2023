from bluetooth import discover_device
import pyaudio
import wave

def discover_device(device_name):
    nearby_devices = discover_devices(lookup_names=True, lookup_class=True)
    
    for addr, name, _ in nearby_devices:
        if name == device_name:
            return addr
    return None

def connect_to_device(device_address):
    port = 1  # SPP profile
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    sock.connect((device_address, port))
    return sock

def play_audio_from_stream(audio_stream):
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(2),
                    channels=1,
                    rate=44100,
                    output=True)
    
    chunk_size = 1024
    data = audio_stream.read(chunk_size)
    
    while data:
        stream.write(data)
        data = audio_stream.read(chunk_size)
    
    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    device_name = "Samsung"
    
    device_address = discover_device(device_name)
    if device_address is None:
        print(f"Device '{device_name}' not found.")
        return
    
    print(f"Connecting to {device_name} ({device_address})...")
    socket = connect_to_device(device_address)
    
    print("Connected. Playing audio...")
    
    # Assume you have an audio file named "sample.wav" in the same directory
    audio_stream = wave.open("sample.wav", 'rb')
    
    play_audio_from_stream(audio_stream)
    
    # Close the Bluetooth socket
    socket.close()

if __name__ == "__main__":
    main()
