import time
import mido
from mido import Message

try:
    # Get available output ports
    output_ports = mido.get_output_names()
    
    if not output_ports:
        print("No MIDI output ports available. Please connect a MIDI device or install a virtual MIDI port.")
        exit()
    
    # Display available ports with numbers
    print("Available MIDI output ports:")
    for i, port_name in enumerate(output_ports):
        print(f"[{i}] {port_name}")
    
    # Get user input for port selection
    while True:
        selection = input("\nSelect a port number: ")
        try:
            port_index = int(selection)
            if 0 <= port_index < len(output_ports):
                selected_port = output_ports[port_index]
                break
            else:
                print(f"Please enter a number between 0 and {len(output_ports)-1}")
        except ValueError:
            print("Please enter a valid number")
    
    # Open the selected port
    print(f"Opening MIDI port: {selected_port}")
    port = mido.open_output(selected_port)
    print(f"Successfully opened MIDI port: {selected_port}")
    
    # Create and send MIDI messages in a loop
    print("Sending MIDI messages. Press Ctrl+C to stop.")
    try:
        while True:
            port.send(Message('note_on', note=60, velocity=64))
            print("Sent note_on message")
            time.sleep(0.5)
            
            port.send(Message('note_off', note=60, velocity=64))
            print("Sent note_off message")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping MIDI output")
    finally:
        port.close()
        print("MIDI port closed")
        
except Exception as e:
    print(f"Error: {e}")