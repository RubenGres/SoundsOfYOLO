import time
import mido
from mido import Message

# Open a MIDI output port
port = mido.open_output(mido.get_output_names()[0])

# Create and send MIDI messages in a loop
while True:
    port.send(Message('note_on', note=60, velocity=64, time=0))
    time.sleep(0.5)

    port.send(Message('note_off', note=60, velocity=64, time=0))
    time.sleep(0.5)
