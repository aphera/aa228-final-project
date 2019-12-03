from mido import MidiFile, MetaMessage, Message

# midi_file = MidiFile("01prelud.mid")
midi_file = MidiFile("var8.mid")
length = 0
for msg in midi_file:
    length += msg.time
    if msg.type == "note_on" and msg.velocity > 0:
    # if isinstance(msg, Message):
        print(msg)

print(length)
