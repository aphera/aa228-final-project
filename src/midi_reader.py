from mido import MidiFile, tempo2bpm


class Observation:
    def __init__(self, seconds_since_last_beat, seconds_since_start, actual_bpm=None):
        self.seconds_since_last_beat = seconds_since_last_beat
        self.seconds_since_start = seconds_since_start
        self.actual_bpm = actual_bpm


def get_observations():
    midi_file = MidiFile("cs1-1pre.mid")
    # midi_file = MidiFile("vs1-1ada.mid")
    tempo = 0
    seconds_since_last_beat = 0
    seconds_since_start = 0
    observations = []
    for msg in midi_file:
        seconds_since_start += msg.time
        if msg.type == "set_tempo":
            tempo = tempo2bpm(msg.tempo)
        if msg.type == "note_on" and msg.velocity > 0:
            observations.append(Observation(seconds_since_last_beat, seconds_since_start, tempo))
            seconds_since_last_beat = 0
        seconds_since_last_beat += msg.time
    return observations
