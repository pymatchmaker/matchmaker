from typing import List

import mido
from matchmaker.utils.symbolic import MidiDeviceInfo


def get_midi_devices() -> List[MidiDeviceInfo]:
    """Get list of MIDI devices
    Returns
    -------
    midi_devices : List[MidiDeviceInfo]
        List of available MIDI devices
    """
    try:
        available_in_ports = mido.get_input_names()
        available_out_ports = mido.get_output_names()
    except Exception as e:
        print(f"Error getting MIDI devices: {e}")
        available_in_ports = []
        available_out_ports = []

    all_devices = list(set(available_in_ports + available_out_ports))
    all_devices.sort()

    midi_devices = []
    for i, device in enumerate(all_devices):
        has_input = device in available_in_ports
        has_output = device in available_out_ports

        midi_device = MidiDeviceInfo(
            name=device,
            device_index=i,
            has_input=has_input,
            has_output=has_output,
        )

        midi_devices.append(midi_device)

    return midi_devices


def get_available_midi_port(port: str = None, is_virtual: bool = False) -> str:
    """
    Get the available MIDI port. If a port is specified, check if it is available.

    Parameters
    ----------
    port : str, optional
        Name of the MIDI port (default is None).

    Returns
    -------
    MidiInputPort
        Available MIDI input port

    Raises
    ------
    RuntimeError
        If no MIDI input ports are available.
    ValueError
        If the specified MIDI port is not available.
    """

    if port is None and is_virtual:
        raise ValueError("Cannot open unspecified virtual port!")
    input_names = mido.get_input_names()
    if not input_names and not is_virtual:
        raise RuntimeError("No MIDI input ports available")

    if port is None and not is_virtual:
        return input_names[0]
    elif port in input_names or is_virtual:
        return port
    else:
        raise ValueError(
            f"Specified MIDI port '{port}' is not available. Available ports: {input_names}"
        )


def check_output_midi_devices() -> bool:
    """
    Check whether the system has MIDI devices with output
    """
    midi_devices = get_midi_devices()
    return len(midi_devices) > 0
