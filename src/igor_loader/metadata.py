def parse_ibw_note(ibw_dict):
    """
    Parse the IGOR wave note into a Python dictionary.
    """
    note_raw = ibw_dict['wave']['note']

    # Decode if necessary
    if isinstance(note_raw, bytes):
        note_raw = note_raw.decode('utf-8', errors='ignore')

    note_dict = {}

    # Common separators: newline or semicolon
    for line in note_raw.replace(';', '\n').splitlines():
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()

        # Try to cast to float or int when possible
        try:
            if '.' in value or 'e' in value.lower():
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass

        note_dict[key] = value

    return note_dict
        
def build_metadata_from_ibw(ibw_dict):
    note = parse_ibw_note(ibw_dict)

    metadata = {}

    # ======================
    # General
    # ======================
    wave_name = ibw_dict['wave']['wave_header']['bname'].decode('utf-8', errors='ignore')
    metadata['entry_title'] = wave_name

    # ======================
    # Instrument
    # ======================
    metadata['instrument'] = {}

    # ----------------------
    # Beam (probe)
    # ----------------------
    metadata['instrument']['beam'] = {}
    metadata['instrument']['beam']['probe'] = {}

    # Generic energy mappings
    metadata['instrument']['beam']['probe']['incident_energy'] = note.get('Eph')
    metadata['instrument']['beam']['probe']['electron_kinetic_energy'] = note.get('Ek')
    metadata['instrument']['beam']['probe']['pass_energy'] = note.get('Ep')

    # Optional / may not exist
    metadata['instrument']['beam']['probe']['pulse_duration'] = note.get('PulseDuration')
    metadata['instrument']['beam']['probe']['frequency'] = note.get('Frequency')
    metadata['instrument']['beam']['probe']['incident_polarization'] = None
    metadata['instrument']['beam']['probe']['extent'] = None

    # ----------------------
    # Analyzer
    # ----------------------
    metadata['instrument']['electronanalyzer'] = {
        'lens_mode': note.get('LensMode'),
        'energy_resolution': None,
        'angular_resolution': None,
        'spatial_resolution': None,
    }

    # ----------------------
    # Detector / acquisition
    # ----------------------
    metadata['instrument']['detector'] = {
        'Udet': note.get('Udet'),
        'Uscr': note.get('Uscr'),
        'Icoil': note.get('Icoil'),
    }

    metadata['instrument']['acquisition'] = {
        'dwell_time': note.get('DwellTime'),
        'exposures': note.get('Exposures'),
        'num_images': note.get('NumOfImages'),
        'pixel_sum': note.get('PixelSum'),
    }

    # ----------------------
    # Vacuum
    # ----------------------
    metadata['instrument']['vacuum_pressure'] = note.get('MCPress')

    # ======================
    # Sample
    # ======================
    metadata['sample'] = {}

    # Temperatures (generic handling)
    temperatures = {}
    for key in note:
        if key.upper().startswith('T_'):
            temperatures[key] = note[key]
    if temperatures:
        metadata['sample']['temperature'] = temperatures

    # Position
    metadata['sample']['position'] = {
        'X': note.get('X'),
        'Y': note.get('Y'),
        'Z': note.get('Z'),
        'Theta': note.get('Theta'),
        'Phi': note.get('Phi'),
        'Omega': note.get('Omega')
    }

    # ======================
    # ROI
    # ======================
    roi_keys = ('ROIX1', 'ROIX2', 'ROIY1', 'ROIY2')
    if any(k in note for k in roi_keys):
        metadata['instrument']['roi'] = {
            'x1': note.get('ROIX1'),
            'x2': note.get('ROIX2'),
            'y1': note.get('ROIY1'),
            'y2': note.get('ROIY2'),
        }

    # ======================
    # Raw note (full provenance)
    # ======================
    metadata['raw_note'] = note

    return metadata