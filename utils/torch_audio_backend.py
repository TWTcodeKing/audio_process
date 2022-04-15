import platform
import logging
import torchaudio

logger = logging.getLogger(__name__)



def get_torchaudio_backend():
    """Get the backend for torchaudio between soundfile and sox_io according to the os.

    Allow users to use soundfile or sox_io according to their os.

    Returns
    -------
    str
        The torchaudio backend to use.
    """
    current_system = platform.system()
    if current_system == "Windows":
        return "soundfile"
    else:
        return "sox_io"


def check_torchaudio_backend():
    """Checks the torchaudio backend and sets it to soundfile if
    windows is detected.
    """
    current_system = platform.system()
    if current_system == "Windows":
        logger.warn(
            "The torchaudio backend is switched to 'soundfile'. Note that 'sox_io' is not supported on Windows."
        )
        torchaudio.set_audio_backend("soundfile")
