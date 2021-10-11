# from .nsynth import NSynth
# from .mtg_drums import MTGDrums
# from .csl_drums import CSLDrums
# from .youtube_pianos import YouTubePianos
# from .sinewaves import Sinewaves
# from .mp3towav_loader import MP3ToWAV
# from drumgan_evaluation.data.loaders.footsteps_loader import FootstepsDataLoader
# from drumgan_evaluation.data.loaders.base_loader import SimpleLoader
# from loaders.footsteps_loader import FootstepsDataLoader
from .base_loader import SimpleLoader
from .footsteps_loader import FootstepsDataLoader

AVAILABLE_DATASETS = {
	# 'nsynth': NSynth,
	# 'mtg-drums': MTGDrums,
	# 'csl-drums': CSLDrums,
	# 'youtube-pianos': YouTubePianos,
	# 'sinewaves': Sinewaves,
	# 'mp3towav': MP3ToWAV,
	'footsteps': FootstepsDataLoader,
	'default': SimpleLoader
}

def get_data_loader(name):
    if name not in AVAILABLE_DATASETS:
    	raise AttributeError(f"Invalid module name. \
                               Available: {AVAILABLE_DATASETS.keys()}")

    return AVAILABLE_DATASETS[name]
