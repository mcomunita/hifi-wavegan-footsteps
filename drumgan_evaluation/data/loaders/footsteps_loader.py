# from drumgan_evaluation.data.loaders.base_loader import AudioDataLoader
# from drumgan_evaluation.data.db_extractors.footsteps_extractor import extract
from .base_loader import AudioDataLoader
from ..db_extractors.footsteps_extractor import extract

class FootstepsDataLoader(AudioDataLoader):
    def load_data(self):
        # print('-- FOOSTEPS DATALOADER: load_data')
        # print()
        self.data, self.metadata = \
            extract(self.data_path, self.criteria)
        # print('len(self.data):')
        # print(len(self.data))
        # print('len(self.metadata): ')
        # print(len(self.metadata))
        # print()
