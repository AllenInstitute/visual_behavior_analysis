import logging
import os


logger = logging.getLogger(__name__)


class Epoch(object):

    def __init__(
        self,
        movie_path,
        movie_index,
        start_frame,
        frame_times,
        movie_name=None,  # we will try to supply this in the future...
    ):
        self.__movie_index = movie_index
        self.__start_frame = start_frame
        self.__stop_frame = start_frame
        self.__movie_path = movie_path
        self.__frame_times = frame_times
        if not movie_name:
            movie_name = os.path.splitext(
                os.path.basename(self.__movie_path),
            )[0]
        self.__movie_name = movie_name

    def extend_epoch(self, stop_frame):
        self.__stop_frame = stop_frame

    def dump(self):
        # the dict is the expected shape of an image epoch
        if self.__movie_index == -1:  # grayscreen
            image_name = 'grayscreen'
            image_category = 'grayscreen'
        else:  # not grayscreen
            image_name = '{}:{}'.format(
                self.__movie_name,
                self.__movie_index,
            )
            image_category = self.__movie_name

        return {
            'image_name': image_name,
            'image_category': image_category,  # this is the best compromise i think we can make at this time :(...
            'frame': self.__start_frame,
            'stop_frame': self.__stop_frame,
            "orientation": None,
            "time": self.__frame_times[self.__start_frame],
            'duration': self.__frame_times[self.__stop_frame],
        }

    @property
    def movie_index(self):
        return self.__movie_index


def get_movie_image_epochs(movie_name, movie_item, frame_times, ):
    """extracts movie presentation periods as a list of image stimulus
    epochs
    """
    movie_path = movie_item['static_stimulus']['movie_path']
    frame_list = movie_item['static_stimulus']['frame_list']
    sweep_frames = range(
        movie_item['starting_frame'],
        movie_item['ending_frame'],
    )  # pretty sure these bounds are correct

    epochs = []
    # hopefully these values dont get decoupled improperly?
    epoch = Epoch(
        movie_path=movie_path,
        movie_index=frame_list[0],
        start_frame=sweep_frames[0],
        frame_times=frame_times,
        movie_name=movie_name,
    )
    for sweep_frame_index, movie_frame_index in \
            zip(sweep_frames[1:], frame_list[1:]):
        if movie_frame_index == epoch.movie_index:
            epoch.extend_epoch(sweep_frame_index)
        else:
            epochs.append(epoch.dump())
            epoch = Epoch(
                movie_path=movie_path,
                movie_index=movie_frame_index,
                start_frame=sweep_frame_index,
                frame_times=frame_times,
            )

    epochs.append(epoch.dump())

    return epochs


def get_movie_metadata(data):
    MAYBE_A_MOVIE = [
        'countdown',
        'fingerprint',
    ]

    try:
        items = data['items']['behavior'].get('items', {})
    except KeyError:
        items = data.get('items', {})

    movie_metadata = {}  # then update with all the movie flavours where they occur...

    # static movie metadata
    movie_metadata.update({
        k: v['static_stimulus']['movie_path']
        for k, v in items.items()
        if k.lower() in MAYBE_A_MOVIE
    })

    return movie_metadata
