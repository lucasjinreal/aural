import sys
import os
from pydub import AudioSegment


if __name__ == '__main__':
    a = sys.argv[1]

    tgt_f = os.path.join(os.path.dirname(a), os.path.basename(a).split('.')[0] + '.mp3')
    sound = AudioSegment.from_file(a)
    sound.export(tgt_f, format='mp3')

    print(f'new audio file saved to: {tgt_f}')