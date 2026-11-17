"""Sheet text is literal, including filenames and labels that resemble math."""
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np
from PIL import Image
import pytest
import soundfile as sf

from earshot.sheet import sheet
from earshot.timeline import timeline


@pytest.mark.parametrize('where,text', [
    ('filename', 'take_$x_$.wav'),
    ('filename', 'take_$x^$.wav'),
    ('filename', 'take_$x$.wav'),
    ('filename', 'take_$5.wav'),
    ('section', '$x_$'),
    ('event', '$x^$'),
    ('kind', '$x_$'),
])
def test_sheet_preserves_literal_text(tmp_path, monkeypatch, where, text):
    p = tmp_path / (text if where == 'filename' else 'ordinary.wav')
    sf.write(p, np.zeros((48000, 2)), 48000)
    tl = timeline(str(p), cache=False)
    tl = copy.deepcopy(tl)
    if where == 'section':
        tl['sections'] = [{'start': 0., 'end': 1., 'label': text}]
    if where in ('event', 'kind'):
        tl['events'] = [{'t': 0.5, 'prominence_db': 20., 'exposed': True,
                       'kind': text if where == 'kind' else 'HIT',
                       'label': text if where == 'event' else 'plain event'}]
    before = mpl.rcParams['text.parse_math']
    before_figures = set(plt.get_fignums())
    original = Figure.savefig
    captured = []

    def save(fig, *args, **kwargs):
        captured.extend((t.get_text(), t.get_parse_math()) for t in fig.findobj(Text) if text in t.get_text())
        return original(fig, *args, **kwargs)
    monkeypatch.setattr(Figure, 'savefig', save)
    out = tmp_path / 'sheet.png'
    try:
        result = sheet(str(p), str(out), tl=tl, width=1000)
        assert captured and all(parse is False for _, parse in captured)
        with Image.open(out) as im:
            assert im.size == (result['width'], result['height'])
            im.verify()
        assert set(plt.get_fignums()) == before_figures
        assert mpl.rcParams['text.parse_math'] == before
    finally:
        for number in set(plt.get_fignums()) - before_figures:
            plt.close(number)
