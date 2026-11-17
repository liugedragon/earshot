"""Cached results retain the current caller's video path, using real media."""
import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from earshot import config, video
from earshot.timeline import timeline


@pytest.fixture(scope='module')
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp('video_paths')
    for name, color in [('original', 'black'), ('other', 'white')]:
        folder = root / name
        folder.mkdir()
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                        f'color=c={color}:s=160x90:r=25:d=2', '-c:v', 'mpeg4',
                        str(folder / 'film.mp4')], check=True, capture_output=True)
    (root / 'empty').mkdir()
    wav = root / 'original' / 'sound.wav'
    with wave.open(str(wav), 'wb') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(48000)
        f.writeframes(b'\0' * (48000 * 2 * 4))
    return root


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'EMB_CACHE', str(tmp_path / 'emb'))
    monkeypatch.setattr(config, 'DEC_CACHE', str(tmp_path / 'decoded'))


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('destination', ['other', 'empty'])
def test_cached_video_path_across_directories(media, monkeypatch, explicit, destination):
    monkeypatch.chdir(media / 'original')
    first = timeline('sound.wav', video='film.mp4') if explicit else timeline('film.mp4')
    assert first['cached'] is False
    monkeypatch.chdir(media / destination)
    current = str(media / 'original' / 'film.mp4')
    result = (timeline(str(media / 'original' / 'sound.wav'), video=current)
              if explicit else timeline(current))
    assert result['cached'] is True
    assert result['video'] == current
    assert float(np.mean(video.frame(result['video'], 0.5))) < 1


@pytest.mark.parametrize('explicit', [False, True])
def test_cached_video_same_directory(media, monkeypatch, explicit):
    monkeypatch.chdir(media / 'original')
    args = ('sound.wav',) if explicit else ('film.mp4',)
    options = {'video': 'film.mp4'} if explicit else {}
    first = timeline(*args, **options)
    second = timeline(*args, **options)
    assert first['cached'] is False and second['cached'] is True
    assert first['video'] == second['video'] == 'film.mp4'


@pytest.mark.parametrize('options', [{'cache': False}, {'fresh': True}])
@pytest.mark.parametrize('explicit', [False, True])
def test_uncached_video_path(media, monkeypatch, options, explicit):
    monkeypatch.chdir(media / 'original')
    if explicit:
        timeline('sound.wav', video='film.mp4')
    else:
        timeline('film.mp4')
    monkeypatch.chdir(media / 'other')
    current = str(media / 'original' / 'film.mp4')
    result = (timeline(str(media / 'original' / 'sound.wav'), video=current, **options)
              if explicit else timeline(current, **options))
    assert result['cached'] is False
    assert result['video'] == current


def test_cached_audio_only_has_no_video(media, monkeypatch):
    monkeypatch.chdir(media / 'original')
    first = timeline('sound.wav')
    monkeypatch.chdir(media / 'other')
    second = timeline(str(media / 'original' / 'sound.wav'))
    assert first['cached'] is False and second['cached'] is True
    assert first['video'] is None and second['video'] is None


def test_cache_record_video_is_not_authoritative(media):
    current = str(media / 'original' / 'film.mp4')
    timeline(current)
    record = next((Path(config.EMB_CACHE) / 'timeline').glob('*.json'))
    content = json.loads(record.read_text(encoding='utf-8'))
    content['video'] = 'an-old-caller-relative-path.mp4'
    record.write_text(json.dumps(content), encoding='utf-8')
    result = timeline(current)
    assert result['cached'] is True
    assert result['video'] == current
    # Reading a cache does not rewrite the stored result.
    assert json.loads(record.read_text(encoding='utf-8'))['video'] == content['video']


def test_sheet_uses_requested_video_after_cache_hit(media, monkeypatch, tmp_path):
    from PIL import Image
    from earshot.sheet import sheet

    monkeypatch.chdir(media / 'original')
    timeline('film.mp4')
    monkeypatch.chdir(media / 'other')
    current = str(media / 'original' / 'film.mp4')
    result = timeline(current)
    output = tmp_path / 'cached.png'
    control = tmp_path / 'control.png'
    rendered = sheet(current, str(output))
    expected = sheet(current, str(control), tl=dict(result, video=current))
    assert rendered['thumbnails'] == expected['thumbnails'] > 0
    np.testing.assert_array_equal(np.asarray(Image.open(output)), np.asarray(Image.open(control)))
