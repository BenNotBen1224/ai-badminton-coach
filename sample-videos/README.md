# Sample clips

No video files are stored in this repo. Put clips here by hand, or fetch clips 04–10 with:

```powershell
.\.venv\Scripts\python.exe scripts\get_footage.py
```

| Clip | Source | Licence | Get it | Used for |
|---|---|---|---|---|
| 01–03 | 2026 YONEX Arise International Challenge, Day 3 Court 1 stream, by Arise Sports Complex (YouTube) | CC BY 4.0 | By hand; see `source-attribution.json` | Accuracy and speed checks |
| 04–08 | China Open 2025 women's singles final, Badminton Sphere, via Wikimedia Commons | CC BY 3.0 | `get_footage.py` | 04–05: accuracy and speed checks; 04: the sample review; 08: held-out check |
| 09–10 | The same final, later parts | CC BY 3.0 | `get_footage.py` | Extra test clips |
| 11 | EuroGames Bern 2023, recreational hall, by Hyruspex via Wikimedia Commons | CC BY-SA 4.0 | By hand; see `11-eurogames-attribution.json` | Edge case: not a broadcast view |

- `get_footage.py` checks the downloaded match file against its recorded hash, then cuts each clip from the same start time with the same settings as the originals. The cuts look the same (average pixel difference under 2 on a 0–255 scale) but are **not byte-identical**, so re-measured numbers can differ slightly from the records.
- The upstream broadcast rights for the China Open footage were not independently cleared. No endorsement by the players, venues, tournaments or BWF is implied.
- Clip 08 is `08-china-open-ws-f.mp4` (source 250–280 s). An earlier window, `08-china-open-ws-e.mp4` (240–270 s), was rejected before scoring because it opens on a side-camera replay.
