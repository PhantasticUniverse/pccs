# Placeholder for Demo Animation

This file will be replaced with an actual simulation GIF after running:

```bash
python -m pccs.main --save-animation docs/assets/pccs_demo.gif --steps 500 --grid-size 128
```

Or generate frames:

```bash
python -m pccs.main --save-frames docs/assets/frames/ --save-interval 10 --steps 1000
```

Then convert to GIF using ffmpeg or ImageMagick:

```bash
ffmpeg -framerate 30 -pattern_type glob -i 'docs/assets/frames/*_composite.png' -vf "scale=512:-1" docs/assets/pccs_demo.gif
```
