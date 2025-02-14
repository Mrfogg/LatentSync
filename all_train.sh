rm -rf high_visual_quality/train && mkdir high_visual_quality/train &&
rm -rf high_visual_quality/val && mkdir high_visual_quality/val &&

./data_processing_pipeline.sh &&
python sample.py &&
python tools/write_fileslist.py &&
python latentsync/whisper/audio2feature.py &&
./train_syncnet.sh && ./train_unet.sh

