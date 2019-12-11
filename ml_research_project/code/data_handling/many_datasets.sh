for d in /l/raw_data/extracted/* ; do
    python create_dataset_new.py --source "$d" --target /l/data/ml_research_project/
done
