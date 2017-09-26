echo 'Recovering models from '$1
python inference.py --batch_size=1024 --category_csv_path=/home/datasets/cdiscount/category_names.csv --test_data_pattern=/home/datasets/cdiscount/test.tfrecord --num_threads=2 --train_model_dirs=$1
